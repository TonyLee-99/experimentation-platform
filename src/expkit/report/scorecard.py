"""Automated experiment scorecard: config in, HTML readout out.

Pipeline: metric catalog -> compiled SQL (sufficient stats) -> SRM + balance checks -> raw and
variance-reduced estimates -> BH across primary/secondary -> guardrails (non-inferiority) ->
Bayesian expected-loss view -> sequential monitoring chart -> decision framework -> HTML.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from scipy import stats

from expkit.config import ExperimentConfig
from expkit.data import ROOT
from expkit.decision import Decision, GuardrailResult, decide
from expkit.metrics.compiler import AdjustmentRequest, VariantStats, cumulate, run, run_segmented
from expkit.metrics.spec import MetricCatalog
from expkit.stats.bayes import BayesDecision, bayes_decision
from expkit.stats.cuped import cuped, variance_reduction
from expkit.stats.delta import delta_ratio
from expkit.stats.multiple import bh_adjust
from expkit.stats.sequential import gsd_boundaries, msprt_pvalues, msprt_z_boundary
from expkit.stats.srm import SRMResult, srm_check
from expkit.stats.ttest import welch
from expkit.stats.types import Estimate

COLORS = {
    "series1": "#2a78d6",
    "series2": "#eb6834",
    "series3": "#1baf7a",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e4e3df",
}


@dataclass
class Balance:
    chi2: float
    df: int
    p_value: float
    max_abs_smd: float
    worst: str
    smd: dict[str, float]


@dataclass
class Readout:
    config: ExperimentConfig
    srm: SRMResult
    balance: Balance | None
    rows: list[dict[str, Any]]
    guardrails: list[GuardrailResult]
    bayes: BayesDecision
    decision: Decision
    sequential_png: str
    sequential_summary: dict[str, Any]
    n: dict[str, float]
    notes: list[str] = field(default_factory=list)


def _where(cfg: ExperimentConfig, variant_col: str, unit_col: str) -> str | None:
    bug = cfg.simulated_bug
    if bug is None:
        return None
    drop_per_10k = round(bug.drop_share * 10000)
    return f"NOT ({variant_col} = {bug.variant} AND ({bug.condition}) AND hash({unit_col}) % 10000 < {drop_per_10k})"


def balance_check(c: VariantStats, t: VariantStats, metric: str, covs: list[str]) -> Balance:
    """Joint test that pre-treatment covariate means are equal across arms (chi-square, k df).

    Uses the same Σx, Σxxᵀ sufficient statistics as CUPED, so it costs nothing extra.
    """
    cc, tc = c.covariate(metric, covs), t.covariate(metric, covs)

    def mean_cov(s: Any) -> tuple[np.ndarray, np.ndarray]:
        m = s.sum_x / s.n
        return m, (s.sum_xx - s.n * np.outer(m, m)) / (s.n - 1)

    mc, Sc = mean_cov(cc)
    mt, St = mean_cov(tc)
    d = mt - mc
    V = Sc / cc.n + St / tc.n
    chi2 = float(d @ np.linalg.solve(V, d))
    pooled_sd = np.sqrt(0.5 * (np.diag(Sc) + np.diag(St)))
    smd = d / pooled_sd
    worst = int(np.argmax(np.abs(smd)))
    return Balance(
        chi2,
        len(covs),
        float(stats.chi2.sf(chi2, len(covs))),
        float(np.abs(smd).max()),
        covs[worst],
        {k: float(v) for k, v in zip(covs, smd, strict=True)},
    )


def _estimate(cat: MetricCatalog, name: str, c: VariantStats, t: VariantStats, alpha: float) -> Estimate:
    m = cat.get(name)
    if m.type == "ratio":
        return delta_ratio(c.ratio(name), t.ratio(name), name, alpha)
    return welch(c.mean(name), t.mean(name), name, alpha)


def _sequential(cfg: ExperimentConfig, cat: MetricCatalog, where: str | None) -> tuple[str, dict[str, Any]]:
    """Primary metric monitored over K looks in a SIMULATED random arrival order."""
    seq = cfg.sequential
    metric = cfg.primary_metric
    src = cat.source
    seg = f"(hash({src.unit} + {seq.seed}) % {seq.looks})"
    covset = cfg.variance_reduction.get(metric)
    adj = [AdjustmentRequest(metric, covset)] if covset else []
    parts = run_segmented(cat, [metric], seg, adj, where)
    looks = list(range(seq.looks))
    ctrl, trt = _coerce(cfg.control), _coerce(cfg.treatment)
    z, var, delta, n_cum = [], [], [], []
    for k in looks:
        c = cumulate([parts[(ctrl, j)] for j in looks[: k + 1]])
        t = cumulate([parts[(trt, j)] for j in looks[: k + 1]])
        if covset:
            covs = cat.covariate_sets[covset]
            e = cuped(c.covariate(metric, covs), t.covariate(metric, covs), metric)
        else:
            e = welch(c.mean(metric), t.mean(metric), metric)
        z.append(e.delta / e.se)
        var.append(e.se**2)
        delta.append(e.delta)
        n_cum.append(c.n + t.n)
    z_a, v_a = np.array(z), np.array(var)
    frac = np.array(n_cum) / n_cum[-1]
    base = parts[(ctrl, 0)].mean(metric).mean
    tau2 = (seq.tau_relative * base) ** 2
    obf = gsd_boundaries(frac, cfg.decision.alpha).z_bounds
    ms_b = msprt_z_boundary(v_a, tau2, cfg.decision.alpha)
    p_ms = msprt_pvalues(np.array(delta), v_a, tau2)
    naive = float(stats.norm.ppf(1 - cfg.decision.alpha / 2))

    def first(hit: np.ndarray) -> int | None:
        idx = np.flatnonzero(hit)
        return int(idx[0]) + 1 if idx.size else None

    summary = {
        "looks": seq.looks,
        "first_look_naive": first(np.abs(z_a) >= naive),
        "first_look_obf": first(np.abs(z_a) >= obf),
        "first_look_msprt": first(p_ms < cfg.decision.alpha),
        "final_z": float(z_a[-1]),
        "tau_relative": seq.tau_relative,
        "label": "SIMULATED arrival order (hash of unit id): the dataset has no timestamps",
    }

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=130)
    x = np.arange(1, seq.looks + 1)
    ax.plot(x, np.abs(z_a), color=COLORS["series1"], lw=2, marker="o", ms=4, label=f"|z| of {metric}")
    ax.plot(x, obf, color=COLORS["series2"], lw=2, label="O'Brien-Fleming boundary")
    ax.plot(x, ms_b, color=COLORS["series3"], lw=2, ls="--", label="mSPRT boundary")
    ax.axhline(naive, color=COLORS["ink2"], lw=1, ls=":", label="naive 1.96 (invalid if peeking)")
    top = max(float(np.nanmax(np.abs(z_a))) * 1.5, 10.0)
    ax.set_yscale("log")
    ax.set_ylim(1, top)
    ax.set_xticks(x)
    ax.set_xlabel("look k (each look adds 1/K of the units)")
    ax.set_ylabel("|z| (log scale)")
    if not np.isfinite(obf[0]):
        ax.annotate(
            "OBF spends ~0 α at look 1 (boundary off-chart)",
            (1, 1.3),
            fontsize=7,
            color=COLORS["ink2"],
        )
    ax.grid(color=COLORS["grid"], lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode(), summary


def _coerce(v: int | str) -> Any:
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def analyze(cfg: ExperimentConfig) -> Readout:
    cat = MetricCatalog.from_yaml(ROOT / cfg.metrics_catalog)
    cat.source.table = str(ROOT / cat.source.table)
    for j in cat.source.joins:
        j.table = str(ROOT / j.table)
    alpha = cfg.decision.alpha
    metric_names = [cfg.primary_metric, *cfg.secondary_metrics, *[g.metric for g in cfg.guardrails]]
    adjustments = [AdjustmentRequest(m, s) for m, s in cfg.variance_reduction.items()]
    adjustments += [AdjustmentRequest(m, s) for m, sets in cfg.compare_covariate_sets.items() for s in sets]
    if cfg.balance_covariate_set:
        adjustments.append(AdjustmentRequest(cfg.primary_metric, cfg.balance_covariate_set))
    where = _where(cfg, cat.source.variant, cat.source.unit)
    stats_by_variant = run(cat, metric_names, adjustments, where, duckdb.connect())
    ctrl, trt = _coerce(cfg.control), _coerce(cfg.treatment)
    c, t = stats_by_variant[ctrl], stats_by_variant[trt]
    srm = srm_check([c.n, t.n], [cfg.expected_split[str(ctrl)], cfg.expected_split[str(trt)]])

    balance, balance_warning = None, None
    if cfg.balance_covariate_set:
        balance = balance_check(c, t, cfg.primary_metric, cat.covariate_sets[cfg.balance_covariate_set])
        if balance.p_value < cfg.decision.balance_p_threshold:
            balance_warning = (
                f"Covariate imbalance: pre-treatment features differ across arms (chi2 = {balance.chi2:,.0f}, "
                f"{balance.df} df, p = {balance.p_value:.1e}; worst SMD {balance.smd[balance.worst]:+.3f} on "
                f"{balance.worst}). Raw differences are biased; the regression-adjusted estimate is used "
                "for the decision."
            )

    rows: list[dict[str, Any]] = []
    decision_estimates: dict[str, Estimate] = {}
    for name in [cfg.primary_metric, *cfg.secondary_metrics]:
        raw = _estimate(cat, name, c, t, alpha)
        row: dict[str, Any] = {
            "metric": name,
            "role": "primary" if name == cfg.primary_metric else "secondary",
            "raw": raw,
            "adjusted": None,
            "vr": None,
        }
        covset = cfg.variance_reduction.get(name)
        if covset and cat.get(name).type == "mean":
            covs = cat.covariate_sets[covset]
            cs_c, cs_t = c.covariate(name, covs), t.covariate(name, covs)
            row["adjusted"] = cuped(cs_c, cs_t, name, alpha, method=f"cuped[{covset}]")
            row["vr"] = variance_reduction(cs_c, cs_t)
        row["vr_compare"] = {}
        for extra in cfg.compare_covariate_sets.get(name, []):
            covs = cat.covariate_sets[extra]
            row["vr_compare"][extra] = variance_reduction(c.covariate(name, covs), t.covariate(name, covs))
            row.setdefault("compare_estimates", {})[extra] = cuped(
                c.covariate(name, covs),
                t.covariate(name, covs),
                name,
                alpha,
                method=f"cuped[{extra}]",
            )
        decision_estimates[name] = row["adjusted"] or raw
        rows.append(row)
    qs = bh_adjust([decision_estimates[r["metric"]].p_value for r in rows])
    for r, q in zip(rows, qs, strict=True):
        r["q"] = float(q)

    guardrails = [
        GuardrailResult(g.metric, _estimate(cat, g.metric, c, t, alpha), g.max_relative_drop) for g in cfg.guardrails
    ]
    primary = decision_estimates[cfg.primary_metric]
    base = c.mean(cfg.primary_metric).mean
    bayes = bayes_decision(
        primary.delta,
        primary.se,
        cfg.decision.loss_threshold_relative * base,
        0.0,
        cfg.decision.prior_sd_relative * base,
    )
    decision = decide(srm.blocked, srm.p_value, guardrails, primary, rows[0]["q"], alpha, balance_warning)
    png, seq_summary = _sequential(cfg, cat, where)
    notes = []
    if cfg.simulated_bug:
        notes.append(
            f"SIMULATION: {cfg.simulated_bug.drop_share:.1%} of variant {cfg.simulated_bug.variant} units with "
            f"({cfg.simulated_bug.condition}) were dropped to emulate a logging bug."
        )
    return Readout(
        cfg,
        srm,
        balance,
        rows,
        guardrails,
        bayes,
        decision,
        png,
        seq_summary,
        {"control": c.n, "treatment": t.n},
        notes,
    )


def render(readout: Readout, path: str | Path) -> Path:
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["pct"] = lambda v, d=2: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.{d}%}"
    env.filters["num"] = lambda v, d=4: "—" if v is None else f"{v:.{d}g}"
    env.filters["p"] = lambda v: "—" if v is None else ("<1e-300" if v == 0 else f"{v:.2g}")
    html = env.get_template("readout.html.j2").render(r=readout, cfg=readout.config)
    out = Path(path)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


def summary_json(r: Readout) -> dict[str, Any]:
    """Machine-readable summary of a readout (used by acceptance checks and docs)."""
    vr: dict[str, float] = {}
    metrics: dict[str, Any] = {}
    for row in r.rows:
        if row["vr"] is not None:
            vr[f"{row['metric']}_cupac"] = row["vr"]
        for k, v in row.get("vr_compare", {}).items():
            vr[f"{row['metric']}_{k}"] = v
        ests = {
            "raw": row["raw"],
            **({"adjusted": row["adjusted"]} if row["adjusted"] else {}),
            **row.get("compare_estimates", {}),
        }
        metrics[row["metric"]] = {
            k: {
                "delta": e.delta,
                "se": e.se,
                "rel_lift": e.rel_lift,
                "rel_ci": [e.rel_ci_low, e.rel_ci_high],
                "p": e.p_value,
                "method": e.method,
            }
            for k, e in ests.items()
        }
        metrics[row["metric"]]["bh_q"] = row["q"]
    return {
        "name": r.config.name,
        "verdict": r.decision.verdict,
        "rule": r.decision.rule,
        "reason": r.decision.reason,
        "warnings": r.decision.warnings,
        "srm_p": r.srm.p_value,
        "balance": None
        if r.balance is None
        else {"chi2": r.balance.chi2, "p": r.balance.p_value, "max_abs_smd": r.balance.max_abs_smd},
        "variance_reduction": vr,
        "metrics": metrics,
        "guardrails": [
            {
                "metric": g.metric,
                "status": g.status,
                "rel_lift": g.estimate.rel_lift,
                "rel_ci": [g.estimate.rel_ci_low, g.estimate.rel_ci_high],
            }
            for g in r.guardrails
        ],
        "bayes_panel": True,
        "bayes_verdict": r.bayes.verdict,
        "bayes_loss_ship": r.bayes.loss_ship,
        "bayes_prob_positive": r.bayes.prob_positive,
        "sequential": r.sequential_summary,
    }
