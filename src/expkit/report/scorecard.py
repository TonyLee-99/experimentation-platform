"""Automated experiment scorecard: config in, HTML readout out.

Two sources:
* ``warehouse`` — unit-level table → compiled SQL (sufficient stats) → SRM + balance checks → raw,
  regression-adjusted and post-stratified estimates → BH → guardrails → economics → decision rules.
* ``summary`` — experiments published only as per-arm (n, mean, variance) snapshots (ASOS). The same
  decision rules run on the final snapshot; the sequential panel uses the *real* daily snapshots.

Panels that carry no information for a given decision (a Bayesian panel with P(effect > 0) = 1, a
sequential chart built on a simulated arrival order) are replaced by a one-line "N/A" with the reason.
"""

from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader
from scipy import stats

from expkit.config import ExperimentConfig
from expkit.data import ROOT
from expkit.db import duckdb_connect
from expkit.decision import Decision, GuardrailResult, decide, fmt_p, primary_rule
from expkit.metrics.compiler import AdjustmentRequest, VariantStats, cumulate, run, run_segmented
from expkit.metrics.spec import MetricCatalog
from expkit.stats.bayes import BayesDecision, bayes_decision
from expkit.stats.bootstrap import BootstrapConfig, bootstrap_diff, needs_bootstrap
from expkit.stats.cuped import cuped, variance_reduction
from expkit.stats.delta import delta_ratio
from expkit.stats.multiple import bh_adjust
from expkit.stats.sequential import gsd_boundaries, msprt_pvalues, msprt_z_boundary
from expkit.stats.srm import SRMResult, srm_check
from expkit.stats.ttest import relative_ci, welch
from expkit.stats.types import Estimate, MeanStats

COLORS = {"series1": "#2a78d6", "series2": "#eb6834", "series3": "#1baf7a", "ink2": "#52514e", "grid": "#e4e3df"}
Z975 = float(stats.norm.ppf(0.975))


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
    diagnostics: list[Estimate]
    robustness: list[dict[str, Any]]
    economics: dict[str, Any] | None
    bayes: BayesDecision
    bayes_informative: bool
    decision: Decision
    naive: dict[str, Any]
    sequential_png: str | None
    sequential_summary: dict[str, Any]
    n: dict[str, float]
    notes: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    pm_note: str = ""


# ----------------------------------------------------------------------------- helpers
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
    smd = d / np.sqrt(0.5 * (np.diag(Sc) + np.diag(St)))
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
    if cat.get(name).type == "ratio":
        return delta_ratio(c.ratio(name), t.ratio(name), name, alpha)
    return welch(c.mean(name), t.mean(name), name, alpha)


def post_stratified(parts: dict[tuple[Any, Any], VariantStats], ctrl: Any, trt: Any, metric: str) -> Estimate:
    """Post-stratified difference in means from per-(variant, stratum) sufficient statistics."""
    strata = sorted({s for (_, s) in parts})
    n_tot = sum(parts[(v, s)].n for (v, s) in parts)
    delta = var = 0.0
    mc_all = sum(parts[(ctrl, s)].mean(metric).sum for s in strata) / sum(parts[(ctrl, s)].n for s in strata)
    for s in strata:
        c, t = parts[(ctrl, s)].mean(metric), parts[(trt, s)].mean(metric)
        w = (c.n + t.n) / n_tot
        delta += w * (t.mean - c.mean)
        var += w**2 * (t.var / t.n + c.var / c.n)
    se = math.sqrt(var)
    p = float(2 * stats.norm.sf(abs(delta / se)))
    rel, rlo, rhi = delta / mc_all, (delta - Z975 * se) / mc_all, (delta + Z975 * se) / mc_all
    return Estimate(
        metric,
        f"post-stratified[{len(strata)} bins]",
        mc_all,
        mc_all + delta,
        delta,
        se,
        delta - Z975 * se,
        delta + Z975 * se,
        p,
        rel,
        rlo,
        rhi,
        sum(parts[(ctrl, s)].n for s in strata),
        sum(parts[(trt, s)].n for s in strata),
        delta / se,
    )


def _clt_check(cat: MetricCatalog, name: str, c: VariantStats, t: VariantStats) -> dict[str, Any]:
    """Is the normal approximation OK? n_min > 355·skew² (Kohavi et al. 2014) and >= 30 events."""
    if cat.get(name).type != "mean":
        return {"ok": True, "skew": None, "n_min": min(c.n, t.n), "required_n": None}
    skew = max(abs(c.skewness(name)), abs(t.skewness(name)))
    n_min = min(c.n, t.n)
    events = min(c.mean(name).sum, t.mean(name).sum)
    return {"ok": not needs_bootstrap(n_min, events, skew), "skew": skew, "n_min": n_min, "required_n": 355 * skew**2}


def _bootstrap_metric(cat: MetricCatalog, name: str, where: str | None, alpha: float, ctrl: Any, trt: Any) -> Estimate:
    """Fallback: pull unit-level values for one metric and run the Poisson bootstrap."""
    m = cat.get(name)
    src = cat.source
    w = f"WHERE {where}" if where else ""
    sql = f"SELECT {src.variant} AS v, CAST({m.expr} AS DOUBLE) AS y FROM read_parquet('{src.table}') {w}"
    d = duckdb_connect().execute(sql).fetchnumpy()
    v, y = np.asarray(d["v"]), np.asarray(d["y"])
    return bootstrap_diff(y[v == ctrl], y[v == trt], metric=name, alpha=alpha, config=BootstrapConfig(n_boot=2000))


def _coerce(v: int | str) -> Any:
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def _robust_row(label: str, e: Estimate, alpha: float, note: str = "") -> dict[str, Any]:
    d = primary_rule(e, e.p_value, alpha)
    return {
        "estimator": label,
        "rel_lift": e.rel_lift,
        "rel_ci": [e.rel_ci_low, e.rel_ci_high],
        "p": e.p_value,
        "verdict_if_gates_waived": d.verdict,
        "note": note,
    }


def _bayes(cfg: ExperimentConfig, e: Estimate, base: float) -> tuple[BayesDecision, bool]:
    b = bayes_decision(
        e.delta, e.se, cfg.decision.loss_threshold_relative * base, 0.0, cfg.decision.prior_sd_relative * base
    )
    informative = abs(b.post_mean) < 6 * b.post_sd  # otherwise P(effect>0) is 0 or 1 and the loss is ~0
    return b, informative


def _chart(x: np.ndarray, z: np.ndarray, obf: np.ndarray, ms_b: np.ndarray, metric: str, xlabel: str) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=130)
    naive = Z975
    ax.plot(x, np.abs(z), color=COLORS["series1"], lw=2, marker="o", ms=3, label=f"|z| of {metric}")
    ax.plot(
        x,
        np.where(np.isfinite(obf), obf, np.nan),
        color=COLORS["series2"],
        lw=2,
        label="O'Brien-Fleming (variance-based information)",
    )
    ax.plot(x, ms_b, color=COLORS["series3"], lw=2, ls="--", label="mSPRT")
    ax.axhline(naive, color=COLORS["ink2"], lw=1, ls=":", label="naive 1.96 (invalid if peeking)")
    ax.set_ylim(0, max(5.0, float(np.nanmax(np.abs(z))) * 1.2))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("|z|")
    ax.grid(color=COLORS["grid"], lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _monotone_info(var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Variance-based information fractions I_k / I_K at looks where information is a new running max."""
    info = 1.0 / var
    rm = np.maximum.accumulate(info)
    keep = (info >= rm) & np.concatenate([[True], np.diff(rm) / rm[1:] > 1e-3])
    keep[-1] = True  # the final look is always the analysis look (same rule as analyses/asos.py)
    keep[:-1] &= info[:-1] < info[-1] * (1 - 1e-3)
    return keep, info / info[-1]


# ----------------------------------------------------------------------------- warehouse source
def _sequential_warehouse(
    cfg: ExperimentConfig, cat: MetricCatalog, where: str | None
) -> tuple[str | None, dict[str, Any]]:
    seq = cfg.sequential
    if not seq.show:
        return None, {
            "shown": False,
            "reason": "N/A: single fixed-horizon analysis. The dataset has no timestamps, "
            "so any look schedule would be simulated.",
        }
    metric = cfg.primary_metric
    src = cat.source
    seg = f"(hash({src.unit} + {seq.seed}) % {seq.looks})"
    covset = cfg.variance_reduction.get(metric)
    adj = [AdjustmentRequest(metric, covset)] if covset else []
    parts = run_segmented(cat, [metric], seg, adj, where)
    ctrl, trt = _coerce(cfg.control), _coerce(cfg.treatment)
    z, var, delta = [], [], []
    for k in range(seq.looks):
        c = cumulate([parts[(ctrl, j)] for j in range(k + 1)])
        t = cumulate([parts[(trt, j)] for j in range(k + 1)])
        covs = cat.covariate_sets[covset] if covset else []
        e = (
            cuped(c.covariate(metric, covs), t.covariate(metric, covs), metric)
            if covset
            else welch(c.mean(metric), t.mean(metric), metric)
        )
        z.append(e.delta / e.se)
        var.append(e.se**2)
        delta.append(e.delta)
    z_a, v_a = np.array(z), np.array(var)
    keep, frac = _monotone_info(v_a)
    obf = np.full(z_a.size, np.inf)
    obf[keep] = gsd_boundaries(frac[keep], cfg.decision.alpha).z_bounds
    base = parts[(ctrl, 0)].mean(metric).mean
    tau2 = (seq.tau_relative * base) ** 2
    png = _chart(np.arange(1, seq.looks + 1), z_a, obf, msprt_z_boundary(v_a, tau2), metric, "look (simulated order)")
    p_ms = msprt_pvalues(np.array(delta), v_a, tau2)
    return png, {
        "shown": True,
        "label": "SIMULATED arrival order",
        "final_z": float(z_a[-1]),
        "first_look_msprt": int(np.argmax(p_ms < cfg.decision.alpha)) + 1,
    }


def _economics(
    cfg: ExperimentConfig, primary: Estimate, compliance: float, base: float, sensitivity: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Per-exposed-user effect (adjusted ITT / compliance) and a break-even grid at ASSUMED prices."""
    eco = cfg.economics
    if eco is None or compliance <= 0:
        return None
    cace, cace_se = primary.delta / compliance, primary.se / compliance
    hi_rel = max((s["rel"] for s in sensitivity), default=primary.rel_lift)
    cace_hi = hi_rel * base / compliance
    grid = []
    for v in eco.value_per_conversion:
        row: dict[str, Any] = {"value_per_conversion": v, "cells": []}
        for c in eco.cost_per_exposed_user:
            lo, mid, hi = v * (cace - Z975 * cace_se) / c, v * cace / c, v * cace_hi / c
            verdict = "profitable" if lo > 1 else ("unprofitable" if hi < 1 else "depends on magnitude")
            row["cells"].append(
                {"cost": c, "iroas": mid, "iroas_low": lo, "iroas_sensitivity_high": hi, "verdict": verdict}
            )
        grid.append(row)
    return {
        "compliance": compliance,
        "cace": cace,
        "cace_se": cace_se,
        "cace_sensitivity_high": cace_hi,
        "exposed_users_per_incremental": 1 / cace,
        "grid": grid,
        "costs": eco.cost_per_exposed_user,
        "note": "ASSUMED prices (no cost data in the dataset). iROAS = value × incremental conversions per exposed "
        "user / cost per exposed user. 'profitable' needs the CI lower bound > 1 under the adjusted estimate; "
        "'depends on magnitude' means the verdict flips within the sampling-sensitivity range.",
    }


def _sensitivity_rows(cfg: ExperimentConfig, metric: str) -> list[dict[str, Any]]:
    if not cfg.sensitivity_results:
        return []
    p = ROOT / "results" / f"{cfg.sensitivity_results}.json"
    if not p.exists():
        return []
    grid = json.loads(p.read_text())["negative_sampling_sensitivity"]
    out = []
    for g in (grid[0], grid[-1]):
        rel = g[f"{metric}_poststrat_relative"]
        se = g[f"{metric}_poststrat_relative_se"]
        out.append(
            {
                "k": g["k_upweight_nonvisitors"],
                "rel": rel,
                "ci": [rel - Z975 * se, rel + Z975 * se],
                "raw": g[f"{metric}_raw_relative"],
            }
        )
    return out


def analyze(cfg: ExperimentConfig) -> Readout:
    if cfg.source_type == "summary":
        return analyze_summary(cfg)
    cat = MetricCatalog.from_yaml(ROOT / cfg.metrics_catalog)
    cat.source.table = str(ROOT / cat.source.table)
    for j in cat.source.joins:
        j.table = str(ROOT / j.table)
    alpha = cfg.decision.alpha
    eco_metric = [cfg.economics.exposure_metric] if cfg.economics else []
    metric_names = list(
        dict.fromkeys(
            [
                cfg.primary_metric,
                *cfg.secondary_metrics,
                *[g.metric for g in cfg.guardrails],
                *cfg.diagnostics,
                *eco_metric,
            ]
        )
    )
    adjustments = [AdjustmentRequest(m, s) for m, s in cfg.variance_reduction.items()]
    adjustments += [AdjustmentRequest(m, s) for m, sets in cfg.compare_covariate_sets.items() for s in sets]
    if cfg.balance_covariate_set:
        adjustments.append(AdjustmentRequest(cfg.primary_metric, cfg.balance_covariate_set))
    where = _where(cfg, cat.source.variant, cat.source.unit)
    con = duckdb_connect()
    sv = run(cat, metric_names, adjustments, where, con)
    ctrl, trt = _coerce(cfg.control), _coerce(cfg.treatment)
    c, t = sv[ctrl], sv[trt]
    srm = srm_check([c.n, t.n], [cfg.expected_split[str(ctrl)], cfg.expected_split[str(trt)]])

    balance, balance_failed, balance_reason = None, False, ""
    if cfg.balance_covariate_set:
        balance = balance_check(c, t, cfg.primary_metric, cat.covariate_sets[cfg.balance_covariate_set])
        balance_failed = balance.p_value < cfg.decision.balance_p_threshold
        balance_reason = (
            f"Pre-treatment features differ across arms (χ² = {balance.chi2:,.0f} on {balance.df} df, "
            f"p = {fmt_p(balance.p_value)}; worst SMD {balance.smd[balance.worst]:+.3f} on {balance.worst}) "
            "although the "
            "overall split passes SRM. Until the cause is known, the effect size is not identified: regression "
            "adjustment assumes selection on observables, and outcome-dependent sampling would break that. "
            "Act on the direction only if it holds under every estimator and sensitivity scenario (table below)."
        )

    rows: list[dict[str, Any]] = []
    decision_estimates: dict[str, Estimate] = {}
    for name in [cfg.primary_metric, *cfg.secondary_metrics]:
        raw = _estimate(cat, name, c, t, alpha)
        clt = _clt_check(cat, name, c, t)
        if not clt["ok"]:
            raw = _bootstrap_metric(cat, name, where, alpha, ctrl, trt)
        row: dict[str, Any] = {
            "metric": name,
            "role": "primary" if name == cfg.primary_metric else "secondary",
            "raw": raw,
            "adjusted": None,
            "vr": None,
            "clt": clt,
            "others": {},
            "vr_compare": {},
        }
        covset = cfg.variance_reduction.get(name)
        if covset and cat.get(name).type == "mean":
            covs = cat.covariate_sets[covset]
            row["adjusted"] = cuped(
                c.covariate(name, covs), t.covariate(name, covs), name, alpha, method=f"cupac[{covset}]"
            )
            row["vr"] = variance_reduction(c.covariate(name, covs), t.covariate(name, covs))
        for extra in cfg.compare_covariate_sets.get(name, []):
            covs = cat.covariate_sets[extra]
            row["vr_compare"][extra] = variance_reduction(c.covariate(name, covs), t.covariate(name, covs))
            row["others"][f"cuped[{extra}]"] = cuped(
                c.covariate(name, covs), t.covariate(name, covs), name, alpha, method=f"cuped[{extra}]"
            )
        if name in cfg.post_stratify:
            parts = run_segmented(
                cat, [name], f"ntile(100) OVER (ORDER BY {cfg.post_stratify[name]})", where=where, con=con
            )
            row["others"]["post-stratified"] = post_stratified(parts, ctrl, trt, name)
        decision_estimates[name] = row["adjusted"] or raw
        rows.append(row)
    qs = bh_adjust([decision_estimates[r["metric"]].p_value for r in rows])
    for r, q in zip(rows, qs, strict=True):
        r["q"] = float(q)

    guardrails = [
        GuardrailResult(g.metric, _estimate(cat, g.metric, c, t, alpha), g.max_relative_drop) for g in cfg.guardrails
    ]
    diagnostics = [_estimate(cat, m, c, t, alpha) for m in cfg.diagnostics]
    primary = decision_estimates[cfg.primary_metric]
    prow = rows[0]
    base = c.mean(cfg.primary_metric).mean

    sens = _sensitivity_rows(cfg, cfg.primary_metric)
    robustness = [_robust_row("raw difference in means", prow["raw"], alpha)]
    for k, e in prow["others"].items():
        robustness.append(_robust_row(k, e, alpha))
    if prow["adjusted"]:
        robustness.append(_robust_row(prow["adjusted"].method + " (decision estimate)", prow["adjusted"], alpha))
    for s in sens:
        robustness.append(
            {
                "estimator": f"post-stratified, negatives re-weighted ×{s['k']:g} (sampling sensitivity)",
                "rel_lift": s["rel"],
                "rel_ci": s["ci"],
                "p": None,
                "verdict_if_gates_waived": "SHIP" if s["ci"][0] > 0 else "NO-SHIP",
                "note": f"raw at this k: {s['raw']:+.1%}",
            }
        )

    economics = None
    if cfg.economics:
        compliance = t.mean(cfg.economics.exposure_metric).mean - c.mean(cfg.economics.exposure_metric).mean
        economics = _economics(cfg, primary, compliance, base, sens)
    bayes, informative = _bayes(cfg, primary, base)
    naive = {
        "estimate": prow["raw"],
        "verdict": primary_rule(prow["raw"], prow["raw"].p_value, alpha).verdict,
        "text": "raw difference in means, no balance or SRM gate",
    }
    decision = decide(srm.blocked, srm.p_value, guardrails, primary, prow["q"], alpha, balance_failed, balance_reason)
    png, seq_summary = _sequential_warehouse(cfg, cat, where)
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
        diagnostics,
        robustness,
        economics,
        bayes,
        informative,
        decision,
        naive,
        png,
        seq_summary,
        {"control": c.n, "treatment": t.n},
        notes,
    )


# ----------------------------------------------------------------------------- summary source (ASOS)
def analyze_summary(cfg: ExperimentConfig) -> Readout:
    from expkit.analyses.asos import load_daily_trajectories

    assert cfg.summary is not None
    alpha = cfg.decision.alpha
    trajs = {
        t.metric: t
        for t in load_daily_trajectories()
        if t.experiment == cfg.summary.experiment_id and t.variant == cfg.summary.variant_id
    }
    by_name = {name: trajs[mid] for name, mid in cfg.summary.metrics.items()}

    def final_estimate(name: str) -> Estimate:
        tr = by_name[name]
        c = MeanStats.from_summary(tr.n_c[-1], tr.mean_c[-1], tr.var_c[-1])
        t = MeanStats.from_summary(tr.n_t[-1], tr.mean_t[-1], tr.var_t[-1])
        return welch(c, t, name, alpha)

    p_tr = by_name[cfg.primary_metric]
    srm = srm_check([p_tr.n_c[-1], p_tr.n_t[-1]], [cfg.expected_split["0"], cfg.expected_split["1"]])
    rows: list[dict[str, Any]] = []
    for name in [cfg.primary_metric, *cfg.secondary_metrics]:
        rows.append(
            {
                "metric": name,
                "role": "primary" if name == cfg.primary_metric else "secondary",
                "raw": final_estimate(name),
                "adjusted": None,
                "vr": None,
                "others": {},
                "vr_compare": {},
                "clt": {"ok": True, "skew": None, "n_min": min(p_tr.n_c[-1], p_tr.n_t[-1]), "required_n": None},
            }
        )
    qs = bh_adjust([r["raw"].p_value for r in rows])
    for r, q in zip(rows, qs, strict=True):
        r["q"] = float(q)
    guardrails = [GuardrailResult(g.metric, final_estimate(g.metric), g.max_relative_drop) for g in cfg.guardrails]
    primary = rows[0]["raw"]
    base = float(p_tr.mean_c[-1])
    bayes, informative = _bayes(cfg, primary, base)
    decision = decide(srm.blocked, srm.p_value, guardrails, primary, rows[0]["q"], alpha)

    # real daily looks: naive peeking vs OBF (variance-based information) vs mSPRT
    z, var, delta = p_tr.z, p_tr.var_delta, p_tr.delta
    keep, frac = _monotone_info(var)
    obf = np.full(z.size, np.inf)
    obf[keep] = gsd_boundaries(frac[keep], alpha).z_bounds
    tau2 = (cfg.sequential.tau_relative * base) ** 2
    p_ms = msprt_pvalues(delta, var, tau2)

    def first(hit: np.ndarray) -> float | None:
        idx = np.flatnonzero(hit)
        return float(p_tr.day[idx[0]]) if idx.size else None

    naive_day = first(np.abs(z) >= Z975)
    naive_sign = float(np.sign(z[np.flatnonzero(np.abs(z) >= Z975)[0]])) if naive_day else 0.0
    png = _chart(p_tr.day, z, obf, msprt_z_boundary(var, tau2), cfg.primary_metric, "day since start (real snapshots)")
    seq_summary = {
        "shown": True,
        "label": "real daily snapshots (ASOS)",
        "first_day_naive": naive_day,
        "first_day_obf": first(np.abs(z) >= obf),
        "first_day_msprt": first(p_ms < alpha),
        "days": float(p_tr.day[-1]),
        "final_z": float(z[-1]),
        "tau_relative": cfg.sequential.tau_relative,
    }
    rel0 = relative_ci(float(p_tr.mean_c[-1]), float(p_tr.mean_t[-1]), 0.0, 0.0, Z975)[0]
    naive = {
        "estimate": primary,
        "verdict": ("SHIP" if naive_sign > 0 else "NO-SHIP") if naive_day else "no call",
        "text": f"a daily-peeking dashboard flagged {'a win' if naive_sign > 0 else 'a loss'} on day {naive_day:g}"
        if naive_day
        else "a daily-peeking dashboard never crossed 1.96",
        "final_rel": rel0,
    }
    robustness = [_robust_row("Welch on final snapshot (decision estimate)", primary, alpha)]
    facts = _trajectory_facts(cfg, by_name)
    return Readout(
        cfg,
        srm,
        None,
        rows,
        guardrails,
        [],
        robustness,
        None,
        bayes,
        informative,
        decision,
        naive,
        png,
        seq_summary,
        {"control": float(p_tr.n_c[-1]), "treatment": float(p_tr.n_t[-1])},
        [],
        facts,
        cfg.pm_note.format(**facts) if cfg.pm_note else "",
    )


def _trajectory_facts(cfg: ExperimentConfig, by_name: dict[str, Any], horizon_days: float = 14.0) -> dict[str, Any]:
    """Numbers the PM note quotes: effect decay over time, and whether extending would clear each guardrail margin.

    Projection: the guardrail SE shrank by a factor r over the last ``horizon_days``; assume the same factor for the
    next period (cumulative per-user metrics shrink much slower than 1/sqrt(n)). The margin is cleared when
    estimate − 1.96·SE > −margin.
    """

    def at_day(tr: Any, day: float) -> int:
        idx = np.flatnonzero(tr.day <= day)
        return int(idx[-1]) if idx.size else 0

    p = by_name[cfg.primary_metric]
    f: dict[str, Any] = {"days": float(p.day[-1])}
    for d in (7, 14):
        f[f"primary_day{d}"] = float(p.rel[at_day(p, d)]) * 100
    f["primary_final"] = float(p.rel[-1]) * 100
    for j, g in enumerate(cfg.guardrails, start=1):
        tr = by_name[g.metric]
        se = np.sqrt(tr.var_rel)
        k0 = at_day(tr, tr.day[-1] - horizon_days)
        r = float(se[-1] / se[k0])
        est = float(tr.rel[-1])
        se_next = float(se[-1]) * r
        se_req = (est + g.max_relative_drop) / Z975
        var_factor = (float(se[-1]) / se_req) ** 2 if se_req > 0 else float("inf")
        periods = math.log(var_factor) / math.log(1 / r**2) if r < 1 and var_factor > 1 else float("inf")
        f.update(
            {
                f"g{j}_day7": float(tr.rel[at_day(tr, 7)]) * 100,
                f"g{j}_final": est * 100,
                f"g{j}_se": float(se[-1]) * 100,
                f"g{j}_se_shrink_pct": (1 - r) * 100,
                f"g{j}_lower_now": (est - Z975 * float(se[-1])) * 100,
                f"g{j}_lower_after": (est - Z975 * se_next) * 100,
                f"g{j}_se_needed": se_req * 100,
                f"g{j}_var_factor": var_factor,
                f"g{j}_days_needed": periods * horizon_days,
            }
        )
    f["n_growth_pct"] = (
        float(
            (p.n_c[-1] + p.n_t[-1])
            / (p.n_c[at_day(p, p.day[-1] - horizon_days)] + p.n_t[at_day(p, p.day[-1] - horizon_days)])
            - 1
        )
        * 100
    )
    f["margin_pct"] = cfg.guardrails[0].max_relative_drop * 100 if cfg.guardrails else 0.0
    return f


# ----------------------------------------------------------------------------- output
def render(readout: Readout, path: str | Path) -> Path:
    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=True)

    def pct(v: float | None, d: int = 2) -> str:
        return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.{d}%}"

    env.filters["pct"] = pct
    env.filters["num"] = lambda v, d=4: "—" if v is None else f"{v:.{d}g}"
    env.filters["p"] = lambda v: "—" if v is None else fmt_p(v)
    html = env.get_template("readout.html.j2").render(r=readout, cfg=readout.config)
    out = Path(path)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


def _e(e: Estimate) -> dict[str, Any]:
    return {
        "delta": e.delta,
        "se": e.se,
        "rel_lift": e.rel_lift,
        "rel_ci": [e.rel_ci_low, e.rel_ci_high],
        "p": e.p_value,
        "method": e.method,
    }


def summary_json(r: Readout) -> dict[str, Any]:
    """Machine-readable summary of a readout (used by acceptance checks and the docs)."""
    vr: dict[str, float] = {}
    metrics: dict[str, Any] = {}
    for row in r.rows:
        if row["vr"] is not None:
            vr[f"{row['metric']}_cupac"] = row["vr"]
        for k, v in row["vr_compare"].items():
            vr[f"{row['metric']}_{k}"] = v
        m = {"raw": _e(row["raw"]), "bh_q": row["q"]}
        if row["adjusted"]:
            m["adjusted"] = _e(row["adjusted"])
        for k, e in row["others"].items():
            m[k] = _e(e)
        metrics[row["metric"]] = m
    return {
        "name": r.config.name,
        "primary_metric": r.config.primary_metric,
        "verdict": r.decision.verdict,
        "rule": r.decision.rule,
        "reason": r.decision.reason,
        "srm_p": r.srm.p_value,
        "balance": None
        if r.balance is None
        else {"chi2": r.balance.chi2, "p": r.balance.p_value, "max_abs_smd": r.balance.max_abs_smd},
        "variance_reduction": vr,
        "metrics": metrics,
        "guardrails": [{"metric": g.metric, "status": g.status, **_e(g.estimate)} for g in r.guardrails],
        "diagnostics": [_e(d) | {"metric": d.metric} for d in r.diagnostics],
        "robustness": r.robustness,
        "economics": r.economics,
        "naive_dashboard": {
            "verdict": r.naive["verdict"],
            "text": r.naive["text"],
            "rel_lift": r.naive["estimate"].rel_lift,
        },
        "bayes_panel": True,
        "bayes_informative": r.bayes_informative,
        "bayes_verdict": r.bayes.verdict,
        "bayes_loss_ship": r.bayes.loss_ship,
        "bayes_prob_positive": r.bayes.prob_positive,
        "bayes_post_mean_relative": r.bayes.post_mean / r.rows[0]["raw"].control,
        "sequential": r.sequential_summary,
        "trajectory_facts": r.facts,
        "pm_note": r.pm_note,
    }
