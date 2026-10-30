"""Render README.md and docs/*.md from docs/templates/*.j2 and results/*.json.

Every number in the rendered docs comes from a results file (or is derived here from one), so the docs cannot
drift from the analysis. ``tests/test_docs_drift.py`` re-renders in memory and fails if a file on disk differs.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from scipy import stats

from expkit.data import ROOT
from expkit.stats.power import total_sample_size
from expkit.stats.sequential import gsd_boundaries

TEMPLATES = ROOT / "docs" / "templates"
TARGETS = {
    "README.md.j2": ROOT / "README.md",
    "decision_memo.md.j2": ROOT / "docs" / "decision_memo.md",
    "appendix.md.j2": ROOT / "docs" / "appendix.md",
    "experiment_brief.md.j2": ROOT / "docs" / "experiment_brief.md",
}
RESULT_FILES = [
    "datasets",
    "aa_tests",
    "aa_tests_85_15",
    "aa_bootstrap",
    "power_curves",
    "criteo_peeking",
    "crosscheck",
    "cupac_models",
    "balance",
    "noncompliance",
    "hte",
    "heavy_tails",
    "asos_portfolio",
    "scorecard_criteo",
    "scorecard_asos_d53f0e",
    "scorecard_criteo_srm_bug",
]


def _load() -> dict[str, Any]:
    return {k: json.loads((ROOT / "results" / f"{k}.json").read_text()) for k in RESULT_FILES}


def mde80(points: list[dict[str, Any]], key: str) -> float:
    """Relative lift at which empirical power crosses 80% (linear interpolation on the simulated grid)."""
    for a, b in itertools.pairwise(points):
        pa, pb = a[f"power_{key}"], b[f"power_{key}"]
        if pa < 0.8 <= pb:
            return float(a["relative_lift"] + (0.8 - pa) / (pb - pa) * (b["relative_lift"] - a["relative_lift"]))
    return float("nan")


def derived(R: dict[str, Any]) -> dict[str, Any]:
    """Quantities computed from results (kept here so templates stay declarative)."""
    pc = R["power_curves"]["curves"]
    d: dict[str, Any] = {
        "mde": {m: {k: mde80(pc[m]["points"], k) for k in ("raw", "cuped", "cupac")} for m in pc},
        "max_power_gap": max(abs(p["power_raw"] - p["analytic_raw"]) for c in pc.values() for p in c["points"]),
    }
    aa = R["aa_tests"]["methods"]
    aa85 = R["aa_tests_85_15"]["methods"]
    keys = [k for k in aa if k in aa85]
    d["aa"] = {
        "min": min(aa[k]["fpr"] for k in keys),
        "max": max(aa[k]["fpr"] for k in keys),
        "min_ks": min(aa[k]["ks_pvalue"] for k in keys),
        "min85": min(aa85[k]["fpr"] for k in keys),
        "max85": max(aa85[k]["fpr"] for k in keys),
        "min_ks85": min(aa85[k]["ks_pvalue"] for k in keys),
    }
    # pre-launch brief: conversion as primary at the real 85/15 split, with the CUPAC variance reduction
    base = R["balance"]["control_means"]["conversion"]
    vr = R["scorecard_criteo"]["variance_reduction"]["conversion_cupac"]
    var = base * (1 - base)
    brief = []
    for rel in (0.05, 0.10, 0.20):
        n_raw = total_sample_size(rel * base, var, share_t=0.85)
        brief.append({"mde_rel": rel, "n_raw": n_raw, "n_cupac": n_raw * (1 - vr)})
    # schedule: enroll 14 days at an assumed 500k eligible users/day; 14-day outcome window; interim read on day 21
    # (first enrollment week's windows complete, information ~0.5) and final read on day 28
    traffic, enroll_days = 500_000, 14
    n_enrolled = traffic * enroll_days
    kz = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) ** 2
    mde_abs = math.sqrt(kz * var * (1 - vr) * (1 / 0.85 + 1 / 0.15) / n_enrolled)
    looks = np.array([0.5, 1.0])
    d["brief"] = {
        "baseline": base,
        "vr": vr,
        "rows": brief,
        "looks": looks.tolist(),
        "obf": gsd_boundaries(looks).z_bounds.tolist(),
        "traffic": traffic,
        "enroll_days": enroll_days,
        "n_enrolled": n_enrolled,
        "mde_rel_at_enrollment": mde_abs / base,
    }
    # EB ranges across the four ASOS metrics
    eb = R["asos_portfolio"]["eb"]
    ms = list(eb)
    d["eb"] = {
        "small_ratio_min": min(eb[m]["winners"]["small_winners_median_ratio_npmle"] for m in ms),
        "small_ratio_max": max(eb[m]["winners"]["small_winners_median_ratio_npmle"] for m in ms),
        "means_ratio_min": min(eb[m]["winners"]["exaggeration_ratio_of_means_npmle"] for m in ms),
        "means_ratio_max": max(eb[m]["winners"]["exaggeration_ratio_of_means_npmle"] for m in ms),
        "gt_ci_min": min(eb[m]["npmle_prior"]["p_true_lift_gt_0.1pct_ci90"][0] for m in ms),
        "gt_ci_max": max(eb[m]["npmle_prior"]["p_true_lift_gt_0.1pct_ci90"][1] for m in ms),
        "cov_win_min": min(eb[m]["simulation_check"]["truth_npmle"]["coverage_95_among_winners"] for m in ms),
        "cov_win_max": max(eb[m]["simulation_check"]["truth_npmle"]["coverage_95_among_winners"] for m in ms),
        "haircut06": {
            m: next(h for h in eb[m]["haircut_at_median_se"] if abs(h["observed"] - 0.006) < 1e-9) for m in ms
        },
    }
    t = R["hte"]["targeting"]
    d["top20"] = {
        "conv_x": next(r for r in t["outcome_conversion"]["rank_by_conversion_x_learner"] if r["top"] == 0.2),
        "conv_resp": next(r for r in t["outcome_conversion"]["rank_by_conversion_response_model"] if r["top"] == 0.2),
        "conv_by_visit_x": next(r for r in t["outcome_conversion"]["rank_by_visit_x_learner"] if r["top"] == 0.2),
        "visit_x": next(r for r in t["outcome_visit"]["rank_by_visit_x_learner"] if r["top"] == 0.2),
        "conv_x_unweighted": next(
            r for r in t["outcome_conversion_unweighted"]["rank_by_conversion_x_learner"] if r["top"] == 0.2
        ),
    }
    return d


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    def pct(v: float, d: int = 1, sign: bool = False) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        return f"{v:+.{d}%}" if sign else f"{v:.{d}%}"

    env.filters["pct"] = pct
    env.filters["spct"] = lambda v, d=1: pct(v, d, True)
    env.filters["f"] = lambda v, d=2: f"{v:.{d}f}"
    env.filters["g"] = lambda v, d=3: f"{v:.{d}g}"
    env.filters["e"] = lambda v, d=0: f"{v:.{d}e}"
    env.filters["int"] = lambda v: f"{round(v):,}"
    env.filters["pp"] = lambda v, d=1: f"{100 * v:.{d}f} pp"
    env.filters["x"] = lambda v, d=2: f"×{v:.{d}f}"
    return env


def render_all(write: bool = True) -> dict[Path, str]:
    R = _load()
    ctx = {"R": R, "D": derived(R)}
    env = _env()
    out = {}
    for name, target in TARGETS.items():
        text = env.get_template(name).render(**ctx)
        out[target] = text
        if write:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
    return out
