"""Re-run the estimator cross-checks on *real* Criteo rows (C1-C5), through the SQL metric layer.

Our numbers come from the compiled DuckDB sufficient-statistics query; the references come from
scipy / statsmodels applied to the unit-level arrays.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from expkit.analyses.common import load_criteo, write_json
from expkit.data import CRITEO_FEATURES, ROOT
from expkit.metrics.compiler import AdjustmentRequest, run
from expkit.metrics.spec import MetricCatalog
from expkit.stats.cuped import cuped, pooled_theta
from expkit.stats.delta import delta_ratio
from expkit.stats.multiple import bh_adjust
from expkit.stats.srm import srm_check
from expkit.stats.ttest import welch

SUBSAMPLE = "hash(unit_id) % 7 = 0"  # ~2M real users, for the statsmodels references


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def main() -> dict[str, Any]:
    t0 = time.time()
    cat = MetricCatalog.from_yaml(ROOT / "metrics/criteo.yaml")
    cat.source.table = str(ROOT / cat.source.table)
    for j in cat.source.joins:
        j.table = str(ROOT / j.table)
    out: dict[str, Any] = {}

    # C1 Welch on the full 13.98M rows
    vs = run(cat, ["visit", "conversion"])
    d = load_criteo(["treatment", "visit", "conversion"])
    c1 = {}
    for m in ("visit", "conversion"):
        ours = welch(vs[0].mean(m), vs[1].mean(m), m)
        y, t = d[m].astype(np.float64), d["treatment"]
        ref = stats.ttest_ind(y[t == 1], y[t == 0], equal_var=False)
        ci = ref.confidence_interval()
        se_ref = (ci.high - ci.low) / (2 * stats.t.ppf(0.975, ref.df))
        c1[m] = {
            "rel_diff_stat": _rel(ours.stat, ref.statistic),
            "rel_diff_se": _rel(ours.se, se_ref),
            "abs_diff_p": abs(ours.p_value - ref.pvalue),
            "rel_diff_estimate": _rel(ours.delta, ci.low + (ci.high - ci.low) / 2),
        }
    out["C1_welch_vs_scipy_full_data"] = c1

    # C2 delta method vs statsmodels cluster-robust OLS on the visit-level data (users with visit=1)
    vs2 = run(cat, ["conversion_per_visit"])
    ours = delta_ratio(vs2[0].ratio("conversion_per_visit"), vs2[1].ratio("conversion_per_visit"))
    var_ref = 0.0
    for arm in (0, 1):
        m = (d["treatment"] == arm) & (d["visit"] == 1)
        yv = d["conversion"][m].astype(np.float64)
        g = np.arange(yv.size)
        fit = sm.OLS(yv, np.ones(yv.size)).fit(cov_type="cluster", cov_kwds={"groups": g})
        n_units, n_clusters = float((d["treatment"] == arm).sum()), float(yv.size)
        # delta uses all randomized users (n/(n-1)); statsmodels uses visiting users (G/(G-1))
        factor = (n_units / (n_units - 1)) / (n_clusters / (n_clusters - 1))
        var_ref += float(fit.bse[0] ** 2) * factor
    out["C2_delta_vs_statsmodels_cluster"] = {
        "rel_diff_se": _rel(ours.se, float(np.sqrt(var_ref))),
        "note": "after the documented n/(n-1) vs G/(G-1) small-sample factor",
    }

    # C3 CUPED from SQL vs statsmodels theta + scipy Welch on adjusted values (2M-user real subsample)
    vs3 = run(cat, ["visit"], [AdjustmentRequest("visit", "cuped")], where=SUBSAMPLE)
    covs = cat.covariate_sets["cuped"]
    e = cuped(vs3[0].covariate("visit", covs), vs3[1].covariate("visit", covs), "visit")
    th_ours = pooled_theta(vs3[0].covariate("visit", covs), vs3[1].covariate("visit", covs))
    sub = load_criteo(["treatment", "visit", *CRITEO_FEATURES], where=SUBSAMPLE)
    X = np.column_stack([sub[f] for f in CRITEO_FEATURES]).astype(np.float64)
    y, t = sub["visit"].astype(np.float64), sub["treatment"]
    th_sm = sm.OLS(y, sm.add_constant(X)).fit().params[1:]
    y_adj = y - (X - X.mean(axis=0)) @ th_sm
    ref = stats.ttest_ind(y_adj[t == 1], y_adj[t == 0], equal_var=False)
    anc = sm.OLS(y, sm.add_constant(np.column_stack([t.astype(np.float64), X]))).fit(cov_type="HC1")
    out["C3_cuped"] = {
        "n": int(y.size),
        "max_rel_diff_theta": float(np.max(np.abs(th_ours - th_sm) / np.abs(th_sm))),
        "rel_diff_stat": _rel(e.stat, ref.statistic),
        "rel_diff_estimate": _rel(e.delta, y_adj[t == 1].mean() - y_adj[t == 0].mean()),
        "ancova_estimate": float(anc.params[1]),
        "cuped_estimate": e.delta,
        "rel_diff_vs_ancova": _rel(e.delta, float(anc.params[1])),
        "rel_diff_se_vs_ancova": _rel(e.se, float(anc.bse[1])),
    }

    # C4 SRM, C5 BH on the real scorecard p-values
    srm = srm_check([vs[0].n, vs[1].n], [0.15, 0.85])
    ref_srm = stats.chisquare([vs[0].n, vs[1].n], f_exp=[0.15 * (vs[0].n + vs[1].n), 0.85 * (vs[0].n + vs[1].n)])
    out["C4_srm_vs_scipy"] = {
        "abs_diff_chi2": abs(srm.chi2 - ref_srm.statistic),
        "abs_diff_p": abs(srm.p_value - ref_srm.pvalue),
    }
    pv = np.array([c["abs_diff_p"] for c in c1.values()] + [0.01, 0.04, 0.2, 0.03])
    out["C5_bh_vs_statsmodels"] = {
        "max_abs_diff": float(np.max(np.abs(bh_adjust(pv) - multipletests(pv, method="fdr_bh")[1])))
    }
    out["seconds"] = round(time.time() - t0, 1)
    write_json("crosscheck", out)
    return out
