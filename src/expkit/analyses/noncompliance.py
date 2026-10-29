"""Non-compliance on Criteo: assignment (treatment) vs. actually being shown an ad (exposure).

Only 3.6% of assigned users were exposed; control users can never be exposed (one-sided
non-compliance). This module estimates

* ITT (effect of being *eligible* for ads) and CACE / LATE (effect of actually *seeing* an ad on
  users who would see one if eligible) with Wald and 2SLS (with/without covariates);
* the complier baseline E[Y(0) | complier] (Abadie 2003);
* a falsification test of the exclusion restriction by strata of predicted exposure propensity;
* how much dilution costs in sample size, and what a "triggered" analysis (Deng & Hu 2015; Kohavi
  et al. 2020 ch. 20) recovers, with an ideal trigger (ghost ads) and with a predicted trigger.
"""

from __future__ import annotations

import math
import time
from typing import Any

import duckdb
import numpy as np
from scipy import stats

from expkit.analyses.common import CUPAC_PARQUET, load_criteo, write_json
from expkit.data import CRITEO_FEATURES, CRITEO_PARQUET
from expkit.db import duckdb_connect
from expkit.stats.cuped import cuped
from expkit.stats.iv import complier_control_mean, required_n_ratio, tsls_with_controls, wald
from expkit.stats.power import total_sample_size
from expkit.stats.types import CovariateStats, RatioStats

Z975 = float(stats.norm.ppf(0.975))


def _arm_stats(con: duckdb.DuckDBPyConnection, y: str, where: str = "TRUE") -> dict[int, RatioStats]:
    rows = con.execute(
        f"""SELECT treatment, count(*), sum({y}), sum(exposure), sum({y}*{y}), sum(exposure*exposure), sum({y}*exposure)
            FROM read_parquet('{CRITEO_PARQUET}') s JOIN read_parquet('{CUPAC_PARQUET}') c USING (unit_id)
            WHERE {where} GROUP BY 1 ORDER BY 1"""
    ).fetchall()
    return {int(r[0]): RatioStats(*(float(v) for v in r[1:])) for r in rows}


def _linearmodels_check(outcome: str) -> dict[str, float]:
    """Full-data cross-check of our Wald SE vs linearmodels IV2SLS (robust, HC0)."""
    from linearmodels.iv import IV2SLS

    d = load_criteo(["treatment", "exposure", outcome])
    import pandas as pd

    df = pd.DataFrame(
        {
            "y": d[outcome].astype(np.float64),
            "d": d["exposure"].astype(np.float64),
            "z": d["treatment"].astype(np.float64),
        }
    )
    df["const"] = 1.0
    fit = IV2SLS(df["y"], df[["const"]], df[["d"]], df[["z"]]).fit(cov_type="robust", debiased=False)
    return {"coef": float(fit.params["d"]), "se": float(fit.std_errors["d"])}


def run() -> dict[str, Any]:
    t0 = time.time()
    con = duckdb_connect()
    out: dict[str, Any] = {}
    exposed_in_control = con.execute(
        f"SELECT sum(exposure) FROM read_parquet('{CRITEO_PARQUET}') WHERE treatment = 0"
    ).fetchone()
    assert exposed_in_control is not None
    out["one_sided"] = {"exposed_users_in_control": int(exposed_in_control[0])}

    feats = load_criteo(
        [
            *CRITEO_FEATURES,
            "treatment",
            "exposure",
            "visit",
            "conversion",
            "cupac_visit",
            "cupac_conversion",
        ],
        with_cupac=True,
    )
    W = np.column_stack([feats[f] for f in CRITEO_FEATURES])
    for y in ("visit", "conversion"):
        s = _arm_stats(con, y)
        iv = wald(s[0], s[1])
        lm = _linearmodels_check(y)
        r: dict[str, Any] = {
            "itt": iv.itt_y,
            "itt_se": iv.itt_y_se,
            "itt_relative": iv.itt_y / iv.control_mean,
            "compliance_rate": iv.itt_d,
            "cace": iv.cace,
            "cace_se": iv.cace_se,
            "cace_ci95": [iv.ci_low, iv.ci_high],
            "cace_p": iv.p_value,
            "linearmodels_iv2sls": lm,
            "rel_diff_coef_vs_linearmodels": abs(iv.cace - lm["coef"]) / abs(lm["coef"]),
            "rel_diff_se_vs_linearmodels": abs(iv.cace_se - lm["se"]) / lm["se"],
        }
        yv = feats[y].astype(np.float64)
        d = feats["exposure"].astype(np.float64)
        z = feats["treatment"].astype(np.float64)
        b, se = tsls_with_controls(yv, d, z, W)
        r["tsls_features"] = {"cace": b, "se": se, "se_ratio_vs_wald": se / iv.cace_se}
        b2, se2 = tsls_with_controls(yv, d, z, feats[f"cupac_{y}"][:, None])
        r["tsls_cupac"] = {"cace": b2, "se": se2, "se_ratio_vs_wald": se2 / iv.cace_se}
        # Wald on the CUPAC-adjusted ITT (the ITT the scorecard uses), for consistency with the headline
        tr, ct = z == 1, z == 0
        xc = feats[f"cupac_{y}"].astype(np.float64)
        adj = cuped(CovariateStats.from_arrays(yv[ct], xc[ct]), CovariateStats.from_arrays(yv[tr], xc[tr]), y)
        r["wald_on_adjusted_itt"] = {"itt": adj.delta, "cace": adj.delta / iv.itt_d, "se": adj.se / iv.itt_d}
        r["headline_cace"] = {
            "value": b2,
            "se": se2,
            "estimator": "2SLS with the CUPAC covariate",
            "unadjusted_wald_upper_reference": iv.cace,
        }

        # complier means (one-sided): treated & exposed are compliers; treated & unexposed never-takers
        st = con.execute(
            f"""SELECT avg({y}) FILTER (treatment=1 AND exposure=1), avg({y}) FILTER (treatment=1 AND exposure=0),
                       avg({y}) FILTER (treatment=0) FROM read_parquet('{CRITEO_PARQUET}')"""
        ).fetchone()
        assert st is not None
        y1_c, y_nt, y_ctrl = (float(v) for v in st)
        pi = iv.itt_d
        y0_c = complier_control_mean(y_ctrl, y_nt, pi)
        r["complier_means"] = {
            "E[Y(1)|complier]": y1_c,
            "E[Y(0)|complier]": y0_c,
            "E[Y|never-taker]": y_nt,
            "cace_check": y1_c - y0_c,
        }
        # dilution: ideal trigger (ghost ads) vs ITT
        var_all = float(np.var(yv))
        var_c = 0.5 * (y1_c * (1 - y1_c) + y0_c * (1 - y0_c))
        n_itt = total_sample_size(iv.itt_y, var_all, share_t=0.85)
        v_t, v_c = float(np.var(yv[tr])), float(np.var(yv[ct]))
        weighted = (v_t / 0.85 + v_c / 0.15) / (pi * (y1_c * (1 - y1_c) / 0.85 + y0_c * (1 - y0_c) / 0.15))
        r["dilution"] = {
            "n_total_needed_itt_at_observed_effect": n_itt,
            "n_ratio_itt_over_ideal_trigger": required_n_ratio(var_all, var_c, pi),
            "n_ratio_itt_over_ideal_trigger_weighted_85_15": weighted,
            "n_total_needed_ideal_trigger": n_itt / required_n_ratio(var_all, var_c, pi),
            "formula_weighted": "(s2_T/0.85 + s2_C/0.15) / (pi * (s2_C1/0.85 + s2_C0/0.15)), complier variances "
            "from complier means (raw contrasts)",
            "note": "ideal trigger = counterfactual exposure logged in control too (ghost ads); Deng & Hu 2015",
        }
        r["kitagawa"] = _kitagawa(con, y)
        r["stratified_by_predicted_exposure"] = _stratified(r_strata := _strata(con, y), iv.itt_y_se)
        r["strata"] = r_strata
        r["retargeting_caveat"] = (
            "Exposure is logged over the same 2-week window as the outcome, and retargeting means site visits can "
            "cause ad eligibility. Compliers are partly selected on outcome behaviour (complier baseline E[Y(0)|C] "
            "vs control mean shows it), so the CACE is a ratio of ITTs for this population, not the effect of a "
            "pre-defined ad opportunity. Ghost-ads designs log the counterfactual first opportunity to avoid this."
        )
        r["predicted_trigger"] = _predicted_trigger(con, y, iv.itt_y, iv.itt_y_se)
        out[y] = r
    out["seconds"] = round(time.time() - t0, 1)
    write_json("noncompliance", out)
    return out


def _strata(con: duckdb.DuckDBPyConnection, y: str) -> list[dict[str, float]]:
    """Exclusion-restriction falsification by deciles of cross-fitted P(exposure | X).

    Under one-sided non-compliance + exclusion, within stratum s:
        ITT_s = π_s · CACE_s, and since Y is binary |CACE_s| <= 1, so |ITT_s| <= π_s.
        ȳ(Z=1, D=0) − ȳ(Z=0) = −π_s (E[Y0|C] − E[Y0|NT]) ∈ [−π_s, π_s].
    A difference significantly outside ±π_s would falsify exclusion (or reveal broken randomization).
    """
    rows = con.execute(
        f"""WITH u AS (SELECT s.treatment, s.exposure, s.{y} AS y, ntile(10) OVER (ORDER BY c.p_exposure) AS dec
                       FROM read_parquet('{CRITEO_PARQUET}') s JOIN read_parquet('{CUPAC_PARQUET}') c USING (unit_id))
            SELECT dec,
                   count(*) FILTER (treatment=1), avg(exposure) FILTER (treatment=1),
                   avg(y) FILTER (treatment=1), var_samp(y) FILTER (treatment=1),
                   count(*) FILTER (treatment=0), avg(y) FILTER (treatment=0), var_samp(y) FILTER (treatment=0),
                   count(*) FILTER (treatment=1 AND exposure=0), avg(y) FILTER (treatment=1 AND exposure=0),
                   var_samp(y) FILTER (treatment=1 AND exposure=0), sum(exposure)
            FROM u GROUP BY dec ORDER BY dec"""
    ).fetchall()
    tot_exp = sum(r[11] for r in rows)
    out = []
    for r in rows:
        dec, n_t, pi, m_t, v_t, n_c, m_c, v_c, n_nt, m_nt, v_nt, n_exp = r
        itt = m_t - m_c
        itt_se = math.sqrt(v_t / n_t + v_c / n_c)
        nt_diff = m_nt - m_c
        nt_se = math.sqrt(v_nt / n_nt + v_c / n_c)
        out.append(
            {
                "decile": int(dec),
                "compliance": float(pi),
                "share_of_all_exposures": n_exp / tot_exp,
                "itt": itt,
                "itt_se": itt_se,
                "itt_exceeds_bound": bool(abs(itt) - Z975 * itt_se > pi),
                "nevertaker_minus_control": nt_diff,
                "nevertaker_minus_control_se": nt_se,
                "nevertaker_diff_outside_bound": bool(abs(nt_diff) - Z975 * nt_se > pi),
                "cace_stratum": itt / pi if pi > 0 else float("nan"),
            }
        )
    return out


def _kitagawa(con: duckdb.DuckDBPyConnection, y: str) -> dict[str, Any]:
    """Sharp testable implications of IV validity under one-sided non-compliance (Balke & Pearl 1997;
    Kitagawa 2015): within each stratum, P(Y=1, D=0 | Z=1) <= P(Y=1 | Z=0) and P(Y=0, D=0 | Z=1) <= P(Y=0 | Z=0).
    Reported as differences (must be <= 0) with z-statistics; a violation is z > 1.96."""
    rows = con.execute(
        f"""WITH u AS (SELECT s.treatment z, s.exposure d, s.{y} AS y, ntile(10) OVER (ORDER BY c.p_exposure) AS dec
                       FROM read_parquet('{CRITEO_PARQUET}') s
                       JOIN read_parquet('{CUPAC_PARQUET}') c USING (unit_id))
            SELECT dec, avg((y=1 AND d=0)::INT) FILTER (z=1), count(*) FILTER (z=1),
                   avg((y=0 AND d=0)::INT) FILTER (z=1), avg(y) FILTER (z=0), count(*) FILTER (z=0)
            FROM u GROUP BY dec ORDER BY dec"""
    ).fetchall()
    out = []
    for dec, a1, n1, a0, p1c, n0 in rows:
        d1 = a1 - p1c
        se1 = math.sqrt(a1 * (1 - a1) / n1 + p1c * (1 - p1c) / n0)
        d0 = a0 - (1 - p1c)
        se0 = math.sqrt(a0 * (1 - a0) / n1 + p1c * (1 - p1c) / n0)
        out.append({"decile": int(dec), "y1_diff": d1, "y1_z": d1 / se1, "y0_diff": d0, "y0_z": d0 / se0})
    zs = [r["y1_z"] for r in out] + [r["y0_z"] for r in out]
    return {
        "strata": out,
        "n_inequalities": len(zs),
        "n_violations_z_gt_1.96": sum(zv > 1.96 for zv in zs),
        "max_z": max(zs),
    }


def _stratified(strata: list[dict[str, Any]], unstratified_se: float) -> dict[str, float]:
    """The unbiased use of a predicted trigger: ITT = sum_s share_s * ITT_s over predicted-exposure deciles."""
    w = 1 / len(strata)  # deciles: equal shares
    itt = sum(w * s["itt"] for s in strata)
    se = math.sqrt(sum((w * s["itt_se"]) ** 2 for s in strata))
    return {"itt": itt, "se": se, "se_ratio_vs_unstratified": se / unstratified_se}


def _predicted_trigger(
    con: duckdb.DuckDBPyConnection, y: str, itt_full: float, itt_full_se: float
) -> list[dict[str, float]]:
    """Analyze only users whose *pre-treatment* predicted exposure is in the top q (both arms),
    then dilute back to the whole population (Deng & Hu 2015). Valid because the trigger is a
    function of pre-treatment features only, applied identically to both arms."""
    out = []
    n_all = con.execute(f"SELECT count(*) FROM read_parquet('{CRITEO_PARQUET}')").fetchone()
    assert n_all is not None
    for top in (0.1, 0.2, 0.3, 0.5):
        r = con.execute(
            f"""WITH u AS (SELECT s.treatment, s.exposure, s.{y} AS y, c.p_exposure,
                                  quantile_cont(c.p_exposure, {1 - top}) OVER () AS thr
                           FROM read_parquet('{CRITEO_PARQUET}') s
                           JOIN read_parquet('{CUPAC_PARQUET}') c USING (unit_id))
                SELECT count(*) FILTER (treatment=1), avg(y) FILTER (treatment=1), var_samp(y) FILTER (treatment=1),
                       count(*) FILTER (treatment=0), avg(y) FILTER (treatment=0), var_samp(y) FILTER (treatment=0),
                       sum(exposure) / (SELECT sum(exposure) FROM read_parquet('{CRITEO_PARQUET}')), count(*)
                FROM u WHERE p_exposure >= thr"""
        ).fetchone()
        assert r is not None
        n_t, m_t, v_t, n_c, m_c, v_c, exp_share, n_trig = r
        share = n_trig / n_all[0]
        itt_trig = m_t - m_c
        se_trig = math.sqrt(v_t / n_t + v_c / n_c)
        diluted, diluted_se = itt_trig * share, se_trig * share
        out.append(
            {
                "top_share_by_predicted_exposure": top,
                "share_of_exposures_captured": float(exp_share),
                "itt_triggered": itt_trig,
                "itt_triggered_se": se_trig,
                "diluted_to_population": diluted,
                "diluted_se": diluted_se,
                "full_itt": itt_full,
                "full_itt_se": itt_full_se,
                "z_full_itt": itt_full / itt_full_se,
                "z_triggered": itt_trig / se_trig,
                "variance_ratio_full_over_diluted": (itt_full_se / diluted_se) ** 2,
                "diluted_within_full_ci": bool(abs(diluted - itt_full) <= Z975 * itt_full_se),
            }
        )
    return out


def main() -> None:
    run()
