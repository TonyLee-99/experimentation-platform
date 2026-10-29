"""ASOS Digital Experiments Dataset: the company's experiment *portfolio*.

78 real experiments / 99 treatments / 4 decision metrics with 12-hourly or daily cumulative
snapshots of (n, mean, variance) per arm. Ground truth is unknown, so this module does four
things that need no ground truth, plus one labeled simulation:

1. SRM screen of every treatment (assuming equal allocation, which the datasheet does not
   state — reported as an assumption).
2. Empirical Bayes across the portfolio: prior of true effects, winner's curse, true win rate,
   FDR among "winners" (normal prior and nonparametric NPMLE prior).
3. Real peeking evidence: on the actual daily trajectories, how often naive daily peeking
   "finds" a winner/loser that the final analysis does not support, vs mSPRT and OBF, and the
   stopping time of each.
4. Bayesian expected-loss vs p-value decisions on the same portfolio.
5. SIMULATION (labeled): null trajectories anchored on the real look schedules and information
   growth, to measure each method's false-positive rate under daily peeking.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from expkit.analyses.common import SEED, write_json
from expkit.data import ASOS_PARQUET
from expkit.db import duckdb_connect
from expkit.stats.bayes import bayes_decision
from expkit.stats.eb import fit_normal_prior, fit_npmle
from expkit.stats.sequential import gsd_boundaries, msprt_pvalues
from expkit.stats.srm import srm_check

ALPHA = 0.05
Z = float(stats.norm.ppf(1 - ALPHA / 2))


@dataclass
class Trajectory:
    experiment: str
    variant: int
    metric: int
    day: np.ndarray
    n_c: np.ndarray
    n_t: np.ndarray
    mean_c: np.ndarray
    mean_t: np.ndarray
    var_c: np.ndarray
    var_t: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.experiment}-v{self.variant}-m{self.metric}"

    @property
    def delta(self) -> np.ndarray:
        out: np.ndarray = self.mean_t - self.mean_c
        return out

    @property
    def var_delta(self) -> np.ndarray:
        out: np.ndarray = self.var_t / self.n_t + self.var_c / self.n_c
        return out

    @property
    def z(self) -> np.ndarray:
        out: np.ndarray = self.delta / np.sqrt(self.var_delta)
        return out

    @property
    def info_fraction(self) -> np.ndarray:
        """n-based fraction n_k / n_K. Kept for reference only: ASOS metrics accumulate per user, so the
        variance of the estimate does not shrink like 1/n (see :meth:`variance_info`)."""
        n = self.n_c + self.n_t
        out: np.ndarray = n / n[-1]
        return out

    @property
    def rel(self) -> np.ndarray:
        """Relative lift at each look."""
        out: np.ndarray = self.mean_t / self.mean_c - 1
        return out

    @property
    def var_rel(self) -> np.ndarray:
        """Delta-method variance of the relative lift at each look."""
        vc, vt = self.var_c / self.n_c, self.var_t / self.n_t
        out: np.ndarray = vt / self.mean_c**2 + self.mean_t**2 * vc / self.mean_c**4
        return out

    @property
    def z_rel(self) -> np.ndarray:
        out: np.ndarray = self.rel / np.sqrt(self.var_rel)
        return out

    def variance_info(self) -> tuple[np.ndarray, np.ndarray]:
        """Variance-based information fractions I_k / I_K (I = 1/Var of the relative lift), at the looks where
        information reaches a new maximum (Lan–DeMets requires increasing information). Returns (mask, t)."""
        info = 1.0 / self.var_rel
        run_max = np.maximum.accumulate(info)
        # a look counts only if information grows by > 0.1% (near-duplicate looks add nothing but numerical noise)
        keep = (info >= run_max) & np.concatenate([[True], np.diff(run_max) / run_max[1:] > 1e-3])
        frac = info / info[keep][-1]
        return keep, frac

    def final_relative(self) -> tuple[float, float]:
        """Final relative lift and delta-method SE."""
        mc, mt = self.mean_c[-1], self.mean_t[-1]
        vc, vt = self.var_c[-1] / self.n_c[-1], self.var_t[-1] / self.n_t[-1]
        rel = mt / mc - 1
        se = math.sqrt(vt / mc**2 + mt**2 * vc / mc**4)
        return float(rel), float(se)


def load_daily_trajectories() -> list[Trajectory]:
    """One look per day: the last snapshot with time_since_start <= d, for d = 1, 2, ..."""
    df = (
        duckdb_connect()
        .execute(
            f"""
        WITH x AS (
          SELECT *, ceil(time_since_start) AS day,
                 row_number() OVER (PARTITION BY experiment_id, variant_id, metric_id, ceil(time_since_start)
                                    ORDER BY time_since_start DESC) AS rk
          FROM read_parquet('{ASOS_PARQUET}')
          WHERE variance_c IS NOT NULL AND variance_t IS NOT NULL
            AND variance_c > 0 AND variance_t > 0 AND mean_c > 0)
        SELECT * FROM x WHERE rk = 1 ORDER BY experiment_id, variant_id, metric_id, day
        """
        )
        .df()
    )
    out = []
    for (e, v, m), g in df.groupby(["experiment_id", "variant_id", "metric_id"], sort=True):
        if len(g) < 2:
            continue
        out.append(
            Trajectory(
                str(e),
                int(v),
                int(m),
                *(
                    g[c].to_numpy(np.float64)
                    for c in (
                        "day",
                        "count_c",
                        "count_t",
                        "mean_c",
                        "mean_t",
                        "variance_c",
                        "variance_t",
                    )
                ),
            )
        )
    return out


# ------------------------------------------------------------------------------------- SRM
def srm_screen() -> dict[str, Any]:
    df = (
        duckdb_connect()
        .execute(
            f"""SELECT experiment_id, variant_id, arg_max(count_c, time_since_start) c,
                   arg_max(count_t, time_since_start) t
            FROM read_parquet('{ASOS_PARQUET}') WHERE metric_id = 1 GROUP BY ALL"""
        )
        .df()
    )
    res = [srm_check([r.c, r.t], [0.5, 0.5]) for r in df.itertuples()]
    flagged = [f"{r.experiment_id}-v{r.variant_id}" for r, s in zip(df.itertuples(), res, strict=True) if s.blocked]
    shares = np.array([s.observed_share[1] for s in res])
    return {
        "assumption": "equal allocation between each treatment and its control (not stated in datasheet)",
        "n_treatments": len(res),
        "n_srm_flagged_p_lt_0.0005": len(flagged),
        "flagged": flagged,
        "treatment_share_quantiles": np.quantile(shares, [0, 0.05, 0.5, 0.95, 1]).tolist(),
    }


# ------------------------------------------------------------------------------------- EB
def _final(trajs: list[Trajectory], metric: int) -> tuple[list[Trajectory], np.ndarray, np.ndarray]:
    tm = [t for t in trajs if t.metric == metric]
    est, se = map(np.array, zip(*[t.final_relative() for t in tm], strict=True))
    return tm, est, se


def portfolio_eb(trajs: list[Trajectory], n_sim: int = 200, n_boot: int = 100) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    out: dict[str, Any] = {}
    for metric in (1, 2, 3, 4):
        tm, est, se = _final(trajs, metric)
        normal = fit_normal_prior(est, se)
        npmle = fit_npmle(est, se)
        p_raw = 2 * stats.norm.sf(np.abs(est / se))
        winners = (p_raw < ALPHA) & (est > 0)
        losers = (p_raw < ALPHA) & (est < 0)
        pm_n, _ = normal.posterior(est, se)
        pm_np, _ = npmle.posterior(est, se)
        ppos_n = normal.prob_positive(est, se)
        ppos_np = npmle.prob_positive(est, se)
        # bootstrap uncertainty of the deconvolved portfolio functionals (resample treatments)
        b_gt, b_lt, b_k = [], [], []
        med = float(np.median(se))
        obs_grid = np.array([0.003, 0.006, 0.01, 0.02])
        b_hair = []
        for _ in range(n_boot):
            i = rng.integers(0, est.size, est.size)
            g = fit_npmle(est[i], se[i], grid_size=200, max_iter=2000)
            b_gt.append(float(g.weights[g.support > 0.001].sum()))
            b_lt.append(float(g.weights[g.support < -0.001].sum()))
            b_k.append(g.excess_kurtosis)
            b_hair.append(g.posterior(obs_grid, np.full(obs_grid.size, med))[0])
        hair_ci = np.quantile(np.array(b_hair), [0.05, 0.95], axis=0)
        r: dict[str, Any] = {
            "n_treatments": len(tm),
            "median_se": float(np.median(se)),
            "normal_prior": {
                "mean": normal.mean,
                "sd": normal.sd,
                "loglik": normal.loglik,
                "p_true_lift_gt_0.1pct": float(stats.norm.sf(0.001, normal.mean, normal.sd)),
            },
            "npmle_prior": {
                "mean": npmle.mean,
                "sd": npmle.sd,
                "loglik": npmle.loglik,
                "excess_kurtosis": npmle.excess_kurtosis,
                "excess_kurtosis_ci90": np.quantile(b_k, [0.05, 0.95]).tolist(),
                "p_true_lift_gt_0.1pct": float(npmle.weights[npmle.support > 0.001].sum()),
                "p_true_lift_gt_0.1pct_ci90": np.quantile(b_gt, [0.05, 0.95]).tolist(),
                "p_true_lift_lt_minus_0.1pct": float(npmle.weights[npmle.support < -0.001].sum()),
                "p_true_lift_lt_minus_0.1pct_ci90": np.quantile(b_lt, [0.05, 0.95]).tolist(),
                "mass_within_0.1pct_of_zero": float(npmle.weights[np.abs(npmle.support) <= 0.001].sum()),
            },
            "bootstrap_note": f"{n_boot} resamples of treatments; 0.1% is below the median SE, so the split "
            "of mass around ±0.1% is weakly identified",
            "n_winners": int(winners.sum()),
            "n_losers": int(losers.sum()),
        }
        if winners.any():
            small = winners & (est < 0.01)
            r["winners"] = {
                "mean_observed_lift": float(est[winners].mean()),
                "mean_posterior_normal": float(pm_n[winners].mean()),
                "mean_posterior_npmle": float(pm_np[winners].mean()),
                "exaggeration_ratio_of_means_normal": float(est[winners].mean() / pm_n[winners].mean()),
                "exaggeration_ratio_of_means_npmle": float(est[winners].mean() / pm_np[winners].mean()),
                "n_small_winners_under_1pct": int(small.sum()),
                "small_winners_median_ratio_npmle": float(np.median(est[small] / pm_np[small]))
                if small.any()
                else None,
                "small_winners_median_ratio_normal": float(np.median(est[small] / pm_n[small]))
                if small.any()
                else None,
                "fdr_normal": float((1 - ppos_n[winners]).mean()),
                "fdr_npmle": float((1 - ppos_np[winners]).mean()),
                "rows": [
                    {
                        "key": tm[i].key,
                        "observed": float(est[i]),
                        "se": float(se[i]),
                        "post_normal": float(pm_n[i]),
                        "post_npmle": float(pm_np[i]),
                        "p_pos_npmle": float(ppos_np[i]),
                    }
                    for i in np.flatnonzero(winners)
                ],
            }
        # PM haircut: a hypothetical observed lift at this metric's median SE -> NPMLE posterior mean, with a 90%
        # bootstrap interval over prior refits (the discrete NPMLE's own posterior intervals are too narrow to quote)
        pm_grid, _ = npmle.posterior(obs_grid, np.full(obs_grid.size, med))
        haircut = [
            {
                "observed": float(obs),
                "z": float(obs / med),
                "posterior_mean": float(pm_grid[j]),
                "posterior_mean_bootstrap_ci90": [float(hair_ci[0, j]), float(hair_ci[1, j])],
            }
            for j, obs in enumerate(obs_grid)
        ]
        r["haircut_at_median_se"] = haircut
        r["simulation_check"] = _eb_simulation(est, se, normal, npmle, n_sim, rng)
        r["decisions"] = _policy_comparison(est, se, normal, npmle)
        out[f"metric_{metric}"] = r
    return out


def _eb_simulation(
    est: np.ndarray, se: np.ndarray, normal: Any, npmle: Any, n_sim: int, rng: np.random.Generator
) -> dict[str, Any]:
    """SIMULATION: (a) truth from the fitted normal prior (self-consistent check); (b) truth from the fitted
    fat-tailed NPMLE prior (misspecified for the normal EB), reporting coverage among significant winners too."""
    out: dict[str, Any] = {"label": "SIMULATION: true effects drawn from a fitted prior, noise from the real SEs"}
    for truth in ("normal", "npmle"):
        cover, cover_win, mse_raw, mse_eb = [], [], [], []
        for _ in range(n_sim):
            if truth == "normal":
                theta = rng.normal(normal.mean, max(normal.sd, 1e-6), se.size)
            else:
                theta = rng.choice(npmle.support, se.size, p=npmle.weights)
            e = theta + rng.normal(0, se)
            pr = fit_normal_prior(e, se)
            m, s = pr.posterior(e, se)
            ok = np.abs(m - theta) <= Z * s
            cover.append(float(ok.mean()))
            w = (np.abs(e / se) > Z) & (e > 0)
            if w.any():
                cover_win.append(float(ok[w].mean()))
            mse_raw.append(float(np.mean((e - theta) ** 2)))
            mse_eb.append(float(np.mean((m - theta) ** 2)))
        out[f"truth_{truth}"] = {
            "n_sim": n_sim,
            "coverage_95_normal_eb": float(np.mean(cover)),
            "coverage_95_among_winners": float(np.mean(cover_win)) if cover_win else None,
            "mse_raw": float(np.mean(mse_raw)),
            "mse_eb_normal": float(np.mean(mse_eb)),
        }
    # backwards-compatible keys used by the acceptance check W2 (pre-registered on the self-consistent case)
    out["coverage_95"] = out["truth_normal"]["coverage_95_normal_eb"]
    out["mse_raw"] = out["truth_normal"]["mse_raw"]
    out["mse_eb_normal"] = out["truth_normal"]["mse_eb_normal"]
    return out


def _policy_comparison(est: np.ndarray, se: np.ndarray, normal: Any, npmle: Any) -> dict[str, Any]:
    """Compare ship rules as *portfolio policies* under one utility.

    Expected-loss rule: normal prior (MLE). Truth model for scoring every rule: the NPMLE posterior.
    Utility of shipping treatment i at launch cost c (relative-lift units): E[θ_i | data] − c.
    """
    pm_np, _ = npmle.posterior(est, se)
    p_harm = 1 - npmle.prob_positive(est, se)
    pm_n, _ = normal.posterior(est, se)
    p = 2 * stats.norm.sf(np.abs(est / se))
    med_se = float(np.median(se))
    loss = np.array(
        [bayes_decision(e, s, 1.0, normal.mean, max(normal.sd, 1e-9)).loss_ship for e, s in zip(est, se, strict=True)]
    )
    rules = {
        "p_value (p<0.05, positive)": (p < ALPHA) & (est > 0),
        "expected loss < 0.1% (fixed eps)": loss < 0.001,
        "expected loss < 0.01% (fixed eps)": loss < 0.0001,
        "expected loss < 0.1 x median SE (eps tied to SE)": loss < 0.1 * med_se,
        "posterior mean > 0": pm_n > 0,
        "P(theta>0) > 0.95 (NPMLE)": (1 - p_harm) > 0.95,
    }
    out: dict[str, Any] = {
        "truth_model": "NPMLE posterior",
        "loss_rule_prior": "normal (MLE)",
        "median_se": med_se,
        "rules": {},
    }
    for name, ship in rules.items():
        r = {
            "n_ship": int(ship.sum()),
            "n_ship_negative_posterior_mean": int((ship & (pm_n < 0)).sum()),
            "expected_total_lift": float(pm_np[ship].sum()),
            "expected_harmful_launches": float(p_harm[ship].sum()),
        }
        for c in (0.0, 0.0005, 0.001):
            r[f"expected_value_cost_{c}"] = float((pm_np[ship] - c).sum())
        out["rules"][name] = r
    return out


# ------------------------------------------------------------------------------- peeking
def _tau_rel(eb: dict[str, Any], metric: int) -> float:
    """mSPRT mixing sd (relative-lift units) = the portfolio prior sd from the normal EB fit. Fitted on the same
    ASOS data (in production it should come from *past* experiments); any fixed τ keeps type-I error <= α."""
    return float(eb[f"metric_{metric}"]["normal_prior"]["sd"])


def _methods_on_path(
    z: np.ndarray, var: np.ndarray, keep: np.ndarray, frac: np.ndarray, tau2: float
) -> dict[str, np.ndarray]:
    """Rejection indicator per look for naive, OBF (variance-based information) and mSPRT (relative scale)."""
    obf = np.full(z.size, np.inf)
    obf[keep] = gsd_boundaries(frac[keep]).z_bounds
    p_ms = msprt_pvalues(z * np.sqrt(var), var, tau2)
    return {"naive": np.abs(z) >= Z, "obf": np.abs(z) >= obf, "msprt": p_ms < ALPHA}


def real_peeking(trajs: list[Trajectory], eb: dict[str, Any]) -> dict[str, Any]:
    """P1/P2 on the actual trajectories (no ground truth), relative scale, variance-based information."""
    rows = []
    for t in trajs:
        if t.day.size < 3:
            continue
        z, var = t.z_rel, t.var_rel
        keep, frac = t.variance_info()
        hits = _methods_on_path(z, var, keep, frac, _tau_rel(eb, t.metric) ** 2)
        final_sig = bool(abs(z[-1]) >= Z)
        rec: dict[str, Any] = {
            "key": t.key,
            "metric": t.metric,
            "days": float(t.day[-1]),
            "final_sig": final_sig,
            "n_to_var_info_max_abs_gap": float(np.max(np.abs(t.info_fraction - frac))),
        }
        for name, hit in hits.items():
            idx = np.flatnonzero(hit)
            if idx.size:
                k = int(idx[0])
                rec[f"{name}_stop_day"] = float(t.day[k])
                rec[f"{name}_unsupported"] = bool((not final_sig) or np.sign(z[k]) != np.sign(z[-1]))
                rec[f"{name}_sign_reversal"] = bool(np.sign(z[k]) != np.sign(z[-1]))
                rec[f"{name}_strong_reversal"] = bool(final_sig and np.sign(z[k]) != np.sign(z[-1]))
            else:
                rec[f"{name}_stop_day"] = None
                rec[f"{name}_unsupported"] = rec[f"{name}_sign_reversal"] = rec[f"{name}_strong_reversal"] = False
        rows.append(rec)
    n = len(rows)
    sig = [r for r in rows if r["final_sig"]]
    summary: dict[str, Any] = {
        "n_trajectories": n,
        "n_final_significant": len(sig),
        "median_horizon_days": float(np.median([r["days"] for r in rows])),
        "median_max_gap_n_vs_variance_information": float(np.median([r["n_to_var_info_max_abs_gap"] for r in rows])),
    }
    for name in ("naive", "msprt", "obf"):
        stopped = [r for r in rows if r[f"{name}_stop_day"] is not None]
        detected = [r for r in sig if r[f"{name}_stop_day"] is not None and not r[f"{name}_unsupported"]]
        summary[name] = {
            "declared_significant": len(stopped),
            "declared_but_final_does_not_support": sum(r[f"{name}_unsupported"] for r in stopped),
            "sign_reversals": sum(r[f"{name}_sign_reversal"] for r in stopped),
            "strong_reversals_final_significant_opposite": sum(r[f"{name}_strong_reversal"] for r in stopped),
            "detection_rate_among_final_significant": len(detected) / max(1, len(sig)),
        }
    both = [
        r for r in sig if all(r[f"{m}_stop_day"] is not None and not r[f"{m}_unsupported"] for m in ("msprt", "obf"))
    ]
    summary["stopping_on_common_set"] = {
        "n_detected_by_both_msprt_and_obf": len(both),
        "median_stop_day_msprt": float(np.median([r["msprt_stop_day"] for r in both])) if both else None,
        "median_stop_day_obf": float(np.median([r["obf_stop_day"] for r in both])) if both else None,
        "median_stop_fraction_msprt": float(np.median([r["msprt_stop_day"] / r["days"] for r in both]))
        if both
        else None,
        "median_stop_fraction_obf": float(np.median([r["obf_stop_day"] / r["days"] for r in both])) if both else None,
    }
    return {"summary": summary, "rows": rows}


def _simulate_paths(t: Trajectory, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brownian information-time model on a real trajectory's variance-based schedule:
    Z_k = B(t_k)/√t_k + (θ / SE_K)·√t_k, at the looks where information increases."""
    keep, frac = t.variance_info()
    tf = frac[keep]
    se_final = math.sqrt(float(t.var_rel[keep][-1]))
    inc = rng.standard_normal((theta.size, tf.size)) * np.sqrt(np.diff(np.concatenate([[0.0], tf])))
    out: np.ndarray = inc.cumsum(axis=1) / np.sqrt(tf) + (theta / se_final)[:, None] * np.sqrt(tf)[None, :]
    return out


def model_benchmark(
    trajs: list[Trajectory], eb: dict[str, Any], priors: dict[int, Any], reps: int = 50
) -> dict[str, Any]:
    """MODEL-BASED expectation of what naive peeking would show on these 361 schedules, under the pure null and under
    true effects drawn from the fitted NPMLE prior, vs what the real trajectories show."""
    rng = np.random.default_rng(SEED + 11)
    res: dict[str, dict[str, float]] = {}
    for label in ("null", "npmle_effects"):
        acc = {"declared": 0.0, "unsupported": 0.0, "sign_reversals": 0.0, "unsupported_but_true_effect_same_sign": 0.0}
        for t in trajs:
            if t.day.size < 3:
                continue
            g = priors[t.metric]
            theta = np.zeros(reps) if label == "null" else rng.choice(g.support, reps, p=g.weights)
            Zp = _simulate_paths(t, theta, rng)
            hit = np.abs(Zp) >= Z
            decl = hit.any(axis=1)
            k = np.argmax(hit, axis=1)
            first_sign = np.sign(Zp[np.arange(reps), k])
            final_sig = np.abs(Zp[:, -1]) >= Z
            unsup = decl & ((~final_sig) | (first_sign != np.sign(Zp[:, -1])))
            acc["declared"] += decl.sum() / reps
            acc["unsupported"] += unsup.sum() / reps
            acc["sign_reversals"] += (decl & (first_sign != np.sign(Zp[:, -1]))).sum() / reps
            acc["unsupported_but_true_effect_same_sign"] += (
                unsup & (np.sign(theta) == first_sign) & (np.abs(theta) > 0.001)
            ).sum() / reps
        res[label] = acc
    return {
        "label": "MODEL-BASED: Brownian information-time model on the real variance-based schedules",
        "reps": reps,
        "expected_per_361": res,
    }


def null_peeking_simulation(trajs: list[Trajectory], eb: dict[str, Any], reps_per_traj: int = 250) -> dict[str, Any]:
    """MODEL-BASED null simulation (not a validation of OBF on real data).

    Uses each real trajectory's look days and its *variance-based* information schedule, and draws z-paths from the
    Brownian null Z_k = B(t_k)/√t_k — exactly the model the OBF boundaries are computed for, so OBF ≈ 5% holds by
    construction. What it does measure is how much naive daily peeking inflates the error under that model.
    """
    rng = np.random.default_rng(SEED + 7)
    hits = {"naive": 0, "msprt": 0, "obf": 0, "fixed_horizon": 0}
    total = 0
    looks = []
    for t in trajs:
        keep, frac = t.variance_info()
        if keep.sum() < 3:
            continue
        tf = frac[keep]
        looks.append(int(t.day.size))
        Zp = _simulate_paths(t, np.zeros(reps_per_traj), rng)
        var = float(t.var_rel[keep][-1]) / tf
        b = gsd_boundaries(tf).z_bounds
        p_ms = msprt_pvalues(Zp * np.sqrt(var), np.broadcast_to(var, Zp.shape), _tau_rel(eb, t.metric) ** 2)
        hits["naive"] += int((np.abs(Zp) >= Z).any(axis=1).sum())
        hits["msprt"] += int((p_ms < ALPHA).any(axis=1).sum())
        hits["obf"] += int((np.abs(Zp) >= b).any(axis=1).sum())
        hits["fixed_horizon"] += int((np.abs(Zp[:, -1]) >= Z).sum())
        total += reps_per_traj
    out: dict[str, Any] = {
        "label": "MODEL-BASED: Brownian null on the real look schedules (variance-based information); "
        "OBF ~5% by construction",
        "n_paths": total,
        "reps_per_trajectory": reps_per_traj,
        "median_looks": float(np.median(looks)),
        "fpr": {},
    }
    for k, v in hits.items():
        lo, hi = stats.binomtest(v, total).proportion_ci(0.95)
        out["fpr"][k] = {"fpr": v / total, "ci95": [lo, hi]}
    return out


def main() -> dict[str, Any]:
    t0 = time.time()
    trajs = load_daily_trajectories()
    eb = portfolio_eb(trajs)
    priors = {m: fit_npmle(*_final(trajs, m)[1:]) for m in (1, 2, 3, 4)}
    total = (
        duckdb_connect()
        .execute(f"SELECT count(DISTINCT experiment_id || variant_id || metric_id) FROM read_parquet('{ASOS_PARQUET}')")
        .fetchone()
    )
    assert total is not None
    out = {
        "trajectory_counts": {
            "all_in_dataset": int(total[0]),
            "with_valid_variance_and_at_least_2_daily_looks": len(trajs),
            "with_at_least_3_daily_looks_used_for_peeking": sum(t.day.size >= 3 for t in trajs),
        },
        "n_daily_trajectories": len(trajs),
        "srm": srm_screen(),
        "eb": eb,
        "real_peeking": real_peeking(trajs, eb),
        "model_benchmark": model_benchmark(trajs, eb, priors),
        "null_peeking_simulation": null_peeking_simulation(trajs, eb),
    }
    out["seconds"] = round(time.time() - t0, 1)
    write_json("asos_portfolio", out)
    return out
