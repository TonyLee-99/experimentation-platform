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

import duckdb
import numpy as np
from scipy import stats

from expkit.analyses.common import SEED, write_json
from expkit.data import ASOS_PARQUET
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
        n = self.n_c + self.n_t
        out: np.ndarray = n / n[-1]
        return out

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
        duckdb.connect()
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
        duckdb.connect()
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
def portfolio_eb(trajs: list[Trajectory], n_sim: int = 200) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    out: dict[str, Any] = {}
    for metric in (1, 2, 3, 4):
        tm = [t for t in trajs if t.metric == metric]
        est, se = map(np.array, zip(*[t.final_relative() for t in tm], strict=True))
        normal = fit_normal_prior(est, se)
        npmle = fit_npmle(est, se)
        p_raw = 2 * stats.norm.sf(np.abs(est / se))
        winners = (p_raw < ALPHA) & (est > 0)
        losers = (p_raw < ALPHA) & (est < 0)
        pm_n, _ = normal.posterior(est, se)
        pm_np, _ = npmle.posterior(est, se)
        ppos_n = normal.prob_positive(est, se)
        ppos_np = npmle.prob_positive(est, se)
        r: dict[str, Any] = {
            "n_treatments": len(tm),
            "median_se": float(np.median(se)),
            "normal_prior": {
                "mean": normal.mean,
                "sd": normal.sd,
                "loglik": normal.loglik,
                "win_rate": float(stats.norm.sf(0, normal.mean, normal.sd))
                if normal.sd > 0
                else float(normal.mean > 0),
                "p_true_lift_gt_0.1pct": float(stats.norm.sf(0.001, normal.mean, normal.sd)),
                "p_true_lift_lt_minus_0.1pct": float(stats.norm.cdf(-0.001, normal.mean, normal.sd)),
            },
            "npmle_prior": {
                "mean": npmle.mean,
                "sd": npmle.sd,
                "loglik": npmle.loglik,
                "excess_kurtosis": npmle.excess_kurtosis,
                "win_rate": npmle.prob_positive_prior,
                "mass_within_0.1pct_of_zero": float(npmle.weights[np.abs(npmle.support) <= 0.001].sum()),
                "p_true_lift_gt_0.1pct": float(npmle.weights[npmle.support > 0.001].sum()),
                "p_true_lift_lt_minus_0.1pct": float(npmle.weights[npmle.support < -0.001].sum()),
            },
            "n_winners": int(winners.sum()),
            "n_losers": int(losers.sum()),
        }
        if winners.any():
            r["winners"] = {
                "mean_observed_lift": float(est[winners].mean()),
                "mean_posterior_normal": float(pm_n[winners].mean()),
                "mean_posterior_npmle": float(pm_np[winners].mean()),
                "exaggeration_ratio_normal": float(est[winners].mean() / pm_n[winners].mean()),
                "exaggeration_ratio_npmle": float(est[winners].mean() / pm_np[winners].mean()),
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
        # W2: simulation from the fitted normal prior with the real SEs (labeled)
        cover: list[float] = []
        mse_raw: list[float] = []
        mse_eb: list[float] = []
        mse_np: list[float] = []
        sd_true = max(normal.sd, 1e-6)
        for _ in range(n_sim):
            theta = rng.normal(normal.mean, sd_true, se.size)
            e = theta + rng.normal(0, se)
            pr = fit_normal_prior(e, se)
            m, s = pr.posterior(e, se)
            cover.append(np.mean(np.abs(m - theta) <= Z * s))
            mse_raw.append(np.mean((e - theta) ** 2))
            mse_eb.append(np.mean((m - theta) ** 2))
            if len(mse_np) < 40:  # NPMLE is slower; 40 sims is enough for an MSE comparison
                mnp, _ = fit_npmle(e, se, grid_size=200).posterior(e, se)
                mse_np.append(np.mean((mnp - theta) ** 2))
        r["simulation_check"] = {
            "label": "SIMULATION: true effects drawn from the fitted normal prior, noise from real SEs",
            "n_sim": n_sim,
            "coverage_95": float(np.mean(cover)),
            "mse_raw": float(np.mean(mse_raw)),
            "mse_eb_normal": float(np.mean(mse_eb)),
            "mse_eb_npmle": float(np.mean(mse_np)),
        }
        # B2: Bayes expected loss vs p-value rule, prior = NPMLE-smoothed normal approx of the portfolio
        r["decisions"] = _decision_comparison(est, se, normal.mean, normal.sd)
        out[f"metric_{metric}"] = r
    return out


def _decision_comparison(est: np.ndarray, se: np.ndarray, mu: float, tau: float) -> dict[str, Any]:
    p = 2 * stats.norm.sf(np.abs(est / se))
    pval_ship = (p < ALPHA) & (est > 0)
    res = {}
    for eps in (0.0001, 0.0005, 0.001, 0.005):
        bayes_ship = np.array(
            [bayes_decision(e, s, eps, mu, max(tau, 1e-9)).ship for e, s in zip(est, se, strict=True)]
        )
        res[f"eps_{eps}"] = {
            "pvalue_ship": int(pval_ship.sum()),
            "bayes_ship": int(bayes_ship.sum()),
            "pvalue_yes_bayes_no": int((pval_ship & ~bayes_ship).sum()),
            "pvalue_no_bayes_yes": int((~pval_ship & bayes_ship).sum()),
        }
    return res


# ------------------------------------------------------------------------------- peeking
def _tau2_rel(eb: dict[str, Any], metric: int, se_final_rel: float) -> float:
    """mSPRT mixing variance (relative-lift units): the portfolio prior sd from EB, floored at the
    final SE so the test keeps power if the fitted prior collapses to ~0."""
    tau = eb[f"metric_{metric}"]["normal_prior"]["sd"]
    return float(max(tau, se_final_rel) ** 2)


def real_peeking(trajs: list[Trajectory], eb: dict[str, Any]) -> dict[str, Any]:
    """P1/P2 on the actual trajectories (no ground truth)."""
    rows = []
    for t in trajs:
        if t.day.size < 3:
            continue
        z = t.z
        p_naive = 2 * stats.norm.sf(np.abs(z))
        final_sig = bool(p_naive[-1] < ALPHA)
        final_sign = np.sign(z[-1])
        # mSPRT on the absolute scale; tau converted from relative units with the final control mean
        _, se_rel = t.final_relative()
        tau2 = _tau2_rel(eb, t.metric, se_rel) * t.mean_c[-1] ** 2
        p_ms = msprt_pvalues(t.delta, t.var_delta, tau2)
        tf = t.info_fraction
        keep = np.concatenate([[True], np.diff(tf) > 0])
        b = np.full(z.size, np.inf)
        b[keep] = gsd_boundaries(tf[keep]).z_bounds
        rec: dict[str, Any] = {
            "key": t.key,
            "metric": t.metric,
            "days": float(t.day[-1]),
            "final_sig": final_sig,
        }
        for name, hit in (
            ("naive", p_naive < ALPHA),
            ("msprt", p_ms < ALPHA),
            ("obf", np.abs(z) >= b),
        ):
            idx = np.flatnonzero(hit)
            if idx.size:
                k = int(idx[0])
                rec[f"{name}_stop_day"] = float(t.day[k])
                rec[f"{name}_unsupported"] = bool((not final_sig) or np.sign(z[k]) != final_sign)
            else:
                rec[f"{name}_stop_day"] = None
                rec[f"{name}_unsupported"] = False
        rows.append(rec)
    n = len(rows)
    summary: dict[str, Any] = {
        "n_trajectories": n,
        "n_final_significant": sum(r["final_sig"] for r in rows),
    }
    for name in ("naive", "msprt", "obf"):
        stopped = [r for r in rows if r[f"{name}_stop_day"] is not None]
        unsup = [r for r in stopped if r[f"{name}_unsupported"]]
        sig = [r for r in rows if r["final_sig"]]
        detected = [r for r in sig if r[f"{name}_stop_day"] is not None and not r[f"{name}_unsupported"]]
        frac_time = [r[f"{name}_stop_day"] / r["days"] for r in detected]
        summary[name] = {
            "declared_significant": len(stopped),
            "declared_but_final_does_not_support": len(unsup),
            "share_of_all_trajectories_unsupported": len(unsup) / n,
            "detection_rate_among_final_significant": len(detected) / max(1, len(sig)),
            "mean_stop_day_detected": float(np.mean([r[f"{name}_stop_day"] for r in detected])) if detected else None,
            "median_stop_day_detected": float(np.median([r[f"{name}_stop_day"] for r in detected]))
            if detected
            else None,
            "mean_stop_fraction_of_horizon": float(np.mean(frac_time)) if frac_time else None,
        }
    summary["median_horizon_days"] = float(np.median([r["days"] for r in rows]))
    return {"summary": summary, "rows": rows}


def null_peeking_simulation(trajs: list[Trajectory], eb: dict[str, Any], reps_per_traj: int = 250) -> dict[str, Any]:
    """SIMULATION: real look schedules + information growth, null (zero-effect) Brownian paths.

    For each real trajectory we keep its actual daily look times and its actual information
    fractions t_k = n_k / n_K, and draw z-paths from the null process Z_k = B(t_k)/√t_k (the
    canonical joint distribution assumed by group-sequential theory). mSPRT uses the real final
    variance scaled as V_k = V_K / t_k.
    """
    rng = np.random.default_rng(SEED + 7)
    hits = {"naive": 0, "msprt": 0, "obf": 0, "fixed_horizon": 0}
    per_metric: dict[int, dict[str, int]] = {}
    total = 0
    looks = []
    for t in trajs:
        tf = t.info_fraction
        keep = np.concatenate([[True], np.diff(tf) > 0])
        tf = tf[keep]
        if tf.size < 3:
            continue
        looks.append(tf.size)
        b = gsd_boundaries(tf).z_bounds
        inc = rng.standard_normal((reps_per_traj, tf.size)) * np.sqrt(np.diff(np.concatenate([[0.0], tf])))
        Zp = inc.cumsum(axis=1) / np.sqrt(tf)
        vK = float(t.var_delta[-1])
        V = vK / tf
        _, se_rel = t.final_relative()
        tau2 = _tau2_rel(eb, t.metric, se_rel) * t.mean_c[-1] ** 2
        p_ms = msprt_pvalues(Zp * np.sqrt(V), np.broadcast_to(V, Zp.shape), tau2)
        res = {
            "naive": (np.abs(Zp) >= Z).any(axis=1),
            "msprt": (p_ms < ALPHA).any(axis=1),
            "obf": (np.abs(Zp) >= b).any(axis=1),
            "fixed_horizon": np.abs(Zp[:, -1]) >= Z,
        }
        pm = per_metric.setdefault(t.metric, {k: 0 for k in [*hits, "n"]})
        for k, v in res.items():
            hits[k] += int(v.sum())
            pm[k] += int(v.sum())
        pm["n"] += reps_per_traj
        total += reps_per_traj
    out: dict[str, Any] = {
        "label": "SIMULATION: null paths on the real ASOS look schedules / information growth",
        "n_paths": total,
        "reps_per_trajectory": reps_per_traj,
        "median_looks": float(np.median(looks)),
        "fpr": {},
        "fpr_by_metric": {},
    }
    for k, v in hits.items():
        lo, hi = stats.binomtest(v, total).proportion_ci(0.95)
        out["fpr"][k] = {"fpr": v / total, "ci95": [lo, hi]}
    for m, d in per_metric.items():
        out["fpr_by_metric"][f"metric_{m}"] = {k: d[k] / d["n"] for k in hits}
    return out


def main() -> dict[str, Any]:
    t0 = time.time()
    trajs = load_daily_trajectories()
    eb = portfolio_eb(trajs)
    out = {
        "n_daily_trajectories": len(trajs),
        "srm": srm_screen(),
        "eb": eb,
        "real_peeking": real_peeking(trajs, eb),
        "null_peeking_simulation": null_peeking_simulation(trajs, eb),
    }
    out["seconds"] = round(time.time() - t0, 1)
    write_json("asos_portfolio", out)
    return out
