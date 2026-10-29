"""A/A tests and power curves on *real* Criteo control units (simulation on real data).

Why this matters: a stats engine can be "correct" on paper and still mis-calibrated on real
metrics (sparse conversions, heavy tails, features that interact with the adjustment). The
platform-team check is to take real units that received the *same* experience — here the
2,096,937 Criteo control users, who were never shown an ad — randomly split them thousands of
times and count how often each method declares a difference. A calibrated method says
"significant" ~5% of the time at α = 0.05, and its p-values are uniform.

Power: same splits, but a known effect is injected into one half (each non-visitor in the
"treatment" half becomes a visitor with probability q). Then we count how often the effect is
detected and compare with the analytic power formula. Both are labeled SIMULATION.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from scipy import stats

from expkit.analyses.common import SEED, load_criteo, write_json
from expkit.data import CRITEO_FEATURES
from expkit.stats.bootstrap import BootstrapConfig, bootstrap_diff
from expkit.stats.power import power_two_sample
from expkit.stats.sequential import gsd_boundaries, msprt_pvalues
from expkit.validation.vectorized import check_against_engine, delta_p, welch_p

ALPHA = 0.05


def _theta(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    out: np.ndarray = np.linalg.solve(Xc.T @ Xc, Xc.T @ (y - y.mean()))
    return out


def load_control_units() -> dict[str, np.ndarray]:
    """Real control units with raw, CUPED-adjusted and CUPAC-adjusted outcomes.

    θ is fitted on all control units pooled. In an A/A split the pooled-θ of Deng et al. (2013)
    is computed on exactly these units, so fixing it up front is the same estimator.
    """
    d = load_criteo(
        [*CRITEO_FEATURES, "visit", "conversion", "cupac_visit", "cupac_conversion"],
        where="treatment = 0",
        with_cupac=True,
    )
    X = np.column_stack([d[f] for f in CRITEO_FEATURES]).astype(np.float64)
    out: dict[str, np.ndarray] = {}
    for m, cupac in (("visit", "cupac_visit"), ("conversion", "cupac_conversion")):
        y = d[m].astype(np.float64)
        th = _theta(y, X)
        out[m] = y
        out[f"{m}_cuped"] = y - (X - X.mean(axis=0)) @ th
        p = d[cupac].astype(np.float64)[:, None]
        thp = _theta(y, p)
        out[f"{m}_cupac"] = y - (p[:, 0] - p.mean()) * thp[0]
    return out


SERIES = [
    "visit",
    "visit_cuped",
    "visit_cupac",
    "conversion",
    "conversion_cuped",
    "conversion_cupac",
]


def _design(u: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    cols, names = [np.ones_like(u["visit"])], ["n"]
    for s in SERIES:
        cols += [u[s], u[s] ** 2]
        names += [f"{s}__s", f"{s}__ss"]
    cols.append(u["conversion"] * u["visit"])
    names.append("conv_x_visit")
    return np.column_stack(cols), names


def run_aa(n_reps: int = 10_000, batch: int = 25, seed: int = SEED, share_t: float = 0.5) -> dict[str, Any]:
    t0 = time.time()
    u = load_control_units()
    n = u["visit"].size
    D, names = _design(u)
    idx = {k: i for i, k in enumerate(names)}
    total = D.sum(axis=0)
    rng = np.random.default_rng(seed)
    pvals: dict[str, list[np.ndarray]] = {k: [] for k in [*SERIES, "conversion_per_visit_delta"]}
    deltas: dict[str, list[np.ndarray]] = {k: [] for k in SERIES}

    first = rng.random(n) < 0.5
    check_against_engine(u["visit"], first.astype(int))
    check_against_engine(u["conversion"], first.astype(int), u["visit"])

    for _ in range(n_reps // batch):
        W = (rng.random((batch, n)) < share_t).astype(np.float64)
        T = W @ D  # treatment-arm sums, shape (batch, k)
        C = total[None, :] - T
        for s in SERIES:
            d, _, p = welch_p(
                C[:, 0],
                C[:, idx[f"{s}__s"]],
                C[:, idx[f"{s}__ss"]],
                T[:, 0],
                T[:, idx[f"{s}__s"]],
                T[:, idx[f"{s}__ss"]],
            )
            pvals[s].append(p)
            deltas[s].append(d)
        # ratio conversion/visit (binary: y² = y, x² = x)
        y, x, xy = idx["conversion__s"], idx["visit__s"], idx["conv_x_visit"]
        _, _, p = delta_p(
            C[:, 0],
            C[:, y],
            C[:, x],
            C[:, y],
            C[:, x],
            C[:, xy],
            T[:, 0],
            T[:, y],
            T[:, x],
            T[:, y],
            T[:, x],
            T[:, xy],
        )
        pvals["conversion_per_visit_delta"].append(p)

    res: dict[str, Any] = {
        "label": f"SIMULATION: A/A re-randomization of real Criteo control units ({share_t:.0%} 'treatment')",
        "share_treatment": share_t,
        "n_units": n,
        "n_reps": (n_reps // batch) * batch,
        "alpha": ALPHA,
        "methods": {},
    }
    for k, ps in pvals.items():
        p = np.concatenate(ps)
        fpr = float((p < ALPHA).mean())
        lo, hi = stats.binomtest(int((p < ALPHA).sum()), p.size).proportion_ci(0.95)
        res["methods"][k] = {
            "fpr": fpr,
            "fpr_ci95": [lo, hi],
            "ks_pvalue": float(stats.kstest(p, "uniform").pvalue),
            "n_reps": int(p.size),
        }
    # empirical variance reduction: Var of A/A deltas across reps (an external check of the formula)
    emp = {}
    for m in ("visit", "conversion"):
        v_raw = float(np.var(np.concatenate(deltas[m])))
        for adj in ("cuped", "cupac"):
            emp[f"{m}_{adj}"] = 1 - float(np.var(np.concatenate(deltas[f"{m}_{adj}"]))) / v_raw
        # analytic: 1 - var(adjusted unit values)/var(raw)
        for adj in ("cuped", "cupac"):
            emp[f"{m}_{adj}_analytic"] = 1 - float(np.var(u[f"{m}_{adj}"]) / np.var(u[m]))
    res["variance_reduction_aa"] = emp
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ---------------------------------------------------------------------------- bootstrap A/A
_BOOT: dict[str, np.ndarray] = {}


def _init_boot(visit: np.ndarray, conv: np.ndarray) -> None:
    _BOOT["visit"], _BOOT["conversion"] = visit, conv


def _boot_rep(args: tuple[int, int, int]) -> tuple[float, float, float, float]:
    seed, sub_n, n_boot = args
    rng = np.random.default_rng(seed)
    idx = rng.choice(_BOOT["visit"].size, size=sub_n, replace=False)
    a = rng.random(sub_n) < 0.5
    out = []
    for m in ("visit", "conversion"):
        y = _BOOT[m][idx]
        b = bootstrap_diff(y[~a], y[a], config=BootstrapConfig(n_boot=n_boot, seed=seed))
        _, _, pw = welch_p(
            *[
                np.array([v])
                for v in (
                    (~a).sum(),
                    y[~a].sum(),
                    (y[~a] ** 2).sum(),
                    a.sum(),
                    y[a].sum(),
                    (y[a] ** 2).sum(),
                )
            ]
        )
        out += [b.p_value, float(pw[0])]
    return out[0], out[1], out[2], out[3]


def run_bootstrap_aa(n_reps: int = 1000, sub_n: int = 20_000, n_boot: int = 500, workers: int = 8) -> dict[str, Any]:
    t0 = time.time()
    d = load_criteo(["visit", "conversion"], where="treatment = 0")
    v, c = d["visit"].astype(np.float64), d["conversion"].astype(np.float64)
    with ProcessPoolExecutor(workers, initializer=_init_boot, initargs=(v, c)) as ex:
        rows = list(ex.map(_boot_rep, [(SEED + i, sub_n, n_boot) for i in range(n_reps)], chunksize=10))
    arr = np.array(rows)
    names = ["visit_bootstrap", "visit_welch", "conversion_bootstrap", "conversion_welch"]
    out: dict[str, Any] = {
        "label": f"SIMULATION: A/A on random subsamples of {sub_n:,} real control units",
        "n_reps": n_reps,
        "sub_n": sub_n,
        "n_boot": n_boot,
        "methods": {},
    }
    for j, k in enumerate(names):
        hits = int((arr[:, j] < ALPHA).sum())
        lo, hi = stats.binomtest(hits, n_reps).proportion_ci(0.95)
        out["methods"][k] = {"fpr": hits / n_reps, "fpr_ci95": [lo, hi]}
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------- power curves
POWER_LIFTS = {
    "visit": [0.005, 0.01, 0.015, 0.02, 0.03, 0.04],
    "conversion": [0.02, 0.04, 0.06, 0.08, 0.12, 0.16],
}


def run_power(n_reps: int = 2000, batch: int = 25, seed: int = SEED + 1) -> dict[str, Any]:
    """Inject a known relative lift into a random half of real control units."""
    t0 = time.time()
    u = load_control_units()
    n = u["visit"].size
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {
        "label": "SIMULATION: known effect injected into A/A splits of real control units",
        "n_units": n,
        "n_reps_per_point": (n_reps // batch) * batch,
        "curves": {},
    }
    for m, lifts in POWER_LIFTS.items():
        y = u[m]
        base = float(y.mean())
        cols = {"raw": y, "cuped": u[f"{m}_cuped"], "cupac": u[f"{m}_cupac"]}
        D = np.column_stack([np.ones(n)] + [c for v in cols.values() for c in (v, v**2)])
        G = np.column_stack([np.ones(n)] + [2 * v + 1 for v in cols.values()])  # for flips: Σ(z+f)² = Σz² + Σf(2z+1)
        zero = (y == 0).astype(np.float64)
        total = D.sum(axis=0)
        curve = []
        for lift in lifts:
            q = lift * base / (1 - base)  # P(flip a non-visitor) giving E[relative lift] = lift
            hits = {k: 0 for k in cols}
            se_acc = {k: 0.0 for k in cols}
            for _ in range(n_reps // batch):
                W = (rng.random((batch, n)) < 0.5).astype(np.float64)
                F = W * (rng.random((batch, n)) < q) * zero[None, :]
                T = W @ D
                C = total[None, :] - T
                FG = F @ G  # Σf, Σf(2z+1) per adjusted series
                for j, k in enumerate(cols):
                    s_t = T[:, 1 + 2 * j] + FG[:, 0]
                    ss_t = T[:, 2 + 2 * j] + FG[:, 1 + j]
                    _, se, p = welch_p(C[:, 0], C[:, 1 + 2 * j], C[:, 2 + 2 * j], T[:, 0], s_t, ss_t)
                    hits[k] += int((p < ALPHA).sum())
                    se_acc[k] += float(se.sum())
            reps = (n_reps // batch) * batch
            delta = q * (1 - base)
            row: dict[str, Any] = {"relative_lift": lift, "absolute_delta": delta}
            for k in cols:
                se_mean = se_acc[k] / reps
                row[f"power_{k}"] = hits[k] / reps
                row[f"analytic_{k}"] = power_two_sample(delta, se_mean)
            curve.append(row)
        out["curves"][m] = {"baseline": base, "points": curve}
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------- peeking A/A
def run_peeking_aa(
    n_reps: int = 2000, looks: int = 24, tau_relative: float = 0.10, seed: int = SEED + 2
) -> dict[str, Any]:
    """A/A with continuous monitoring on real control units in a random arrival order.

    Criteo has no timestamps, so each rep draws a random arrival order (a random look index per
    unit) and a random 50/50 split. Methods: naive |z| >= 1.96 at any look, mSPRT, Lan-DeMets OBF.
    """
    t0 = time.time()
    u = load_control_units()
    n = u["visit"].size
    rng = np.random.default_rng(seed)
    frac = np.arange(1, looks + 1) / looks
    obf = gsd_boundaries(frac).z_bounds
    z_naive = stats.norm.ppf(1 - ALPHA / 2)
    series = {"visit": u["visit"], "conversion": u["conversion"], "visit_cupac": u["visit_cupac"]}
    hits = {k: {"naive": 0, "msprt": 0, "obf": 0, "fixed_horizon": 0} for k in series}
    for _ in range(n_reps):
        cell = rng.integers(0, looks, n) * 2 + (rng.random(n) < 0.5)
        cnt = np.bincount(cell, minlength=2 * looks).reshape(looks, 2).cumsum(axis=0)
        for k, y in series.items():
            s1 = np.bincount(cell, weights=y, minlength=2 * looks).reshape(looks, 2).cumsum(axis=0)
            s2 = np.bincount(cell, weights=y * y, minlength=2 * looks).reshape(looks, 2).cumsum(axis=0)
            d, se, p = welch_p(cnt[:, 0], s1[:, 0], s2[:, 0], cnt[:, 1], s1[:, 1], s2[:, 1])
            z = d / se
            tau2 = (tau_relative * float(u["visit" if k != "conversion" else "conversion"].mean())) ** 2
            p_ms = msprt_pvalues(d, se**2, tau2)
            h = hits[k]
            h["naive"] += int((np.abs(z) >= z_naive).any())
            h["msprt"] += int((p_ms < ALPHA).any())
            h["obf"] += int((np.abs(z) >= obf).any())
            h["fixed_horizon"] += int(p[-1] < ALPHA)
    out: dict[str, Any] = {
        "label": "SIMULATION: A/A on real Criteo control units, random arrival order, continuous monitoring",
        "n_units": n,
        "n_reps": n_reps,
        "looks": looks,
        "tau_relative": tau_relative,
        "by_metric": {},
    }
    for k, h in hits.items():
        out["by_metric"][k] = {
            m: {"fpr": v / n_reps, "ci95": list(stats.binomtest(v, n_reps).proportion_ci(0.95))} for m, v in h.items()
        }
    out["fpr"] = out["by_metric"]["visit"]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    write_json("aa_tests", run_aa())
    write_json("aa_tests_85_15", run_aa(share_t=0.85, seed=SEED + 85))
    write_json("aa_bootstrap", run_bootstrap_aa())
    write_json("power_curves", run_power())
    write_json("criteo_peeking", run_peeking_aa())
