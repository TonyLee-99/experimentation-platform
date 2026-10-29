"""Uplift evaluation without ground truth: Qini curves and targeting tables.

Individual effects are never observed, so we evaluate a *ranking*: sort holdout users by
predicted uplift and, for each top-k%, estimate the incremental outcome among them from the
randomized split (Radcliffe 2007, "Using control groups to target on predicted lift").

    Qini(k) = Y_T(k) − Y_C(k) · N_T(k) / N_C(k)

where Y_T(k), N_T(k) are the treated outcomes/counts among the top k. Qini(k) estimates the
incremental outcomes you would get by treating the top-k users, scaled to the treated count.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def qini_curve(
    score: np.ndarray, t: np.ndarray, y: np.ndarray, w: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(fraction targeted, Qini) at every unit, sorted by descending score (optional unit weights)."""
    order = np.argsort(-score, kind="stable")
    tt, yy = t[order].astype(np.float64), y[order].astype(np.float64)
    ww = np.ones_like(tt) if w is None else w[order].astype(np.float64)
    n_t = np.cumsum(ww * tt)
    n_c = np.cumsum(ww * (1 - tt))
    y_t = np.cumsum(ww * tt * yy)
    y_c = np.cumsum(ww * (1 - tt) * yy)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = y_t - np.where(n_c > 0, y_c * n_t / n_c, 0.0)
    frac = np.cumsum(ww) / ww.sum()
    return frac, q


def normalized_qini(frac: np.ndarray, q: np.ndarray) -> float:
    """NON-STANDARD summary: area between the Qini curve and the random line, divided by the total
    increment Qini(1). 0 = random; its maximum is about 0.5 when there are no negative effects. Reported
    as "Qini area above random (share of total increment)". For the standard Radcliffe coefficient,
    which is normalized by the perfect curve, see :func:`qini_coefficient`.
    """
    area = float(np.trapezoid(q, frac))
    return float(area / q[-1] - 0.5)


def targeting_table(
    score: np.ndarray,
    t: np.ndarray,
    outcomes: dict[str, np.ndarray],
    exposure: np.ndarray,
    tops: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 1.0),
    weights: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Incremental outcomes kept and ad spend used when only the top-k% are targeted.

    Increment in segment = (ȳ_T − ȳ_C) · N_segment (difference in means within the segment,
    optionally inverse-propensity weighted). Spend proxy = share of all ad exposures (impressions)
    that fall in the segment, measured on treated users.
    """
    order = np.argsort(-score, kind="stable")
    n = score.size
    w = np.ones(n) if weights is None else weights
    total: dict[str, float] = {}
    rows = []
    for top in tops:
        seg = order[: round(top * n)]
        ts, ws = t[seg] == 1, w[seg]
        row: dict[str, Any] = {
            "top": top,
            "share_of_exposures": float(exposure[seg][ts].sum() / exposure[t == 1].sum()),
        }
        for name, y in outcomes.items():
            ys = y[seg].astype(np.float64)
            m1 = np.average(ys[ts], weights=ws[ts])
            m0 = np.average(ys[~ts], weights=ws[~ts])
            inc = (m1 - m0) * seg.size
            row[f"incremental_{name}"] = float(inc)
            if top == 1.0:
                total[name] = float(inc)
        rows.append(row)
    for row in rows:
        for name in outcomes:
            row[f"share_of_incremental_{name}"] = row[f"incremental_{name}"] / total[name]
    return rows


def qini_coefficient(score: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
    """Standard Qini coefficient (Radcliffe 2007): (A_model − A_random) / (A_perfect − A_random).

    The perfect ranking puts treated responders first and control responders last (key y·t − y·(1−t)),
    as in scikit-uplift's ``perfect_qini_curve``. Areas are taken over the targeted-fraction axis.
    """
    f, q = qini_curve(score, t, y)
    perfect_key = y * t - y * (1 - t)
    fp, qp = qini_curve(perfect_key.astype(np.float64), t, y)
    a_model = float(np.trapezoid(q, f))
    a_perfect = float(np.trapezoid(qp, fp))
    a_random = 0.5 * float(q[-1])
    return (a_model - a_random) / (a_perfect - a_random)


def targeting_with_ci(
    score: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    exposure: np.ndarray,
    weights: np.ndarray,
    tops: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
    boot_reps: int = 100,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Targeting table with one shared denominator and Poisson-bootstrap CIs.

    For each top-k% by ``score``: the share of the *full-population* propensity-weighted increment
    that is kept, the share of exposures (exposed treated users) used, the efficiency ratio
    (increment share / exposure share) and the marginal efficiency of the excluded tail
    ((1 − increment share) / (1 − exposure share)). Every ranking uses the same denominator, the
    IPW increment over the whole holdout.
    """
    order = np.argsort(-score, kind="stable")
    tt = t[order].astype(np.float64)
    yy = y[order].astype(np.float64)
    ee = exposure[order].astype(np.float64)
    ww = weights[order].astype(np.float64)
    n = tt.size
    cuts = [round(k * n) for k in tops]
    rng = np.random.default_rng(seed)

    def shares(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = ww * b
        c1, c0 = np.cumsum(w * tt), np.cumsum(w * (1 - tt))
        s1, s0 = np.cumsum(w * tt * yy), np.cumsum(w * (1 - tt) * yy)
        cnt = np.cumsum(b)
        exp_c = np.cumsum(b * tt * ee)
        idx = np.array([c - 1 for c in cuts] + [n - 1])
        inc = (s1[idx] / c1[idx] - s0[idx] / c0[idx]) * cnt[idx]
        return inc[:-1] / inc[-1], exp_c[idx[:-1]] / exp_c[-1]

    inc_share, exp_share = shares(np.ones(n))
    boot = [shares(rng.poisson(1.0, n).astype(np.float64)) for _ in range(boot_reps)]
    bi = np.array([b[0] for b in boot])
    be = np.array([b[1] for b in boot])
    beff = bi / be
    rows = []
    for j, k in enumerate(tops):
        rows.append(
            {
                "top": k,
                "share_of_incremental": float(inc_share[j]),
                "share_of_incremental_ci90": np.quantile(bi[:, j], [0.05, 0.95]).tolist(),
                "share_of_exposures": float(exp_share[j]),
                "efficiency": float(inc_share[j] / exp_share[j]),
                "efficiency_ci90": np.quantile(beff[:, j], [0.05, 0.95]).tolist(),
                "tail_marginal_efficiency": float((1 - inc_share[j]) / (1 - exp_share[j])),
            }
        )
    return rows
