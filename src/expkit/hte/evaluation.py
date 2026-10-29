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
    """Area between the Qini curve and the random-targeting line, over the total increment.

    0 = no better than random; positive = ranking concentrates incrementality at the top.
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
