"""Benjamini–Hochberg false-discovery-rate control (Benjamini & Hochberg 1995).

Scorecards test many metrics at once; at α = 0.05 per metric, a 20-metric scorecard shows a
"significant" metric by chance most of the time. BH controls the expected share of false
discoveries among the metrics flagged.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def bh_adjust(p_values: ArrayLike) -> NDArray[np.float64]:
    """BH-adjusted p-values (a.k.a. q-values in the step-up sense)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    # enforce monotonicity from the largest p downwards
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    return out


def bh_reject(p_values: ArrayLike, alpha: float = 0.05) -> NDArray[np.bool_]:
    return bh_adjust(p_values) <= alpha
