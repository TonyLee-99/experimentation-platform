"""Vectorized versions of the engine's tests, for running thousands of simulated experiments.

They take per-arm sums for many replications at once (arrays) and must agree with the scalar
engine functions; :func:`check_against_engine` asserts that on the first replication.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from expkit.stats.delta import delta_ratio
from expkit.stats.ttest import welch
from expkit.stats.types import MeanStats, RatioStats


def welch_p(
    n_c: np.ndarray,
    s_c: np.ndarray,
    ss_c: np.ndarray,
    n_t: np.ndarray,
    s_t: np.ndarray,
    ss_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(delta, se, p) of Welch's test from arrays of sums."""
    m_c, m_t = s_c / n_c, s_t / n_t
    v_c = (ss_c - s_c**2 / n_c) / (n_c - 1)
    v_t = (ss_t - s_t**2 / n_t) / (n_t - 1)
    a, b = v_c / n_c, v_t / n_t
    se = np.sqrt(a + b)
    df = (a + b) ** 2 / (a**2 / (n_c - 1) + b**2 / (n_t - 1))
    t = (m_t - m_c) / se
    p = 2 * stats.t.sf(np.abs(t), df)
    return m_t - m_c, se, p


def delta_p(
    n_c: np.ndarray,
    y_c: np.ndarray,
    x_c: np.ndarray,
    y2_c: np.ndarray,
    x2_c: np.ndarray,
    xy_c: np.ndarray,
    n_t: np.ndarray,
    y_t: np.ndarray,
    x_t: np.ndarray,
    y2_t: np.ndarray,
    x2_t: np.ndarray,
    xy_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def var(
        n: np.ndarray,
        sy: np.ndarray,
        sx: np.ndarray,
        sy2: np.ndarray,
        sx2: np.ndarray,
        sxy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mx, my = sx / n, sy / n
        vx = (sx2 - n * mx**2) / (n - 1)
        vy = (sy2 - n * my**2) / (n - 1)
        cxy = (sxy - n * mx * my) / (n - 1)
        r = my / mx
        return r, (vy - 2 * r * cxy + r**2 * vx) / (n * mx**2)

    r_c, v_c = var(n_c, y_c, x_c, y2_c, x2_c, xy_c)
    r_t, v_t = var(n_t, y_t, x_t, y2_t, x2_t, xy_t)
    se = np.sqrt(v_c + v_t)
    z = (r_t - r_c) / se
    return r_t - r_c, se, 2 * stats.norm.sf(np.abs(z))


def check_against_engine(y: np.ndarray, assign: np.ndarray, x: np.ndarray | None = None) -> None:
    """Assert vectorized == scalar engine on one assignment (guards against drift)."""
    c, t = assign == 0, assign == 1
    if x is None:
        ref = welch(MeanStats.from_array(y[c]), MeanStats.from_array(y[t]))
        d, se, p = welch_p(
            np.array([c.sum()]),
            np.array([y[c].sum()]),
            np.array([(y[c] ** 2).sum()]),
            np.array([t.sum()]),
            np.array([y[t].sum()]),
            np.array([(y[t] ** 2).sum()]),
        )
    else:
        ref = delta_ratio(RatioStats.from_arrays(y[c], x[c]), RatioStats.from_arrays(y[t], x[t]))
        rc, rt = RatioStats.from_arrays(y[c], x[c]), RatioStats.from_arrays(y[t], x[t])
        d, se, p = delta_p(
            *[np.array([v]) for v in (rc.n, rc.sum_y, rc.sum_x, rc.sum_y2, rc.sum_x2, rc.sum_xy)],
            *[np.array([v]) for v in (rt.n, rt.sum_y, rt.sum_x, rt.sum_y2, rt.sum_x2, rt.sum_xy)],
        )
    assert np.isclose(d[0], ref.delta, rtol=1e-9, atol=1e-15)
    assert np.isclose(se[0], ref.se, rtol=1e-9)
    assert np.isclose(p[0], ref.p_value, rtol=1e-6, atol=1e-12)
