"""Non-compliance: intention-to-treat (ITT) vs complier average causal effect (CACE / LATE).

In ad-incrementality tests (Criteo, Google "ghost ads", Meta Conversion Lift) users are
*assigned* to be eligible for ads (Z = treatment), but only some are actually *exposed*
(D = exposure). Control users can never be exposed: one-sided non-compliance.

* ITT_Y = E[Y | Z=1] − E[Y | Z=0] — the effect of the *policy* of being eligible.
* With the exclusion restriction (assignment moves Y only through exposure) and one-sided
  non-compliance, the Wald / IV estimator CACE = ITT_Y / ITT_D is the average effect of
  exposure on the users who would be exposed if assigned (Angrist, Imbens & Rubin 1996).

The Wald estimator is the just-identified 2SLS estimator with instrument Z. Its HC0 robust
variance equals the delta-method variance below with population (ddof = 0) moments; this is
checked against ``linearmodels.IV2SLS`` in tests/test_crosscheck.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from expkit.stats.types import FloatArray, RatioStats


@dataclass(frozen=True)
class IVResult:
    itt_y: float
    itt_y_se: float
    itt_d: float
    itt_d_se: float
    cace: float
    cace_se: float
    ci_low: float
    ci_high: float
    p_value: float
    control_mean: float


def _moments(s: RatioStats, ddof: int) -> tuple[float, float, float, float, float]:
    """mean y, mean d, var y, var d, cov(y, d) with RatioStats(y=outcome, x=exposure)."""
    n = s.n
    my, md = s.sum_y / n, s.sum_x / n
    vy = (s.sum_y2 - n * my**2) / (n - ddof)
    vd = (s.sum_x2 - n * md**2) / (n - ddof)
    cyd = (s.sum_xy - n * my * md) / (n - ddof)
    return my, md, vy, vd, cyd


def wald(control: RatioStats, treatment: RatioStats, alpha: float = 0.05, ddof: int = 0) -> IVResult:
    """Wald/IV estimate. ``RatioStats`` fields: y = outcome, x = exposure (D)."""
    my_c, md_c, vy_c, vd_c, c_c = _moments(control, ddof)
    my_t, md_t, vy_t, vd_t, c_t = _moments(treatment, ddof)
    n_c, n_t = control.n, treatment.n
    dy = my_t - my_c
    dd = md_t - md_c
    var_dy = vy_t / n_t + vy_c / n_c
    var_dd = vd_t / n_t + vd_c / n_c
    cov = c_t / n_t + c_c / n_c
    beta = dy / dd
    var_beta = (var_dy - 2 * beta * cov + beta**2 * var_dd) / dd**2
    se = math.sqrt(var_beta)
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = float(2 * stats.norm.sf(abs(beta / se)))
    return IVResult(dy, math.sqrt(var_dy), dd, math.sqrt(var_dd), beta, se,
                    beta - z * se, beta + z * se, p, my_c)


def tsls_with_controls(
    y: FloatArray, d: FloatArray, z: FloatArray, W: FloatArray, chunk: int = 1_000_000
) -> tuple[float, float]:
    """Just-identified 2SLS of y on [1, W, d] with instruments [1, W, z]; HC0 robust SE of d.

    Computed in two streaming passes over row chunks so it scales to 14M rows without building
    the full design matrix. Returns (coefficient on d, robust SE).
    """
    n = y.size
    k = W.shape[1] + 2

    def design(sl: slice) -> tuple[FloatArray, FloatArray]:
        ones = np.ones((sl.stop - sl.start, 1))
        w = W[sl].astype(np.float64)
        X = np.hstack([ones, w, d[sl, None].astype(np.float64)])
        Z = np.hstack([ones, w, z[sl, None].astype(np.float64)])
        return X, Z

    ZX = np.zeros((k, k))
    Zy = np.zeros(k)
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        X, Z = design(sl)
        ZX += Z.T @ X
        Zy += Z.T @ y[sl].astype(np.float64)
    beta = np.linalg.solve(ZX, Zy)
    meat = np.zeros((k, k))
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        X, Z = design(sl)
        u = y[sl].astype(np.float64) - X @ beta
        Zu = Z * u[:, None]
        meat += Zu.T @ Zu
    inv = np.linalg.inv(ZX)
    V = inv @ meat @ inv.T
    return float(beta[-1]), float(math.sqrt(V[-1, -1]))


def complier_control_mean(control_mean: float, never_taker_mean: float, compliance: float) -> float:
    """E[Y(0) | complier] under one-sided non-compliance (Abadie 2003 / Imbens & Rubin ch. 24).

    E[Y | Z=0] = π·E[Y(0)|C] + (1−π)·E[Y(0)|NT], and E[Y | Z=1, D=0] = E[Y(0)|NT].
    """
    return (control_mean - (1 - compliance) * never_taker_mean) / compliance


def required_n_ratio(var_all: float, var_compliers: float, compliance: float) -> float:
    """How many times more *randomized* users an ITT test needs than a perfectly triggered test.

    ITT: effect π·CACE, unit variance σ²_all, n_ITT ∝ σ²_all / (π·CACE)².
    Triggered (analyze only would-be-exposed users in both arms, e.g. ghost ads): effect CACE on
    π·n analyzed users with variance σ²_C, so randomized n_trig ∝ σ²_C / (π · CACE²).
    Ratio n_ITT / n_trig = σ²_all / (π · σ²_C). (Deng & Hu 2015, "Diluted Treatment Effect
    Estimation for Trigger Analysis in Online Controlled Experiments", WSDM.)
    """
    return var_all / (compliance * var_compliers)


