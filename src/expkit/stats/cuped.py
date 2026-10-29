"""CUPED / CUPAC regression adjustment from sufficient statistics.

CUPED (Deng, Xu, Kohavi & Walker 2013) replaces y with y − θᵀ(x − x̄) where x is a vector
of pre-treatment covariates and θ is the pooled (both arms) OLS slope. Because x is
independent of assignment, the adjusted difference in means is still unbiased, and its
variance shrinks by the share of variance in y that x explains (R²).

CUPAC (DoorDash, 2020) is CUPED with a single covariate: an ML prediction of y built only from
pre-treatment features, fitted on data the analysis does not use (here: 2-fold cross-fitting).

Everything below works from :class:`CovariateStats`, which are plain sums that SQL can produce,
so the adjustment can run inside the warehouse.
"""

from __future__ import annotations

import numpy as np

from expkit.stats.ttest import welch_from_moments
from expkit.stats.types import CovariateStats, Estimate, FloatArray


def _cov_parts(s: CovariateStats) -> tuple[float, FloatArray, FloatArray]:
    """Sample (ddof = 1) var(y), cov(x, y) and cov(x, x) of one CovariateStats block."""
    n = s.n
    mx = s.sum_x / n
    my = s.sum_y / n
    var_y = (s.sum_y2 - n * my**2) / (n - 1)
    cov_xy = (s.sum_xy - n * mx * my) / (n - 1)
    cov_xx = (s.sum_xx - n * np.outer(mx, mx)) / (n - 1)
    return float(var_y), cov_xy, cov_xx


def pooled_theta(control: CovariateStats, treatment: CovariateStats) -> FloatArray:
    """θ = Cov(x,x)⁻¹ Cov(x,y) on the pooled sample (Deng et al. 2013, §3.2)."""
    _, cov_xy, cov_xx = _cov_parts(control + treatment)
    theta: FloatArray = np.linalg.solve(cov_xx, cov_xy)
    return theta


def variance_reduction(control: CovariateStats, treatment: CovariateStats) -> float:
    """Share of the variance of the difference-in-means removed by the adjustment."""
    raw = cuped(control, treatment, theta=np.zeros_like(control.sum_x))
    adj = cuped(control, treatment)
    return 1.0 - (adj.se**2) / (raw.se**2)


def cuped(
    control: CovariateStats,
    treatment: CovariateStats,
    metric: str = "metric",
    alpha: float = 0.05,
    theta: FloatArray | None = None,
    method: str = "cuped",
) -> Estimate:
    """Regression-adjusted difference in means with a Welch test on the adjusted values.

    Adjusted arm mean: ȳ_a − θᵀ(x̄_a − x̄_pooled). Adjusted unit variance within arm a:
    var(y) − 2θᵀcov(x,y) + θᵀcov(x,x)θ. Passing ``theta=0`` returns the plain Welch test.
    """
    if theta is None:
        theta = pooled_theta(control, treatment)
    pooled = control + treatment
    mx_pool = pooled.sum_x / pooled.n

    def adjusted(s: CovariateStats) -> tuple[float, float]:
        var_y, cov_xy, cov_xx = _cov_parts(s)
        mean = s.sum_y / s.n - float(theta @ (s.sum_x / s.n - mx_pool))
        var = var_y - 2 * float(theta @ cov_xy) + float(theta @ cov_xx @ theta)
        return mean, max(var, 0.0)

    mc, vc = adjusted(control)
    mt, vt = adjusted(treatment)
    est = welch_from_moments(metric, method, mc, mt, vc, vt, control.n, treatment.n, alpha)
    # Relative lift is reported against the *unadjusted* control mean, which is the number a PM
    # recognizes; the adjustment only changes the numerator's noise.
    raw_mc = control.sum_y / control.n
    if raw_mc != 0:
        z = (est.ci_high - est.delta) / est.se if est.se > 0 else 0.0
        est.rel_lift = est.delta / raw_mc
        est.rel_ci_low = (est.delta - z * est.se) / raw_mc
        est.rel_ci_high = (est.delta + z * est.se) / raw_mc
        est.control = raw_mc
        est.treatment = raw_mc + est.delta
    est.extra["theta_norm"] = float(np.linalg.norm(theta))
    return est
