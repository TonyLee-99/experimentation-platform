"""Delta method for ratio metrics (Deng, Knoblich & Lu 2018, "Applying the Delta Method in
Metric Analytics").

A ratio metric such as *conversions per visit* is Σy / Σx where both y and x are summed over
randomization units (users). Visits from the same user are correlated, so treating each visit
as independent understates the variance. Linearizing R = ȳ/x̄ around the means gives

    Var(R̂) ≈ (σ²_y − 2R σ_xy + R² σ²_x) / (n · x̄²)

with unit-level sample (co)variances. This equals the cluster-robust (by user) variance of a
visit-level regression with the usual G/(G−1) correction — see tests/test_crosscheck.py.
"""

from __future__ import annotations

import math

from scipy import stats

from expkit.stats.ttest import relative_ci
from expkit.stats.types import Estimate, RatioStats


def ratio_variance(s: RatioStats) -> float:
    """Delta-method variance of Σy/Σx for one arm (unit-level, ddof = 1)."""
    n = s.n
    mx = s.sum_x / n
    my = s.sum_y / n
    var_x = (s.sum_x2 - n * mx**2) / (n - 1)
    var_y = (s.sum_y2 - n * my**2) / (n - 1)
    cov_xy = (s.sum_xy - n * mx * my) / (n - 1)
    r = my / mx
    return (var_y - 2 * r * cov_xy + r**2 * var_x) / (n * mx**2)


def delta_ratio(
    control: RatioStats,
    treatment: RatioStats,
    metric: str = "ratio",
    alpha: float = 0.05,
) -> Estimate:
    """z-test for the difference of two ratio metrics using the delta method."""
    rc, rt = control.ratio, treatment.ratio
    vc, vt = ratio_variance(control), ratio_variance(treatment)
    delta = rt - rc
    se = math.sqrt(vc + vt)
    z = delta / se if se > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    q = float(stats.norm.ppf(1 - alpha / 2))
    rel, rlo, rhi = relative_ci(rc, rt, vc, vt, q)
    return Estimate(
        metric=metric,
        method="delta",
        control=rc,
        treatment=rt,
        delta=delta,
        se=se,
        ci_low=delta - q * se,
        ci_high=delta + q * se,
        p_value=p,
        rel_lift=rel,
        rel_ci_low=rlo,
        rel_ci_high=rhi,
        n_control=control.n,
        n_treatment=treatment.n,
        stat=z,
    )
