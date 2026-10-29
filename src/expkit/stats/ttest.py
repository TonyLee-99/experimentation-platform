"""Welch two-sample t-test from sufficient statistics.

Welch (1947) does not assume equal variances, which matters when allocations are unequal
(Criteo is 85/15). Degrees of freedom use the Welch–Satterthwaite approximation, so the
result is identical to ``scipy.stats.ttest_ind(equal_var=False)``.
"""

from __future__ import annotations

import math

from scipy import stats

from expkit.stats.types import Estimate, MeanStats


def relative_ci(
    mean_c: float, mean_t: float, var_mean_c: float, var_mean_t: float, z: float
) -> tuple[float, float, float]:
    """Relative lift (mean_t / mean_c - 1) and its delta-method CI.

    Var(m_t / m_c) ≈ Var(m_t)/m_c² + m_t² Var(m_c)/m_c⁴ (independent arms).
    """
    if mean_c == 0:
        return float("nan"), float("nan"), float("nan")
    rel = mean_t / mean_c - 1.0
    var = var_mean_t / mean_c**2 + mean_t**2 * var_mean_c / mean_c**4
    half = z * math.sqrt(var)
    return rel, rel - half, rel + half


def welch_df(var_mean_c: float, var_mean_t: float, n_c: float, n_t: float) -> float:
    num = (var_mean_c + var_mean_t) ** 2
    den = var_mean_c**2 / (n_c - 1) + var_mean_t**2 / (n_t - 1)
    return float("inf") if den == 0 else num / den


def welch_from_moments(
    metric: str,
    method: str,
    mean_c: float,
    mean_t: float,
    var_c: float,
    var_t: float,
    n_c: float,
    n_t: float,
    alpha: float = 0.05,
) -> Estimate:
    """Core Welch test given per-arm means and unit-level sample variances."""
    vm_c = var_c / n_c
    vm_t = var_t / n_t
    delta = mean_t - mean_c
    se = math.sqrt(vm_c + vm_t)
    df = welch_df(vm_c, vm_t, n_c, n_t)
    if se == 0:
        t_stat, p = 0.0, 1.0
    else:
        t_stat = delta / se
        p = float(2 * stats.t.sf(abs(t_stat), df))
    q = float(stats.t.ppf(1 - alpha / 2, df))
    z = float(stats.norm.ppf(1 - alpha / 2))
    rel, rlo, rhi = relative_ci(mean_c, mean_t, vm_c, vm_t, z)
    return Estimate(
        metric=metric,
        method=method,
        control=mean_c,
        treatment=mean_t,
        delta=delta,
        se=se,
        ci_low=delta - q * se,
        ci_high=delta + q * se,
        p_value=p,
        rel_lift=rel,
        rel_ci_low=rlo,
        rel_ci_high=rhi,
        n_control=n_c,
        n_treatment=n_t,
        stat=t_stat,
        df=df,
    )


def welch(
    control: MeanStats,
    treatment: MeanStats,
    metric: str = "metric",
    alpha: float = 0.05,
) -> Estimate:
    """Welch t-test on difference in means (treatment − control)."""
    return welch_from_moments(
        metric,
        "welch",
        control.mean,
        treatment.mean,
        control.var,
        treatment.var,
        control.n,
        treatment.n,
        alpha,
    )
