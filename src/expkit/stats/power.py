"""Analytic power, minimum detectable effect (MDE) and sample size for a two-sample z-test."""

from __future__ import annotations

import math

from scipy import stats


def power_two_sample(delta: float, se: float, alpha: float = 0.05) -> float:
    """P(reject | true effect = delta) for a two-sided z-test with standard error ``se``."""
    if se <= 0:
        return 1.0
    z = stats.norm.ppf(1 - alpha / 2)
    shift = abs(delta) / se
    return float(stats.norm.cdf(shift - z) + stats.norm.cdf(-shift - z))


def mde(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest |effect| detectable with the given power."""
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)


def se_two_sample(var_c: float, var_t: float, n_c: float, n_t: float) -> float:
    return math.sqrt(var_c / n_c + var_t / n_t)


def total_sample_size(
    delta: float,
    var: float,
    share_t: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Total units needed (both arms) to detect ``delta`` with unit variance ``var``.

    n = (z_{1−α/2} + z_power)² · var · (1/share_t + 1/(1 − share_t)) / delta².
    """
    k = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) ** 2
    return float(k * var * (1 / share_t + 1 / (1 - share_t)) / delta**2)
