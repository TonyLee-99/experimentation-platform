"""Bayesian decision view: expected loss of a ship decision.

A p-value answers "how surprising is this data if nothing changed?". A product decision needs
"what do I lose, on average, if I ship and I am wrong?". Under a normal posterior
θ | data ~ N(m, s²) the expected losses are closed-form (e.g. Stucchio 2015, "Bayesian A/B
Testing at VWO"; the same rule is used by Statsig/Eppo Bayesian modes):

    E[loss | ship]      = E[max(−θ, 0)] = s·φ(m/s) − m·Φ(−m/s)
    E[loss | not ship]  = E[max( θ, 0)] = s·φ(m/s) + m·Φ( m/s)

Rule: ship if E[loss | ship] < ε, where ε is the "cost of being wrong" a PM is willing to
accept (e.g. 0.1% relative lift). The prior N(μ₀, τ²) can come from the company's own
experiment portfolio (empirical Bayes, see :mod:`expkit.stats.eb`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True)
class BayesDecision:
    post_mean: float
    post_sd: float
    prob_positive: float
    loss_ship: float
    loss_hold: float
    threshold: float

    @property
    def ship(self) -> bool:
        return self.loss_ship < self.threshold

    @property
    def verdict(self) -> str:
        return "ship" if self.ship else "don't ship"


def normal_posterior(
    estimate: float, se: float, prior_mean: float = 0.0, prior_sd: float = math.inf
) -> tuple[float, float]:
    """Conjugate normal update. ``prior_sd = inf`` gives the flat-prior posterior N(θ̂, se²)."""
    if math.isinf(prior_sd):
        return estimate, se
    w = prior_sd**2 / (prior_sd**2 + se**2)
    mean = prior_mean + w * (estimate - prior_mean)
    sd = math.sqrt(w) * se
    return mean, sd


def expected_losses(post_mean: float, post_sd: float) -> tuple[float, float]:
    """(E[max(−θ,0)], E[max(θ,0)]) for θ ~ N(post_mean, post_sd²)."""
    if post_sd == 0:
        return max(-post_mean, 0.0), max(post_mean, 0.0)
    r = post_mean / post_sd
    pdf = float(stats.norm.pdf(r))
    loss_ship = post_sd * pdf - post_mean * float(stats.norm.cdf(-r))
    loss_hold = post_sd * pdf + post_mean * float(stats.norm.cdf(r))
    return loss_ship, loss_hold


def bayes_decision(
    estimate: float,
    se: float,
    threshold: float,
    prior_mean: float = 0.0,
    prior_sd: float = math.inf,
) -> BayesDecision:
    m, s = normal_posterior(estimate, se, prior_mean, prior_sd)
    ls, lh = expected_losses(m, s)
    p_pos = float(stats.norm.sf(0, loc=m, scale=s)) if s > 0 else float(m > 0)
    return BayesDecision(m, s, p_pos, ls, lh, threshold)
