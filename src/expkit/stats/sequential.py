"""Sequential testing: why you cannot just look at the p-value every day.

A fixed-horizon test promises a 5% false-positive rate only if you look *once*. Looking every
day and stopping at the first p < 0.05 inflates the error rate well above 5% (Johari et al.
2017; Kohavi et al. 2020 ch. 17). Two standard fixes are implemented:

1. **mSPRT** — mixture Sequential Probability Ratio Test (Robbins 1970; Johari, Koomen, Pekelis &
   Walsh 2017/2022, "Always Valid Inference", used by Optimizely and Eppo). For an estimate
   θ̂ₙ ~ N(θ, Vₙ) and a N(0, τ²) mixing distribution over θ, the mixture likelihood ratio is

       Λₙ = sqrt(Vₙ / (Vₙ + τ²)) · exp( τ² θ̂ₙ² / (2 Vₙ (Vₙ + τ²)) ).

   By Ville's inequality P(∃n: Λₙ ≥ 1/α | H₀) ≤ α, so you may look after every user.
   The always-valid p-value is pₙ = min(1, min_{m≤n} 1/Λₘ).

2. **Group-sequential with Lan–DeMets O'Brien–Fleming alpha spending** (O'Brien & Fleming 1979;
   Lan & DeMets 1983). α is "spent" as a function of the information fraction t:
   α(t) = 2 − 2Φ(z_{1−α/2} / √t). Boundaries are found by exact recursive numerical integration
   of the Brownian-motion null (Armitage, McPherson & Rowe 1969; Jennison & Turnbull 2000 ch.19),
   which supports any number of irregularly spaced looks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, stats

from expkit.stats.types import FloatArray

# --------------------------------------------------------------------------------------------
# mSPRT
# --------------------------------------------------------------------------------------------


def msprt_lambda(theta_hat: ArrayLike, var: ArrayLike, tau2: float) -> FloatArray:
    """Gaussian-mixture likelihood ratio Λ for each look (vectorized)."""
    th = np.asarray(theta_hat, dtype=np.float64)
    v = np.asarray(var, dtype=np.float64)
    log_lam = 0.5 * np.log(v / (v + tau2)) + tau2 * th**2 / (2 * v * (v + tau2))
    out: FloatArray = np.exp(np.minimum(log_lam, 700.0))
    return out


def msprt_pvalues(theta_hat: ArrayLike, var: ArrayLike, tau2: float) -> FloatArray:
    """Always-valid p-values: running minimum of 1/Λ, capped at 1."""
    lam = msprt_lambda(theta_hat, var, tau2)
    p = np.minimum(1.0, 1.0 / lam)
    out: FloatArray = np.minimum.accumulate(p, axis=-1)
    return out


def msprt_ci(theta_hat: float, var: float, tau2: float, alpha: float = 0.05) -> tuple[float, float]:
    """Always-valid confidence interval at a single look (inverts Λ(θ̂ − θ₀) < 1/α)."""
    # Λ < 1/α  <=>  (θ̂−θ₀)² < 2V(V+τ²)/τ² · (log(1/α) + 0.5·log((V+τ²)/V))
    half = math.sqrt(2 * var * (var + tau2) / tau2 * (math.log(1 / alpha) + 0.5 * math.log((var + tau2) / var)))
    return theta_hat - half, theta_hat + half


def msprt_z_boundary(var: ArrayLike, tau2: float, alpha: float = 0.05) -> FloatArray:
    """The mSPRT rejection region expressed on the z = θ̂/√V scale (for charts)."""
    v = np.asarray(var, dtype=np.float64)
    z2 = 2 * (v + tau2) / tau2 * (math.log(1 / alpha) + 0.5 * np.log((v + tau2) / v))
    out: FloatArray = np.sqrt(z2)
    return out


# --------------------------------------------------------------------------------------------
# Group sequential (Lan–DeMets O'Brien–Fleming spending), two-sided symmetric boundaries
# --------------------------------------------------------------------------------------------


def obf_spending(t: ArrayLike, alpha: float = 0.05) -> FloatArray:
    """Cumulative type-I error spent by information fraction t (Lan–DeMets OBF-type)."""
    tt = np.clip(np.asarray(t, dtype=np.float64), 1e-12, 1.0)
    out: FloatArray = 2.0 - 2.0 * stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) / np.sqrt(tt))
    return out


def pocock_spending(t: ArrayLike, alpha: float = 0.05) -> FloatArray:
    """Lan–DeMets Pocock-type spending α·log(1 + (e − 1)t), provided for comparison."""
    tt = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    out: FloatArray = alpha * np.log(1 + (math.e - 1) * tt)
    return out


@dataclass(frozen=True)
class GSDBoundaries:
    info_fraction: FloatArray
    z_bounds: FloatArray  # reject at look k if |Z_k| >= z_bounds[k]
    alpha_spent: FloatArray  # cumulative


def gsd_boundaries(
    info_fraction: ArrayLike,
    alpha: float = 0.05,
    spending: str = "obf",
    grid_points: int = 801,
) -> GSDBoundaries:
    """Two-sided group-sequential boundaries for arbitrary information fractions.

    Works on the score scale S_k = Z_k·√t_k, which under H₀ is Brownian motion with independent
    N(0, t_k − t_{k−1}) increments. At each look we keep the sub-density of S_k on the
    continuation region (no earlier rejection) on a grid, and solve for the boundary that spends
    exactly α(t_k) − α(t_{k−1}).
    """
    t = np.asarray(info_fraction, dtype=np.float64)
    if np.any(np.diff(t) <= 0) or t[0] <= 0 or t[-1] > 1 + 1e-12:
        raise ValueError("information fractions must be strictly increasing in (0, 1]")
    spend_fn = obf_spending if spending == "obf" else pocock_spending
    cum = spend_fn(t, alpha)
    cum[-1] = alpha if abs(t[-1] - 1) < 1e-12 else cum[-1]
    inc = np.diff(np.concatenate([[0.0], cum]))
    bounds = np.empty(t.size)

    # look 1: S_1 ~ N(0, t_1)
    bounds[0] = stats.norm.isf(inc[0] / 2) if inc[0] > 1e-15 else np.inf
    b_score = min(bounds[0], 10.0) * math.sqrt(t[0])
    grid = np.linspace(-b_score, b_score, grid_points)
    dens = stats.norm.pdf(grid, scale=math.sqrt(t[0]))
    weights = _trapezoid_weights(grid)

    for k in range(1, t.size):
        sd = math.sqrt(t[k] - t[k - 1])
        sqrt_tk = math.sqrt(t[k])
        mass = dens * weights

        def cross_prob(c: float, mass: FloatArray = mass, sd: float = sd, sqrt_tk: float = sqrt_tk) -> float:
            b = c * sqrt_tk
            return float(
                np.sum(mass * (stats.norm.cdf((-b - grid) / sd) + stats.norm.sf((b - grid) / sd)))
            )

        if inc[k] <= 1e-15:
            bounds[k] = np.inf
        else:
            bounds[k] = optimize.brentq(lambda c: cross_prob(c) - inc[k], 0.0, 40.0, xtol=1e-10)
        # propagate the continuation sub-density to look k
        b_score = min(bounds[k], 10.0) * sqrt_tk
        new_grid = np.linspace(-b_score, b_score, grid_points)
        kern = stats.norm.pdf((new_grid[:, None] - grid[None, :]) / sd) / sd
        dens = kern @ mass
        grid = new_grid
        weights = _trapezoid_weights(grid)

    return GSDBoundaries(t, bounds, cum)


def _trapezoid_weights(grid: FloatArray) -> FloatArray:
    h = grid[1] - grid[0]
    w = np.full(grid.size, h)
    w[0] = w[-1] = h / 2
    return w


def first_crossing(z_path: ArrayLike, bounds: ArrayLike) -> int:
    """Index of the first look where |z| >= bound, or -1 if never."""
    hit = np.abs(np.asarray(z_path)) >= np.asarray(bounds)
    idx = np.flatnonzero(hit)
    return int(idx[0]) if idx.size else -1


def crossing_probabilities(
    info_fraction: ArrayLike, z_bounds: ArrayLike, grid_points: int = 801
) -> FloatArray:
    """Cumulative two-sided H₀ crossing probability of given z-boundaries at each look.

    Same recursive integration as :func:`gsd_boundaries`, run forward for fixed boundaries.
    Used to verify classical tables (e.g. Pocock / O'Brien–Fleming constants).
    """
    t = np.asarray(info_fraction, dtype=np.float64)
    c = np.asarray(z_bounds, dtype=np.float64)
    out = np.empty(t.size)
    out[0] = 2 * stats.norm.sf(c[0])
    b = min(c[0], 10.0) * math.sqrt(t[0])
    grid = np.linspace(-b, b, grid_points)
    dens = stats.norm.pdf(grid, scale=math.sqrt(t[0]))
    for k in range(1, t.size):
        sd = math.sqrt(t[k] - t[k - 1])
        mass = dens * _trapezoid_weights(grid)
        bk = min(c[k], 10.0) * math.sqrt(t[k])
        new_grid = np.linspace(-bk, bk, grid_points)
        kern = stats.norm.pdf((new_grid[:, None] - grid[None, :]) / sd) / sd
        dens = kern @ mass
        grid = new_grid
        cont = float(np.sum(dens * _trapezoid_weights(grid)))
        out[k] = 1.0 - cont
    return out
