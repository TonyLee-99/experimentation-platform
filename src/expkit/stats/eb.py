"""Empirical Bayes across an experiment portfolio: winner's curse and true win rate.

Each experiment i gives an estimate θ̂ᵢ ~ N(θᵢ, sᵢ²). If we pick the "winners" (largest or
significant θ̂ᵢ), their θ̂ᵢ are biased upward — the winner's curse (Lee & Shen 2018, "Winner's
Curse: Bias Estimation for Total Effects of Features in Online Controlled Experiments", Airbnb;
Deng et al. 2021 "On the Prior Distribution of Effect Sizes"; Azevedo, Deng, Montiel Olea,
Rao & Weyl 2020 "A/B Testing with Fat Tails", JPE).

Treating the true effects θᵢ as draws from a prior g, we estimate g from all experiments
(deconvolution) and use the posterior E[θᵢ | θ̂ᵢ] as a shrunk, de-biased estimate.

* :func:`fit_normal_prior` — g = N(μ, τ²), MLE with heterogeneous sᵢ.
* :func:`fit_npmle` — g is a discrete distribution on a fine grid, fitted by EM
  (Kiefer–Wolfowitz NPMLE; Koenker & Mizera 2014; Efron 2016 "g-modeling"). It can capture
  fat tails and point masses near zero, which the normal prior cannot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, stats

from expkit.stats.types import FloatArray


@dataclass(frozen=True)
class NormalPrior:
    mean: float
    sd: float
    loglik: float

    def posterior(self, est: ArrayLike, se: ArrayLike) -> tuple[FloatArray, FloatArray]:
        e = np.asarray(est, dtype=np.float64)
        s = np.asarray(se, dtype=np.float64)
        w = self.sd**2 / (self.sd**2 + s**2)
        mean = self.mean + w * (e - self.mean)
        sd = np.sqrt(w) * s
        return mean, sd

    def prob_positive(self, est: ArrayLike, se: ArrayLike) -> FloatArray:
        m, s = self.posterior(est, se)
        out: FloatArray = stats.norm.sf(0, loc=m, scale=s)
        return out


@dataclass(frozen=True)
class DiscretePrior:
    support: FloatArray
    weights: FloatArray
    loglik: float

    def _post_weights(self, est: ArrayLike, se: ArrayLike) -> FloatArray:
        e = np.asarray(est, dtype=np.float64)[:, None]
        s = np.asarray(se, dtype=np.float64)[:, None]
        like = stats.norm.pdf((e - self.support[None, :]) / s) / s
        post = like * self.weights[None, :]
        out: FloatArray = post / post.sum(axis=1, keepdims=True)
        return out

    def posterior(self, est: ArrayLike, se: ArrayLike) -> tuple[FloatArray, FloatArray]:
        p = self._post_weights(est, se)
        mean = p @ self.support
        var = p @ self.support**2 - mean**2
        return mean, np.sqrt(np.maximum(var, 0.0))

    def prob_positive(self, est: ArrayLike, se: ArrayLike) -> FloatArray:
        p = self._post_weights(est, se)
        out: FloatArray = p[:, self.support > 0].sum(axis=1)
        return out

    def posterior_interval(
        self, est: ArrayLike, se: ArrayLike, level: float = 0.95
    ) -> tuple[FloatArray, FloatArray]:
        p = self._post_weights(est, se)
        cdf = np.cumsum(p, axis=1)
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
        lo = self.support[np.argmax(cdf >= lo_q, axis=1)]
        hi = self.support[np.argmax(cdf >= hi_q, axis=1)]
        return lo, hi

    @property
    def mean(self) -> float:
        return float(self.weights @ self.support)

    @property
    def sd(self) -> float:
        return float(math.sqrt(self.weights @ (self.support - self.mean) ** 2))

    @property
    def excess_kurtosis(self) -> float:
        c = self.support - self.mean
        m2 = float(self.weights @ c**2)
        m4 = float(self.weights @ c**4)
        return m4 / m2**2 - 3.0 if m2 > 0 else float("nan")

    @property
    def prob_positive_prior(self) -> float:
        return float(self.weights[self.support > 0].sum())


def fit_normal_prior(est: ArrayLike, se: ArrayLike) -> NormalPrior:
    """Maximize Σ log N(θ̂ᵢ; μ, τ² + sᵢ²) over (μ, log τ)."""
    e = np.asarray(est, dtype=np.float64)
    s = np.asarray(se, dtype=np.float64)

    def nll(p: FloatArray) -> float:
        mu, log_tau = p
        v = np.exp(2 * log_tau) + s**2
        return float(0.5 * np.sum(np.log(2 * np.pi * v) + (e - mu) ** 2 / v))

    tau0 = max(float(np.std(e)), 1e-8)
    res = optimize.minimize(nll, x0=np.array([float(np.mean(e)), math.log(tau0)]), method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    mu, log_tau = res.x
    return NormalPrior(float(mu), float(math.exp(log_tau)), float(-res.fun))


def fit_npmle(
    est: ArrayLike,
    se: ArrayLike,
    grid_size: int = 400,
    max_iter: int = 5000,
    tol: float = 1e-10,
) -> DiscretePrior:
    """Kiefer–Wolfowitz NPMLE of the prior on a fixed grid via EM."""
    e = np.asarray(est, dtype=np.float64)
    s = np.asarray(se, dtype=np.float64)
    lo = float(e.min() - 1.0 * s.max())
    hi = float(e.max() + 1.0 * s.max())
    support = np.linspace(lo, hi, grid_size)
    if not np.any(support == 0):
        support = np.sort(np.append(support, 0.0))
    like = stats.norm.pdf((e[:, None] - support[None, :]) / s[:, None]) / s[:, None]
    w = np.full(support.size, 1.0 / support.size)
    prev = -np.inf
    for _ in range(max_iter):
        mix = like @ w
        ll = float(np.sum(np.log(mix)))
        w = w * (like / mix[:, None]).mean(axis=0)
        if ll - prev < tol:
            break
        prev = ll
    ll = float(np.sum(np.log(like @ w)))
    return DiscretePrior(support, w, ll)
