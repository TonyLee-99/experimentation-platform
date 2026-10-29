"""Poisson bootstrap fallback.

When samples are small or a metric is very skewed, the normal approximation behind Welch and the
delta method can be poor. The bootstrap makes no distributional assumption. We use the *Poisson*
bootstrap (each unit gets a Poisson(1) weight) because it is embarrassingly parallel and is what
large platforms use at scale (Chamandy et al. 2012, "Estimating Uncertainty for Massive Data
Streams", Google; also used in Spotify's and Netflix's engines).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from expkit.stats.types import Estimate


@dataclass(frozen=True)
class BootstrapConfig:
    n_boot: int = 1000
    seed: int = 0
    chunk: int = 200


def _boot_means(y: np.ndarray, x: np.ndarray | None, n_boot: int, rng: np.random.Generator, chunk: int) -> np.ndarray:
    """Bootstrap replicates of Σwy/Σw (mean) or Σwy/Σwx (ratio)."""
    out = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        w = rng.poisson(1.0, size=(b, y.size)).astype(np.float64)
        num = w @ y
        den = w.sum(axis=1) if x is None else w @ x
        out[start : start + b] = num / den
    return out


def bootstrap_diff(
    y_c: ArrayLike,
    y_t: ArrayLike,
    x_c: ArrayLike | None = None,
    x_t: ArrayLike | None = None,
    metric: str = "metric",
    alpha: float = 0.05,
    config: BootstrapConfig | None = None,
) -> Estimate:
    """Percentile-bootstrap CI and p-value for treatment − control (mean or ratio metric).

    The two-sided p-value is 2·min(P*(Δ ≤ 0), P*(Δ ≥ 0)), i.e. the smallest α at which the
    percentile interval would exclude 0.
    """
    cfg = config or BootstrapConfig()
    rng = np.random.default_rng(cfg.seed)
    yc = np.asarray(y_c, dtype=np.float64)
    yt = np.asarray(y_t, dtype=np.float64)
    xc = None if x_c is None else np.asarray(x_c, dtype=np.float64)
    xt = None if x_t is None else np.asarray(x_t, dtype=np.float64)
    bc = _boot_means(yc, xc, cfg.n_boot, rng, cfg.chunk)
    bt = _boot_means(yt, xt, cfg.n_boot, rng, cfg.chunk)
    d = bt - bc
    mc = yc.sum() / (yc.size if xc is None else xc.sum())
    mt = yt.sum() / (yt.size if xt is None else xt.sum())
    delta = mt - mc
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    p = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))
    se = float(d.std(ddof=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_boot = bt / bc - 1
    finite = rel_boot[np.isfinite(rel_boot)]
    rlo, rhi = np.quantile(finite, [alpha / 2, 1 - alpha / 2]) if finite.size else (np.nan, np.nan)
    return Estimate(
        metric=metric,
        method="bootstrap",
        control=float(mc),
        treatment=float(mt),
        delta=float(delta),
        se=se,
        ci_low=float(lo),
        ci_high=float(hi),
        p_value=p,
        rel_lift=float(mt / mc - 1) if mc != 0 else float("nan"),
        rel_ci_low=float(rlo),
        rel_ci_high=float(rhi),
        n_control=float(yc.size),
        n_treatment=float(yt.size),
        extra={"n_boot": float(cfg.n_boot)},
    )


def needs_bootstrap(n_min: float, n_events_min: float, skew: float) -> bool:
    """Engine rule for falling back to the bootstrap.

    Rule of thumb from Kohavi, Deng, Longbotham & Xu (2014, KDD, "Seven Rules of Thumb for Web Site
    Experimenters", rule 7): the CLT is adequate once
    n > 355·skewness² per arm. We also require at least 30 "events" (non-zero units).
    """
    return n_min < 355 * skew**2 or n_events_min < 30
