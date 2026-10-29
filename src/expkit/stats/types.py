"""Plain data containers shared by every estimator.

Platforms (Eppo, Statsig, Netflix XP) never move unit-level rows into the stats engine.
The warehouse aggregates each variant down to *sufficient statistics* and the engine works
from those. These classes are those sufficient statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MeanStats:
    """Sufficient statistics of a mean metric for one variant: n, Σy, Σy²."""

    n: float
    sum: float
    sum_sq: float

    @property
    def mean(self) -> float:
        return self.sum / self.n

    @property
    def var(self) -> float:
        """Unbiased sample variance (ddof = 1)."""
        return (self.sum_sq - self.sum**2 / self.n) / (self.n - 1)

    @classmethod
    def from_array(cls, y: FloatArray) -> MeanStats:
        y = np.asarray(y, dtype=np.float64)
        return cls(float(y.size), float(y.sum()), float(np.dot(y, y)))

    @classmethod
    def from_summary(cls, n: float, mean: float, var: float) -> MeanStats:
        """Rebuild sums from a published (n, mean, variance) triple, e.g. the ASOS dataset."""
        return cls(n, n * mean, (n - 1) * var + n * mean**2)


@dataclass(frozen=True)
class RatioStats:
    """Sufficient statistics of a ratio metric Σy / Σx for one variant (y, x per unit)."""

    n: float
    sum_y: float
    sum_x: float
    sum_y2: float
    sum_x2: float
    sum_xy: float

    @property
    def ratio(self) -> float:
        return self.sum_y / self.sum_x

    @classmethod
    def from_arrays(cls, y: FloatArray, x: FloatArray) -> RatioStats:
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        return cls(
            float(y.size),
            float(y.sum()),
            float(x.sum()),
            float(y @ y),
            float(x @ x),
            float(x @ y),
        )


@dataclass(frozen=True)
class CovariateStats:
    """Sufficient statistics for regression adjustment (CUPED/CUPAC) for one variant.

    y is the outcome, X the (n, k) covariate matrix. Stored: n, Σy, Σy², Σx (k), Σxy (k),
    Σxxᵀ (k, k). All are additive across rows, so they can be computed by SQL GROUP BY.
    """

    n: float
    sum_y: float
    sum_y2: float
    sum_x: FloatArray
    sum_xy: FloatArray
    sum_xx: FloatArray

    @classmethod
    def from_arrays(cls, y: FloatArray, X: FloatArray) -> CovariateStats:
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        return cls(
            float(y.size),
            float(y.sum()),
            float(y @ y),
            X.sum(axis=0),
            X.T @ y,
            X.T @ X,
        )

    def __add__(self, other: CovariateStats) -> CovariateStats:
        return CovariateStats(
            self.n + other.n,
            self.sum_y + other.sum_y,
            self.sum_y2 + other.sum_y2,
            self.sum_x + other.sum_x,
            self.sum_xy + other.sum_xy,
            self.sum_xx + other.sum_xx,
        )


@dataclass
class Estimate:
    """Result of comparing treatment vs control on one metric with one method."""

    metric: str
    method: str
    control: float
    treatment: float
    delta: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    rel_lift: float
    rel_ci_low: float
    rel_ci_high: float
    n_control: float
    n_treatment: float
    stat: float = float("nan")
    df: float = float("inf")
    extra: dict[str, float] = field(default_factory=dict)

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
