"""Meta-learners for heterogeneous treatment effects (Künzel, Sekhon, Bickel & Yu 2019, PNAS).

* T-learner: fit μ₁(x) on treated, μ₀(x) on control; τ̂(x) = μ₁(x) − μ₀(x).
* X-learner: impute individual effects D₁ = Y − μ₀(X) on treated and D₀ = μ₁(X) − Y on control,
  regress each on X to get τ₁(x), τ₀(x), and combine τ̂(x) = e(x)·τ₀(x) + (1 − e(x))·τ₁(x),
  where e(x) is the propensity of treatment. The X-learner is designed for very unbalanced
  arms — exactly Criteo's 85/15 split — because it lets the large arm's model do most of the work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np

BASE_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 2000,
    "lambda_l2": 10.0,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "num_threads": 10,
    "seed": 7,
}


def _fit(X: np.ndarray, y: np.ndarray, objective: str, rounds: int) -> lgb.Booster:
    return lgb.train({**BASE_PARAMS, "objective": objective}, lgb.Dataset(X, y), num_boost_round=rounds)


@dataclass
class TLearner:
    rounds: int = 400
    models: dict[int, lgb.Booster] = field(default_factory=dict)

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> TLearner:
        for arm in (0, 1):
            m = t == arm
            self.models[arm] = _fit(X[m], y[m], "binary", self.rounds)
        return self

    def mu(self, X: np.ndarray, arm: int) -> np.ndarray:
        return np.asarray(self.models[arm].predict(X), dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.mu(X, 1) - self.mu(X, 0), dtype=np.float64)


@dataclass
class XLearner:
    rounds: int = 400
    base: TLearner = field(default_factory=TLearner)
    tau: dict[int, lgb.Booster] = field(default_factory=dict)

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray, base: TLearner | None = None) -> XLearner:
        self.base = base or TLearner(self.rounds).fit(X, t, y)
        tr, ct = t == 1, t == 0
        d1 = y[tr] - self.base.mu(X[tr], 0)
        d0 = self.base.mu(X[ct], 1) - y[ct]
        self.tau[1] = _fit(X[tr], d1, "regression", self.rounds)
        self.tau[0] = _fit(X[ct], d0, "regression", self.rounds)
        return self

    def predict(self, X: np.ndarray, propensity: np.ndarray | float) -> np.ndarray:
        e = np.asarray(propensity)
        tau0 = np.asarray(self.tau[0].predict(X), dtype=np.float64)
        tau1 = np.asarray(self.tau[1].predict(X), dtype=np.float64)
        return np.asarray(e * tau0 + (1 - e) * tau1, dtype=np.float64)
