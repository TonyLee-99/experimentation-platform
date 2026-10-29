"""Cross-fitted CUPAC covariates and the predicted exposure ("ghost trigger") for Criteo.

Criteo has no pre-period metric, so classic CUPED (pre-period value of the same metric) is not
available. CUPAC (DoorDash 2020) instead uses an ML prediction of the outcome from
pre-treatment features. To keep the covariate independent of the unit's own assignment and
outcome, we use 2-fold cross-fitting: a model trained on fold A predicts fold B and vice versa.
The model never sees the treatment indicator.

The same procedure predicts P(exposure | features) using treated units only (exposure is only
observable when treated). Applied to *both* arms, this is a pre-treatment "would-be-exposed"
score — the counterfactual trigger that ghost-ads logging would record for control users
(Johnson, Lewis & Nubbemeyer 2017, "Ghost Ads", JMR).
"""

from __future__ import annotations

import time

import duckdb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from expkit.analyses.common import CUPAC_PARQUET, SEED, load_criteo, write_json
from expkit.data import CRITEO_FEATURES

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 2000,
    "lambda_l2": 10.0,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "num_threads": 10,
    "seed": SEED,
}
N_ROUNDS = 600
# History: v1 used lr=0.1, 63 leaves, 300 rounds -> conversion CUPAC VR 7.6%. One re-tune to a more
# regularized model (lr=0.03, 31 leaves, min_data 2000, L2=10, 600 rounds) gave 14.2% on held-out
# control units. Documented in docs/ACCEPTANCE.md (V4).


def assign_folds(n: int, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.int8)


def _crossfit(X: np.ndarray, y: np.ndarray, fold: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, list[float]]:
    pred = np.empty(y.size, dtype=np.float32)
    aucs = []
    for k in (0, 1):
        tr = (fold != k) & train_mask
        te = fold == k
        booster = lgb.train(LGB_PARAMS, lgb.Dataset(X[tr], y[tr]), num_boost_round=N_ROUNDS)
        pred[te] = np.asarray(booster.predict(X[te]), dtype=np.float32)
        ev = te & train_mask
        aucs.append(float(roc_auc_score(y[ev], pred[ev])))
    return pred, aucs


def fit_cupac(force: bool = False) -> dict[str, object]:
    if CUPAC_PARQUET.exists() and not force:
        return {"skipped": True}
    t0 = time.time()
    d = load_criteo(["unit_id", *CRITEO_FEATURES, "treatment", "exposure", "visit", "conversion"])
    X = np.column_stack([d[f] for f in CRITEO_FEATURES])
    n = X.shape[0]
    fold = assign_folds(n)
    everyone = np.ones(n, dtype=bool)
    report: dict[str, object] = {"n_units": n, "n_rounds": N_ROUNDS, "params": LGB_PARAMS}
    preds = {}
    for target in ("visit", "conversion"):
        preds[target], aucs = _crossfit(X, d[target].astype(np.float32), fold, everyone)
        report[f"auc_{target}"] = aucs
    treated = d["treatment"] == 1
    preds["exposure"], aucs = _crossfit(X, d["exposure"].astype(np.float32), fold, treated)
    report["auc_exposure_treated"] = aucs
    tbl = {
        "unit_id": d["unit_id"],
        "fold": fold,
        "cupac_visit": preds["visit"],
        "cupac_conversion": preds["conversion"],
        "p_exposure": preds["exposure"],
    }
    con = duckdb.connect()
    import pyarrow as pa

    con.register("t", pa.table(tbl))
    CUPAC_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY t TO '{CUPAC_PARQUET}' (FORMAT parquet, COMPRESSION zstd)")
    report["seconds"] = round(time.time() - t0, 1)
    write_json("cupac_models", report)
    return report
