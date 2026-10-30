"""HTE and targeting on Criteo's real treatment (train on fold 0, evaluate on held-out fold 1).

Also hosts the randomization balance check (classifier two-sample test, Lopez-Paz & Oquab 2017),
because the propensity model it fits is reused by the X-learner and the IPW targeting table.
"""

from __future__ import annotations

import time
from typing import Any

import lightgbm as lgb
import numpy as np
from scipy import stats
from sklearn.metrics import log_loss, roc_auc_score

from expkit.analyses.common import SEED, load_criteo, read_json, write_json
from expkit.data import CRITEO_FEATURES, PROCESSED
from expkit.hte.evaluation import normalized_qini, qini_coefficient, qini_curve, targeting_with_ci
from expkit.hte.learners import BASE_PARAMS, TLearner, XLearner


def auc_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of an AUC."""
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(np.sqrt(var))


def balance_check(
    X: np.ndarray, t: np.ndarray, train: np.ndarray, test: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    booster = lgb.train({**BASE_PARAMS, "objective": "binary"}, lgb.Dataset(X[train], t[train]), 300)
    e = np.asarray(booster.predict(X[test]), dtype=np.float64)
    auc = float(roc_auc_score(t[test], e))
    n1, n0 = int(t[test].sum()), int((1 - t[test]).sum())
    se = auc_se(auc, n1, n0)
    base = float(t[train].mean())
    res = {
        "method": "classifier two-sample test (LightGBM, train fold 0, test fold 1)",
        "auc": auc,
        "auc_se": se,
        "z_vs_0.5": (auc - 0.5) / se,
        "p_value": float(stats.norm.sf((auc - 0.5) / se)),
        "logloss_model": float(log_loss(t[test], e)),
        "logloss_constant": float(log_loss(t[test], np.full(e.size, base))),
        "n_test": int(test.sum()),
        "paper_reference": "Diemert et al. (2021) report a C2ST on 300 samples with p = 0.137",
    }
    return res, e


def _boot(score: np.ndarray, t: np.ndarray, y: np.ndarray, reps: int, rng: np.random.Generator) -> dict[str, Any]:
    order = np.argsort(-score, kind="stable")
    s_t, s_y = t[order], y[order]
    ident = np.arange(score.size, 0, -1).astype(np.float64)  # preserves the order inside qini_curve
    q_norm, share20 = [], []
    for _ in range(reps):
        w = rng.poisson(1.0, score.size).astype(np.float64)
        f, q = qini_curve(ident, s_t, s_y, w)
        q_norm.append(normalized_qini(f, q))
        share20.append(q[np.searchsorted(f, 0.2)] / q[-1])
    return {
        "normalized_qini_ci95": np.quantile(q_norm, [0.025, 0.975]).tolist(),
        "share_top20_ci95": np.quantile(share20, [0.025, 0.975]).tolist(),
    }


SCORES_PARQUET = PROCESSED / "criteo_hte_scores.parquet"
RESPONSE = "response_model"


def run(boot_reps: int = 100) -> dict[str, Any]:
    """Train on fold 0, score the fold-1 holdout, persist the scores, then evaluate them."""
    t0 = time.time()
    d = load_criteo(
        [
            "unit_id",
            *CRITEO_FEATURES,
            "treatment",
            "exposure",
            "visit",
            "conversion",
            "fold",
            "cupac_visit",
            "cupac_conversion",
            "p_exposure",
        ],
        with_cupac=True,
    )
    X = np.column_stack([d[f] for f in CRITEO_FEATURES])
    t = d["treatment"].astype(np.int8)
    train, test = d["fold"] == 0, d["fold"] == 1
    balance, e_test = balance_check(X, t.astype(np.float64), train, test)
    Xh = X[test]
    cols: dict[str, np.ndarray] = {
        "unit_id": d["unit_id"][test],
        "treatment": t[test],
        "exposure": d["exposure"][test],
        "visit": d["visit"][test],
        "conversion": d["conversion"][test],
        "propensity": e_test,
    }
    for y in ("visit", "conversion"):
        yv = d[y].astype(np.float64)
        tl = TLearner().fit(X[train], t[train], yv[train])
        xl = XLearner().fit(X[train], t[train], yv[train], base=tl)
        cols[f"{y}__t_learner"] = tl.predict(Xh)
        cols[f"{y}__x_learner"] = xl.predict(Xh, e_test)
        cols[f"{y}__{RESPONSE}"] = d[f"cupac_{y}"][test].astype(np.float64)
    cols["exposure_propensity"] = d["p_exposure"][test].astype(np.float64)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table(cols), SCORES_PARQUET, compression="zstd")
    out = evaluate_scores(boot_reps)
    out["n_train"] = int(train.sum())
    out["balance"] = balance
    out["seconds_total"] = round(time.time() - t0, 1)
    write_json("hte", out)
    return out


def evaluate_scores(boot_reps: int = 100) -> dict[str, Any]:
    """Qini metrics and targeting tables from the persisted holdout scores (fast; no training)."""
    import pyarrow.parquet as pq

    tb = pq.read_table(SCORES_PARQUET)
    c = {k: tb.column(k).to_numpy() for k in tb.column_names}
    th = c["treatment"].astype(np.float64)
    rng = np.random.default_rng(SEED)
    out: dict[str, Any] = {"n_holdout": int(th.size)}
    for y in ("visit", "conversion"):
        yv = c[y].astype(np.float64)
        ranks = {
            "x_learner": c[f"{y}__x_learner"],
            "t_learner": c[f"{y}__t_learner"],
            RESPONSE: c[f"{y}__{RESPONSE}"],
            "exposure_propensity": c["exposure_propensity"],
            "random": rng.random(th.size),
        }
        res: dict[str, Any] = {}
        for name, s in ranks.items():
            f, q = qini_curve(s, th, yv)
            r = {
                "qini_coefficient": qini_coefficient(s, th, yv),
                "qini_area_above_random_share": normalized_qini(f, q),
            }
            if name in ("t_learner", "x_learner", "random", RESPONSE):
                b = _boot(s, th, yv, boot_reps, rng)
                r["qini_area_above_random_share_ci95"] = b["normalized_qini_ci95"]
            res[name] = r
        out[f"qini_{y}"] = res
    out["qini_note"] = (
        "qini_coefficient = standard Radcliffe (normalized by the perfect curve); "
        "qini_area_above_random_share = area between curve and random line / total increment (non-standard)"
    )
    out["causalml_check"] = _causalml_check(c["visit__x_learner"], c["treatment"], c["visit"], rng)
    e = c["propensity"]
    ipw = np.where(th == 1, 1 / e, 1 / (1 - e))
    expo = c["exposure"].astype(np.float64)
    out["targeting"] = {
        "definition": (
            "IPW-weighted increments; every ranking shares one denominator (IPW increment over the whole holdout). "
            "share_of_exposures = share of exposed treated users (exposure is binary per user). "
            "efficiency = increment share / exposure share; tail = same ratio for the excluded users. "
            "CIs: Poisson bootstrap."
        ),
        "outcome_conversion": {
            name: targeting_with_ci(s, th, c["conversion"], expo, ipw, boot_reps=boot_reps, seed=SEED)
            for name, s in (
                ("rank_by_conversion_x_learner", c["conversion__x_learner"]),
                ("rank_by_visit_x_learner", c["visit__x_learner"]),
                ("rank_by_conversion_response_model", c[f"conversion__{RESPONSE}"]),
                ("rank_by_exposure_propensity", c["exposure_propensity"]),
            )
        },
        "outcome_visit": {
            name: targeting_with_ci(s, th, c["visit"], expo, ipw, boot_reps=boot_reps, seed=SEED)
            for name, s in (
                ("rank_by_visit_x_learner", c["visit__x_learner"]),
                ("rank_by_conversion_x_learner", c["conversion__x_learner"]),
            )
        },
        "outcome_conversion_unweighted": {
            "rank_by_conversion_x_learner": targeting_with_ci(
                c["conversion__x_learner"], th, c["conversion"], expo, np.ones(th.size), boot_reps=boot_reps, seed=SEED
            )
        },
        "total_ipw_incremental_conversions_holdout": float(
            (
                np.average(c["conversion"][th == 1], weights=ipw[th == 1])
                - np.average(c["conversion"][th == 0], weights=ipw[th == 0])
            )
            * th.size
        ),
    }
    return out


def _causalml_check(score: np.ndarray, t: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    import pandas as pd
    from causalml.metrics import get_qini

    idx = rng.choice(score.size, 300_000, replace=False)
    s, tt, yy = score[idx], t[idx].astype(np.float64), y[idx].astype(np.float64)
    # Tree-ensemble scores are piecewise constant (many exact ties) and pandas' default sort is not
    # stable, so tied rows would be ordered differently. Break ties identically for both by replacing
    # the score with its (stable) rank; compare only points where both curves are finite.
    s = np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(np.float64)
    ref = get_qini(
        pd.DataFrame({"model": s, "w": tt, "y": yy}),
        outcome_col="y",
        treatment_col="w",
        normalize=False,
    )
    _, ours = qini_curve(s, tt, yy)
    ref_vals = ref["model"].to_numpy()[1:]  # causalml prepends a 0 row
    ok = np.isfinite(ref_vals) & (np.cumsum((1 - tt)[np.argsort(-s, kind="stable")]) > 0)
    ref_vals, ours = ref_vals[ok], ours[ok]
    return {
        "n": int(idx.size),
        "max_abs_diff": float(np.max(np.abs(ref_vals - ours))),
        "max_rel_diff": float(np.max(np.abs(ref_vals - ours)) / np.max(np.abs(ours))),
    }


def main() -> None:
    if SCORES_PARQUET.exists():
        # reuse persisted holdout scores; training is deterministic given the fixed seeds and folds
        out = evaluate_scores()
        prev = read_json("hte")
        out["n_train"], out["balance"] = prev["n_train"], prev["balance"]
        write_json("hte", out)
    else:
        run()
