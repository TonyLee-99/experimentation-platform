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

from expkit.analyses.common import SEED, load_criteo, write_json
from expkit.data import CRITEO_FEATURES
from expkit.hte.evaluation import normalized_qini, qini_curve, targeting_table
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


def run(boot_reps: int = 100) -> dict[str, Any]:
    t0 = time.time()
    d = load_criteo(
        [
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
    out: dict[str, Any] = {"n_train": int(train.sum()), "n_holdout": int(test.sum())}
    out["balance"], e_test = balance_check(X, t.astype(np.float64), train, test)
    Xh, th = X[test], t[test]
    rng = np.random.default_rng(SEED)
    scores: dict[str, dict[str, np.ndarray]] = {}
    for y in ("visit", "conversion"):
        yv = d[y].astype(np.float64)
        tl = TLearner().fit(X[train], t[train], yv[train])
        xl = XLearner().fit(X[train], t[train], yv[train], base=tl)
        scores[y] = {
            "t_learner": tl.predict(Xh),
            "x_learner": xl.predict(Xh, e_test),
            "response_model (P(visit|x) / P(conv|x))": d[f"cupac_{y}"][test].astype(np.float64),
            "exposure_propensity": d["p_exposure"][test].astype(np.float64),
            "random": rng.random(int(test.sum())),
        }
        res: dict[str, Any] = {}
        for name, s in scores[y].items():
            f, q = qini_curve(s, th, yv[test])
            r = {
                "normalized_qini": normalized_qini(f, q),
                "share_of_incremental_top20": float(q[np.searchsorted(f, 0.2)] / q[-1]),
            }
            if name in ("t_learner", "x_learner", "random"):
                r.update(_boot(s, th, yv[test], boot_reps, rng))
            res[name] = r
        out[f"qini_{y}"] = res
    # H3: our Qini vs causalml on a 300k random holdout subsample
    out["causalml_check"] = _causalml_check(scores["visit"]["x_learner"], th, d["visit"][test], rng)
    # H4: targeting tables (raw diff-in-means and IPW with the estimated propensity)
    ipw = np.where(th == 1, 1 / e_test, 1 / (1 - e_test))
    outcomes = {"visits": d["visit"][test], "conversions": d["conversion"][test]}
    out["targeting"] = {}
    for y in ("visit", "conversion"):
        for label, w in (("raw", None), ("ipw", ipw)):
            out["targeting"][f"rank_by_{y}_x_learner__{label}"] = targeting_table(
                scores[y]["x_learner"],
                th,
                outcomes,
                d["exposure"][test].astype(np.float64),
                weights=w,
            )
    out["targeting"]["rank_by_exposure_propensity__raw"] = targeting_table(
        scores["visit"]["exposure_propensity"], th, outcomes, d["exposure"][test].astype(np.float64)
    )
    out["seconds"] = round(time.time() - t0, 1)
    write_json("hte", out)
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


def recheck_causalml() -> dict[str, Any]:
    """Re-run only the H3 check (holdout fold, score = cross-fitted response model) and patch hte.json."""
    from expkit.analyses.common import read_json

    d = load_criteo(["treatment", "visit", "fold", "cupac_visit"], with_cupac=True)
    test = d["fold"] == 1
    res: dict[str, Any] = _causalml_check(
        d["cupac_visit"][test].astype(np.float64), d["treatment"][test], d["visit"][test], np.random.default_rng(SEED)
    )
    res["score"] = "cupac_visit (cross-fitted response model) on holdout fold 1"
    out = read_json("hte")
    out["causalml_check"] = res
    write_json("hte", out)
    return res


def main() -> None:
    run()
