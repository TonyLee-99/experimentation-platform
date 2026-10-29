"""Criteo randomization forensics: the imbalance is real, but its cause decides what is identified.

Facts (all computed here):
* Global SRM passes (split 0.85000), yet pre-treatment features predict assignment (C2ST AUC on
  7M held-out users, `hte.json`) and treatment share rises in the top prognostic decile.

What the dataset paper documents (Diemert et al. 2021, arXiv:2111.10106, §3):
1. Several advertiser tests with *different* treatment ratios were merged, then each test was
   sub-sampled to a common 85% ratio. This removes the test-id confounder of v1.
2. "Non-uniform negative sampling on labels has been performed so that the original incrementality
   level cannot be deduced." That is selection on the outcome.
3. Features were captured before assignment ("we can easily verify … that they were all captured in
   the legit period").

Two competing explanations for the imbalance, and what each predicts:
(a) residue of per-test re-balancing → assignment depends on X only → adjusting for X is the fix,
    and the imbalance should also show among non-visitors;
(b) outcome-dependent negative sampling → P(kept | T, X) = r + (1 − r)·P(Y=1 | T, X) → treated
    users are over-represented where visits are common; adjusting for X does NOT undo it, risk
    differences / relative lifts are not identified without r, and only odds ratios are invariant
    (the case–control result).

This module measures the patterns each predicts and reports the visit / conversion lift as a
sensitivity interval over an assumed uniform negative keep-rate r = 1/k (reweighting non-visitors by
k), next to the sampling-invariant Mantel–Haenszel odds ratio.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from expkit.analyses.common import load_criteo, write_json
from expkit.data import CRITEO_FEATURES

K_GRID = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
N_BINS = 100


def _bins(score: np.ndarray, n_bins: int) -> np.ndarray:
    q = np.quantile(score, np.linspace(0, 1, n_bins + 1))
    return np.clip(np.searchsorted(q, score, side="right") - 1, 0, n_bins - 1)


def _post_stratified(y: np.ndarray, t: np.ndarray, b: np.ndarray, w: np.ndarray, n_bins: int) -> tuple[float, float]:
    """Weighted post-stratified difference in means and its SE (unweighted-count approximation)."""

    def wsum(mask: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.bincount(b[mask], weights=v[mask], minlength=n_bins)

    t1, t0 = t == 1, t == 0
    w1, w0 = wsum(t1, w), wsum(t0, w)
    m1, m0 = wsum(t1, w * y) / w1, wsum(t0, w * y) / w0
    share = np.bincount(b, weights=w, minlength=n_bins) / w.sum()
    n1 = np.bincount(b[t1], minlength=n_bins)
    n0 = np.bincount(b[t0], minlength=n_bins)
    var = np.sum(share**2 * (m1 * (1 - m1) / n1 + m0 * (1 - m0) / n0))
    return float(np.sum(share * (m1 - m0))), float(math.sqrt(var))


def odds_ratios(y: np.ndarray, t: np.ndarray, b: np.ndarray, n_bins: int) -> dict[str, float]:
    """Crude and Mantel–Haenszel (stratified) odds ratios; invariant to outcome-dependent sampling."""

    def cnt(mask: np.ndarray) -> np.ndarray:
        return np.bincount(b[mask], minlength=n_bins).astype(np.float64)

    a, bb = cnt((t == 1) & (y == 1)), cnt((t == 1) & (y == 0))
    c, d = cnt((t == 0) & (y == 1)), cnt((t == 0) & (y == 0))
    nn = a + bb + c + d
    crude = a.sum() * d.sum() / (bb.sum() * c.sum())
    mh = np.sum(a * d / nn) / np.sum(bb * c / nn)
    return {"crude_or": float(crude), "mantel_haenszel_or": float(mh), "bins": n_bins}


def run() -> dict[str, Any]:
    d = load_criteo(
        [*CRITEO_FEATURES, "treatment", "visit", "conversion", "cupac_visit", "cupac_conversion"], with_cupac=True
    )
    t = d["treatment"].astype(np.float64)
    v = d["visit"].astype(np.float64)
    conv = d["conversion"].astype(np.float64)
    X = np.column_stack([d[f] for f in CRITEO_FEATURES]).astype(np.float64)
    b_vis = _bins(d["cupac_visit"].astype(np.float64), N_BINS)
    b_conv = _bins(d["cupac_conversion"].astype(np.float64), N_BINS)
    dec = _bins(d["cupac_visit"].astype(np.float64), 10)

    deciles = []
    for k in range(10):
        m = dec == k
        deciles.append(
            {
                "decile": k + 1,
                "n": int(m.sum()),
                "visit_rate": float(v[m].mean()),
                "treatment_share_all": float(t[m].mean()),
                "treatment_share_nonvisitors": float(t[m & (v == 0)].mean()),
                "treatment_share_visitors": float(t[m & (v == 1)].mean()),
            }
        )
    sd = X.std(axis=0)
    sens = []
    for kk in K_GRID:
        w = np.where(v == 1, 1.0, kk)
        wb = np.bincount(b_vis, weights=w, minlength=N_BINS)
        share = np.bincount(b_vis, weights=w * t, minlength=N_BINS) / wb
        pbar = float((w * t).sum() / w.sum())
        nb = np.bincount(b_vis, minlength=N_BINS)
        chi2 = float(np.sum((share - pbar) ** 2 / (pbar * (1 - pbar) / nb)))
        mt = np.average(X[t == 1], axis=0, weights=w[t == 1])
        mc = np.average(X[t == 0], axis=0, weights=w[t == 0])
        vt, vc = np.average(v[t == 1], weights=w[t == 1]), np.average(v[t == 0], weights=w[t == 0])
        ct, cc = np.average(conv[t == 1], weights=w[t == 1]), np.average(conv[t == 0], weights=w[t == 0])
        ps_v, ps_v_se = _post_stratified(v, t, b_vis, w, N_BINS)
        ps_c, ps_c_se = _post_stratified(conv, t, b_conv, w, N_BINS)
        sens.append(
            {
                "k_upweight_nonvisitors": kk,
                "implied_negative_keep_rate": 1 / kk,
                "chi2_treatment_share_100_bins": chi2,
                "max_abs_smd": float(np.max(np.abs((mt - mc) / sd))),
                "visit_raw_relative": float(vt / vc - 1),
                "visit_poststrat_relative": ps_v / float(vc),
                "visit_poststrat_relative_se": ps_v_se / float(vc),
                "conversion_raw_relative": float(ct / cc - 1),
                "conversion_poststrat_relative": ps_c / float(cc),
                "conversion_poststrat_relative_se": ps_c_se / float(cc),
            }
        )

    def rng_of(key: str) -> list[float]:
        vals = [s[key] for s in sens]
        return [min(vals), max(vals)]

    out: dict[str, Any] = {
        "deciles_by_prognostic_score": deciles,
        "negative_sampling_sensitivity": sens,
        "sensitivity_interval": {
            "visit_poststrat_relative": rng_of("visit_poststrat_relative"),
            "conversion_poststrat_relative": rng_of("conversion_poststrat_relative"),
            "visit_raw_relative": rng_of("visit_raw_relative"),
            "conversion_raw_relative": rng_of("conversion_raw_relative"),
            "k_range": [K_GRID[0], K_GRID[-1]],
        },
        "odds_ratios": {
            "visit": odds_ratios(v, t, b_vis, N_BINS),
            "conversion": odds_ratios(conv, t, b_conv, N_BINS),
        },
        "control_means": {"visit": float(v[t == 0].mean()), "conversion": float(conv[t == 0].mean())},
        "explanations": {
            "a_rebalancing_residue": "assignment depends on X only; X-adjustment fixes it; "
            "imbalance also among non-visitors",
            "b_outcome_dependent_negative_sampling": "selection on Y; X-adjustment does not fix it; only the OR is "
            "invariant; pure Y-selection predicts treatment share <= 0.85 among non-visitors",
            "observed_nonvisitor_share_top_decile": deciles[-1]["treatment_share_nonvisitors"],
            "observed_nonvisitor_share_deciles_1_8_mean": float(
                np.mean([x["treatment_share_nonvisitors"] for x in deciles[:8]])
            ),
        },
        "source_note": "Diemert et al. 2021 (arXiv:2111.10106) §3 and Table 3: their permutation C2ST "
        "(300 permutation draws; classifier log-loss 0.42307 vs constant 0.42269, p = 0.137) did not reject.",
    }
    write_json("balance", out)
    _figure(out)
    return out


def _figure(out: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from expkit.analyses.common import FIGURES

    dec = out["deciles_by_prognostic_score"]
    sens = out["negative_sampling_sensitivity"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = [r["decile"] for r in dec]
    ax = axes[0]
    ax.plot(x, [r["treatment_share_all"] * 100 for r in dec], "o-", color="#2a78d6", lw=2, label="all users")
    ax.plot(
        x,
        [r["treatment_share_nonvisitors"] * 100 for r in dec],
        "s--",
        color="#eb6834",
        lw=2,
        label="non-visitors only",
    )
    ax.axhline(85, color="#52514e", lw=1, ls=":", label="design 85%")
    ax.set_xticks(x)
    ax.set_xlabel("decile of predicted P(visit | pre-treatment features)")
    ax.set_ylabel("treatment share (%)")
    ax.set_title("SRM passes overall, but treatment share depends on user type", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    k = [s["k_upweight_nonvisitors"] for s in sens]
    ax.plot(k, [s["visit_raw_relative"] * 100 for s in sens], "o-", color="#52514e", lw=2, label="raw difference")
    ax.plot(
        k,
        [s["visit_poststrat_relative"] * 100 for s in sens],
        "o-",
        color="#2a78d6",
        lw=2,
        label="post-stratified on 100 prognostic bins",
    )
    ax.set_xscale("log")
    ax.set_xticks(k, [f"{val:g}" for val in k])
    ax.set_xlabel("non-visitors re-weighted ×k (assumed sampling correction)")
    ax.set_ylabel("relative visit lift (%)")
    ax.set_title("The adjusted lift depends on the unknown sampling rate", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7)
    for a in axes:
        a.grid(color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "criteo_imbalance.png", dpi=140)
    plt.close(fig)


def main() -> None:
    run()
