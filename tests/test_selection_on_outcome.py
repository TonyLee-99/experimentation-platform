"""Why covariate adjustment cannot fix outcome-dependent sampling (the Criteo construction risk).

Synthetic RCT (labeled simulation): T is randomized independently of X. Then negatives (Y = 0) are
kept with probability r < 1, as in "non-uniform negative sampling on labels". The test shows
(1) the retained sample is imbalanced on X even though T was randomized,
(2) post-stratifying on X does not recover the true lift,
(3) the Mantel–Haenszel odds ratio does, and so does re-weighting negatives by 1/r.
"""

from __future__ import annotations

import numpy as np

from expkit.analyses.balance import _bins, _post_stratified, odds_ratios


def test_negative_sampling_creates_imbalance_that_adjustment_cannot_fix() -> None:
    rng = np.random.default_rng(42)
    n = 2_000_000
    x = rng.normal(size=n)
    t = (rng.random(n) < 0.85).astype(np.float64)
    logit = -3.5 + 1.5 * x + 0.4 * t  # constant conditional log-OR 0.4
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float64)
    b = _bins(x, 50)
    true_rd, _ = _post_stratified(y, t, b, np.ones(n), 50)

    r = 0.2
    keep = (y == 1) | (rng.random(n) < r)
    xs, ts, ys, bs = x[keep], t[keep], y[keep], b[keep]

    # (1) imbalance appears: treatment share is higher in the high-x (high-outcome) bins
    top = bs >= 45
    assert ts[top].mean() - ts[~top].mean() > 0.005
    # (2) post-stratification on x (as released) is far from the truth
    ps_rd, _ = _post_stratified(ys, ts, bs, np.ones(ys.size), 50)
    base_true = y[t == 0].mean()
    base_sel = ys[ts == 0].mean()
    assert abs(ps_rd / base_sel - true_rd / base_true) > 0.08  # 0.226 vs truth 0.352
    # (3a) the stratified odds ratio is invariant to outcome-dependent sampling
    mh = odds_ratios(ys, ts, bs, 50)["mantel_haenszel_or"]
    assert abs(np.log(mh) - 0.4) < 0.03
    # (3b) re-weighting negatives by 1/r restores the population lift
    w = np.where(ys == 1, 1.0, 1 / r)
    rw_rd, _ = _post_stratified(ys, ts, bs, w, 50)
    base_rw = np.average(ys[ts == 0], weights=w[ts == 0])
    assert abs(rw_rd / base_rw - true_rd / base_true) < 0.03  # within ~2 SE (SE ~0.0125)
    assert xs.size < n
