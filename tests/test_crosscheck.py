"""Every estimator against an external referee (scipy / statsmodels / linearmodels / gsdesign).

These tests use small *synthetic* inputs on purpose: the point is algebraic agreement with a
reference implementation. The same checks are re-run on real Criteo rows by
``expkit validate crosscheck`` (results/crosscheck.json).
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from expkit.stats.cuped import cuped, pooled_theta
from expkit.stats.delta import delta_ratio
from expkit.stats.iv import tsls_with_controls, wald
from expkit.stats.multiple import bh_adjust, bh_reject
from expkit.stats.sequential import (
    crossing_probabilities,
    gsd_boundaries,
    msprt_lambda,
    obf_spending,
)
from expkit.stats.srm import srm_check
from expkit.stats.ttest import welch
from expkit.stats.types import CovariateStats, MeanStats, RatioStats

RNG = np.random.default_rng(20261029)


def test_welch_matches_scipy() -> None:
    yc = RNG.exponential(1.0, 4000)
    yt = RNG.exponential(1.1, 9000)
    ours = welch(MeanStats.from_array(yc), MeanStats.from_array(yt))
    ref = stats.ttest_ind(yt, yc, equal_var=False)
    assert ours.stat == pytest.approx(ref.statistic, rel=1e-10)
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    assert ours.df == pytest.approx(ref.df, rel=1e-10)
    ci = ref.confidence_interval(0.95)
    assert ours.ci_low == pytest.approx(ci.low, rel=1e-8)
    assert ours.ci_high == pytest.approx(ci.high, rel=1e-8)


def test_welch_from_summary_roundtrip() -> None:
    y = RNG.normal(3, 2, 1000)
    s = MeanStats.from_summary(y.size, y.mean(), y.var(ddof=1))
    assert s.mean == pytest.approx(y.mean())
    assert s.var == pytest.approx(y.var(ddof=1))


def _clustered(n_users: int, rate: float, conv: float) -> tuple[np.ndarray, np.ndarray]:
    visits = 1 + RNG.poisson(rate, n_users)  # every user >= 1 visit
    p_user = RNG.beta(2, 2 / conv - 2, n_users)
    conversions = RNG.binomial(visits, p_user)
    return conversions.astype(float), visits.astype(float)


def test_delta_method_matches_cluster_robust_ols() -> None:
    yc, xc = _clustered(3000, 2.0, 0.1)
    yt, xt = _clustered(3500, 2.2, 0.12)
    ours = delta_ratio(RatioStats.from_arrays(yc, xc), RatioStats.from_arrays(yt, xt))

    # Reference: expand to visit level, regress visit outcome on intercept with user clusters,
    # separately per arm (statsmodels default correction G/(G-1)), then combine.
    def arm_var(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
        users = np.repeat(np.arange(y.size), x.astype(int))
        outcome = np.zeros(users.size)
        start = np.concatenate([[0], np.cumsum(x.astype(int))[:-1]])
        for i in range(y.size):  # put y_i conversions on the first y_i visits of user i
            outcome[start[i] : start[i] + int(y[i])] = 1.0
        fit = sm.OLS(outcome, np.ones(users.size)).fit(cov_type="cluster", cov_kwds={"groups": users})
        return float(fit.params[0]), float(fit.bse[0] ** 2)

    rc, vc = arm_var(yc, xc)
    rt, vt = arm_var(yt, xt)
    assert ours.control == pytest.approx(rc, rel=1e-12)
    assert ours.delta == pytest.approx(rt - rc, rel=1e-10)
    assert ours.se == pytest.approx(np.sqrt(vc + vt), rel=1e-6)


def test_delta_reduces_to_welch_when_denominator_is_one() -> None:
    yc = RNG.binomial(1, 0.1, 5000).astype(float)
    yt = RNG.binomial(1, 0.12, 5000).astype(float)
    d = delta_ratio(RatioStats.from_arrays(yc, np.ones_like(yc)), RatioStats.from_arrays(yt, np.ones_like(yt)))
    w = welch(MeanStats.from_array(yc), MeanStats.from_array(yt))
    assert d.se == pytest.approx(w.se, rel=1e-12)
    assert d.delta == pytest.approx(w.delta, rel=1e-12)


def _cov_data(n: int, effect: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = RNG.normal(size=(n, 3))
    t = RNG.binomial(1, 0.5, n)
    y = X @ np.array([1.0, -0.5, 0.25]) + effect * t + RNG.normal(size=n)
    return y, X, t


def test_cuped_equals_welch_on_adjusted_values_with_statsmodels_theta() -> None:
    y, X, t = _cov_data(20000, 0.05)
    c = CovariateStats.from_arrays(y[t == 0], X[t == 0])
    tr = CovariateStats.from_arrays(y[t == 1], X[t == 1])
    ours = cuped(c, tr)
    theta_sm = sm.OLS(y, sm.add_constant(X)).fit().params[1:]
    np.testing.assert_allclose(pooled_theta(c, tr), theta_sm, rtol=1e-8)
    y_adj = y - (X - X.mean(axis=0)) @ theta_sm
    ref = stats.ttest_ind(y_adj[t == 1], y_adj[t == 0], equal_var=False)
    assert ours.stat == pytest.approx(ref.statistic, rel=1e-6)
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-6, abs=1e-300)
    assert ours.delta == pytest.approx(y_adj[t == 1].mean() - y_adj[t == 0].mean(), rel=1e-8)


def test_cuped_close_to_ancova() -> None:
    y, X, t = _cov_data(50000, 0.05)
    ours = cuped(
        CovariateStats.from_arrays(y[t == 0], X[t == 0]),
        CovariateStats.from_arrays(y[t == 1], X[t == 1]),
    )
    fit = sm.OLS(y, sm.add_constant(np.column_stack([t, X]))).fit(cov_type="HC1")
    assert ours.delta == pytest.approx(fit.params[1], rel=0.01)
    assert ours.se == pytest.approx(fit.bse[1], rel=0.01)


def test_cuped_reduces_variance() -> None:
    y, X, t = _cov_data(20000, 0.0)
    c = CovariateStats.from_arrays(y[t == 0], X[t == 0])
    tr = CovariateStats.from_arrays(y[t == 1], X[t == 1])
    raw = cuped(c, tr, theta=np.zeros(3))
    adj = cuped(c, tr)
    # R² of y on X is 1.3125 / 2.3125 ≈ 0.568
    assert 1 - adj.se**2 / raw.se**2 == pytest.approx(0.568, abs=0.03)


def test_srm_matches_scipy() -> None:
    counts = [10_123, 89_877]
    r = srm_check(counts, [0.1, 0.9])
    ref = stats.chisquare(counts, f_exp=[10_000, 90_000])
    assert r.chi2 == pytest.approx(ref.statistic, rel=1e-12)
    assert r.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert not r.blocked
    assert srm_check([10_500, 89_500], [0.1, 0.9]).blocked


def test_bh_matches_statsmodels() -> None:
    p = RNG.uniform(0, 0.2, 40) ** 2
    rej, adj, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    np.testing.assert_allclose(bh_adjust(p), adj, atol=1e-12)
    np.testing.assert_array_equal(bh_reject(p), rej)


def test_obf_boundaries_vs_independent_gsdesign() -> None:
    gsdesign = pytest.importorskip("gsdesign")
    t = np.arange(1, 6) / 5
    b = gsd_boundaries(t).z_bounds
    g = gsdesign.h1(32, 0.0, t[0], -b[0], b[0])
    cum = [1 - g[2].sum()]
    for k in range(1, 5):
        g = gsdesign.hupdate(32, 0.0, t[k], -b[k], b[k], 0.0, t[k - 1], g)
        cum.append(1 - g[2].sum())
    np.testing.assert_allclose(cum, obf_spending(t), atol=1e-6)
    assert b[0] == pytest.approx(stats.norm.ppf(0.975) / np.sqrt(0.2), rel=1e-6)


def test_classical_constants_jennison_turnbull() -> None:
    t = np.arange(1, 6) / 5
    k = np.arange(1, 6)
    assert crossing_probabilities(t, 2.040 * np.sqrt(5 / k))[-1] == pytest.approx(0.05, abs=5e-4)
    assert crossing_probabilities(t, np.full(5, 2.413))[-1] == pytest.approx(0.05, abs=5e-4)


def test_obf_monte_carlo_type1() -> None:
    t = np.arange(1, 6) / 5
    b = gsd_boundaries(t).z_bounds
    rng = np.random.default_rng(7)
    S = (rng.standard_normal((200_000, 5)) * np.sqrt(0.2)).cumsum(axis=1)
    fpr = float((np.abs(S / np.sqrt(t)) >= b).any(axis=1).mean())
    assert 0.047 <= fpr <= 0.053


def test_msprt_lambda_matches_numerical_integration() -> None:
    from scipy import integrate

    for th, v, tau2 in [(0.01, 1e-4, 4e-4), (-0.3, 0.05, 0.01), (0.0, 1.0, 2.0)]:

        def integrand(theta: float, th: float = th, v: float = v, tau2: float = tau2) -> float:
            lr = stats.norm.pdf(th, loc=theta, scale=np.sqrt(v)) / stats.norm.pdf(th, 0, np.sqrt(v))
            return float(lr * stats.norm.pdf(theta, 0, np.sqrt(tau2)))

        ref, _ = integrate.quad(integrand, -np.inf, np.inf, epsabs=1e-13, epsrel=1e-12)
        assert float(msprt_lambda(th, v, tau2)) == pytest.approx(ref, rel=1e-6)


def test_wald_matches_linearmodels_iv2sls() -> None:
    lm = pytest.importorskip("linearmodels")
    n = 20000
    z = RNG.binomial(1, 0.85, n)
    complier = RNG.binomial(1, 0.2, n)
    d = z * complier
    y = RNG.binomial(1, 0.05 + 0.1 * d + 0.02 * complier).astype(float)
    ours = wald(RatioStats.from_arrays(y[z == 0], d[z == 0]), RatioStats.from_arrays(y[z == 1], d[z == 1]))
    import pandas as pd

    df = pd.DataFrame({"y": y, "d": d.astype(float), "z": z.astype(float), "const": 1.0})
    ref = lm.iv.IV2SLS(df["y"], df[["const"]], df[["d"]], df[["z"]]).fit(cov_type="robust", debiased=False)
    assert ours.cace == pytest.approx(ref.params["d"], rel=1e-10)
    assert ours.cace_se == pytest.approx(ref.std_errors["d"], rel=1e-6)

    # with covariates: streaming implementation vs linearmodels
    W = RNG.normal(size=(n, 2))
    y2 = y + W @ np.array([0.1, -0.05])
    b, se = tsls_with_controls(y2, d.astype(float), z.astype(float), W, chunk=7000)
    df2 = df.assign(y2=y2, w0=W[:, 0], w1=W[:, 1])
    ref2 = lm.iv.IV2SLS(df2["y2"], df2[["const", "w0", "w1"]], df2[["d"]], df2[["z"]]).fit(
        cov_type="robust", debiased=False
    )
    assert b == pytest.approx(ref2.params["d"], rel=1e-8)
    assert se == pytest.approx(ref2.std_errors["d"], rel=1e-6)


def test_obf_type1_with_near_duplicate_looks() -> None:
    """Regression test: looks with almost identical information (Δt ~ 1e-9) used to break the point-sampled
    transition kernel (false-positive rates of 20-95% on some real ASOS schedules). Cell-integrated transitions
    keep the Monte-Carlo type-I error at alpha."""
    t = np.array([0.1067, 0.2928, 0.2949, 0.2949 + 1e-9, 0.5, 0.5 + 2e-9, 0.75, 1.0])
    b = gsd_boundaries(t).z_bounds
    rng = np.random.default_rng(11)
    S = (rng.standard_normal((200_000, t.size)) * np.sqrt(np.diff(np.concatenate([[0.0], t])))).cumsum(axis=1)
    fpr = float((np.abs(S / np.sqrt(t)) >= b).any(axis=1).mean())
    assert 0.047 <= fpr <= 0.053
