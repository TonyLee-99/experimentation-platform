"""Property-based tests (hypothesis) and closed-form checks for each estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from expkit.stats.bayes import bayes_decision, expected_losses, normal_posterior
from expkit.stats.bootstrap import BootstrapConfig, bootstrap_diff, needs_bootstrap
from expkit.stats.cuped import cuped
from expkit.stats.delta import delta_ratio
from expkit.stats.eb import fit_normal_prior, fit_npmle
from expkit.stats.iv import complier_control_mean, required_n_ratio
from expkit.stats.multiple import bh_adjust
from expkit.stats.power import mde, power_two_sample, total_sample_size
from expkit.stats.sequential import msprt_ci, msprt_lambda, msprt_pvalues, msprt_z_boundary
from expkit.stats.ttest import welch
from expkit.stats.types import CovariateStats, MeanStats, RatioStats

finite = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)
sample = arrays(np.float64, st.integers(5, 60), elements=finite).filter(lambda a: a.std() > 1e-3)


@settings(max_examples=60, deadline=None)
@given(sample, sample)
def test_welch_swap_flips_sign(a: np.ndarray, b: np.ndarray) -> None:
    ab = welch(MeanStats.from_array(a), MeanStats.from_array(b))
    ba = welch(MeanStats.from_array(b), MeanStats.from_array(a))
    assert ab.delta == pytest.approx(-ba.delta, abs=1e-9)
    assert ab.p_value == pytest.approx(ba.p_value, abs=1e-9)
    assert 0.0 <= ab.p_value <= 1.0


@settings(max_examples=60, deadline=None)
@given(sample, sample, st.floats(-100, 100), st.floats(0.1, 50))
def test_welch_location_scale(a: np.ndarray, b: np.ndarray, shift: float, scale: float) -> None:
    base = welch(MeanStats.from_array(a), MeanStats.from_array(b))
    moved = welch(MeanStats.from_array(a * scale + shift), MeanStats.from_array(b * scale + shift))
    assert moved.delta == pytest.approx(base.delta * scale, rel=1e-6, abs=1e-6)
    assert moved.stat == pytest.approx(base.stat, rel=1e-5, abs=1e-5)


@settings(max_examples=40, deadline=None)
@given(sample, sample)
def test_cuped_theta_zero_equals_welch(a: np.ndarray, b: np.ndarray) -> None:
    xa = np.linspace(0, 1, a.size)
    xb = np.linspace(0, 1, b.size)
    c = cuped(CovariateStats.from_arrays(a, xa), CovariateStats.from_arrays(b, xb), theta=np.zeros(1))
    w = welch(MeanStats.from_array(a), MeanStats.from_array(b))
    assert c.delta == pytest.approx(w.delta, rel=1e-9, abs=1e-9)
    assert c.se == pytest.approx(w.se, rel=1e-6)


@settings(max_examples=60, deadline=None)
@given(arrays(np.float64, st.integers(1, 50), elements=st.floats(0, 1)))
def test_bh_properties(p: np.ndarray) -> None:
    adj = bh_adjust(p)
    assert np.all(adj >= p - 1e-15)
    assert np.all(adj <= 1.0)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-15)  # monotone in p


def test_welch_closed_form_known_case() -> None:
    # Two arms with n=4: [1,2,3,4] vs [3,4,5,6]: delta=2, var=5/3 each, se = sqrt(5/6)
    w = welch(
        MeanStats.from_array(np.array([1.0, 2, 3, 4])),
        MeanStats.from_array(np.array([3.0, 4, 5, 6])),
    )
    assert w.delta == 2.0
    assert w.se == pytest.approx(math.sqrt(5 / 6))
    assert w.df == pytest.approx(6.0)


def test_ratio_closed_form() -> None:
    # y = 2x exactly -> ratio 2 with zero variance
    x = np.array([1.0, 2, 3, 4])
    r = delta_ratio(RatioStats.from_arrays(2 * x, x), RatioStats.from_arrays(3 * x, x))
    assert r.control == 2.0 and r.treatment == 3.0
    assert r.se == pytest.approx(0.0, abs=1e-12)


def test_power_and_mde_consistency() -> None:
    se = 0.01
    d = mde(se, power=0.8)
    assert power_two_sample(d, se) == pytest.approx(0.8, abs=1e-3)
    assert power_two_sample(0.0, se) == pytest.approx(0.05, abs=1e-12)
    # classic: 2 arms, p=0.5 variance 0.25, delta=0.1 -> about 785 total
    assert total_sample_size(0.1, 0.25) == pytest.approx(784.9, abs=1)


@settings(max_examples=50, deadline=None)
@given(st.floats(-1, 1), st.floats(1e-4, 1), st.floats(1e-4, 1))
def test_msprt_properties(th: float, v: float, tau2: float) -> None:
    lam = float(msprt_lambda(th, v, tau2))
    assert lam > 0
    assert lam >= float(msprt_lambda(0.0, v, tau2)) - 1e-12  # minimized at theta_hat = 0
    z_b = float(msprt_z_boundary(v, tau2))
    assert float(msprt_lambda(z_b * math.sqrt(v), v, tau2)) == pytest.approx(20.0, rel=1e-6)
    lo, hi = msprt_ci(th, v, tau2)
    assert lo < th < hi


def test_msprt_pvalues_monotone_and_valid() -> None:
    rng = np.random.default_rng(3)
    reps, looks = 2000, 600
    x = rng.standard_normal((reps, looks)).cumsum(axis=1)
    n = np.arange(1, looks + 1)
    theta_hat = x / n
    var = 1.0 / n
    p = msprt_pvalues(theta_hat, var, tau2=0.1)
    assert np.all(np.diff(p, axis=1) <= 0)
    assert (p[:, -1] < 0.05).mean() <= 0.05


def test_expected_loss_vs_monte_carlo() -> None:
    rng = np.random.default_rng(11)
    for m, s in [(0.01, 0.02), (-0.005, 0.01), (0.0, 1.0)]:
        draws = rng.normal(m, s, 1_000_000)
        ls, lh = expected_losses(m, s)
        assert ls == pytest.approx(np.maximum(-draws, 0).mean(), rel=1e-2)
        assert lh == pytest.approx(np.maximum(draws, 0).mean(), rel=1e-2)
        assert lh - ls == pytest.approx(m, abs=1e-12)  # E[max(θ,0)] − E[max(−θ,0)] = E[θ]


def test_normal_posterior_shrinks() -> None:
    m, s = normal_posterior(0.1, 0.05, 0.0, 0.05)
    assert m == pytest.approx(0.05) and s == pytest.approx(0.05 / math.sqrt(2))
    d = bayes_decision(0.1, 0.01, threshold=1e-4)
    assert d.ship and d.prob_positive > 0.99


def test_eb_recovers_prior() -> None:
    rng = np.random.default_rng(5)
    theta = rng.normal(0.01, 0.02, 3000)
    se = rng.uniform(0.005, 0.03, 3000)
    est = theta + rng.normal(0, se)
    prior = fit_normal_prior(est, se)
    assert prior.mean == pytest.approx(0.01, abs=0.002)
    assert prior.sd == pytest.approx(0.02, abs=0.002)
    np_prior = fit_npmle(est, se, grid_size=200)
    assert np_prior.loglik >= prior.loglik - 1e-6
    pm, _ = prior.posterior(est, se)
    assert np.mean((pm - theta) ** 2) < np.mean((est - theta) ** 2)


def test_iv_helpers() -> None:
    assert complier_control_mean(0.1, 0.08, 0.5) == pytest.approx(0.12)
    assert required_n_ratio(0.05, 0.25, 0.04) == pytest.approx(5.0)


def test_bootstrap_agrees_with_welch() -> None:
    rng = np.random.default_rng(9)
    yc, yt = rng.exponential(1.0, 20000), rng.exponential(1.02, 20000)
    b = bootstrap_diff(yc, yt, config=BootstrapConfig(n_boot=1000, seed=1))
    w = welch(MeanStats.from_array(yc), MeanStats.from_array(yt))
    assert (b.ci_high - b.ci_low) == pytest.approx(w.ci_high - w.ci_low, rel=0.05)
    assert b.delta == pytest.approx(w.delta)
    # ratio-metric bootstrap runs
    x = rng.poisson(2, 20000) + 1.0
    br = bootstrap_diff(yc * x, yt * x, x, x, config=BootstrapConfig(n_boot=200))
    assert br.ci_low < br.delta < br.ci_high
    assert needs_bootstrap(100, 1000, 2.0) and not needs_bootstrap(1e6, 1e4, 2.0)
