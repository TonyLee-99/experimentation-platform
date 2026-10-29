"""The SQL metric layer must reproduce numpy sufficient statistics exactly (synthetic parquet)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from expkit.metrics.compiler import AdjustmentRequest, compile_sql, cumulate, run, run_segmented
from expkit.metrics.spec import MetricCatalog
from expkit.stats.cuped import cuped
from expkit.stats.types import CovariateStats, MeanStats, RatioStats


@pytest.fixture()
def catalog(tmp_path: Path) -> tuple[MetricCatalog, pd.DataFrame]:
    rng = np.random.default_rng(1)
    n = 5000
    df = pd.DataFrame(
        {
            "uid": np.arange(n),
            "arm": rng.integers(0, 2, n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    df["clicks"] = rng.poisson(2 + df.x1.clip(-1, 1))
    df["orders"] = rng.binomial(df.clicks, 0.2)
    path = tmp_path / "units.parquet"
    df.to_parquet(path)
    cat = MetricCatalog.model_validate(
        {
            "source": {"table": str(path), "unit": "uid", "variant": "arm"},
            "metrics": [
                {"name": "clicks", "type": "mean", "expr": "clicks"},
                {
                    "name": "orders_per_click",
                    "type": "ratio",
                    "numerator": "orders",
                    "denominator": "clicks",
                },
            ],
            "covariate_sets": {"pre": ["x1", "x2"]},
        }
    )
    return cat, df


def test_mean_and_ratio_stats_match_numpy(catalog: tuple[MetricCatalog, pd.DataFrame]) -> None:
    cat, df = catalog
    vs = run(cat, ["clicks", "orders_per_click"])
    for arm in (0, 1):
        g = df[df.arm == arm]
        ref = MeanStats.from_array(g.clicks.to_numpy(float))
        got = vs[arm].mean("clicks")
        assert got.n == ref.n and got.sum == pytest.approx(ref.sum) and got.sum_sq == pytest.approx(ref.sum_sq)
        rr = RatioStats.from_arrays(g.orders.to_numpy(float), g.clicks.to_numpy(float))
        rg = vs[arm].ratio("orders_per_click")
        assert rg.sum_xy == pytest.approx(rr.sum_xy) and rg.sum_x2 == pytest.approx(rr.sum_x2)


def test_covariate_stats_and_cuped_match_numpy(catalog: tuple[MetricCatalog, pd.DataFrame]) -> None:
    cat, df = catalog
    vs = run(cat, ["clicks"], [AdjustmentRequest("clicks", "pre")])
    cs = {
        a: CovariateStats.from_arrays(df[df.arm == a].clicks.to_numpy(float), df[df.arm == a][["x1", "x2"]].to_numpy())
        for a in (0, 1)
    }
    got = cuped(vs[0].covariate("clicks", ["x1", "x2"]), vs[1].covariate("clicks", ["x1", "x2"]))
    ref = cuped(cs[0], cs[1])
    assert got.delta == pytest.approx(ref.delta, rel=1e-10)
    assert got.se == pytest.approx(ref.se, rel=1e-10)


def test_segments_are_additive(catalog: tuple[MetricCatalog, pd.DataFrame]) -> None:
    cat, _ = catalog
    full = run(cat, ["clicks"])
    parts = run_segmented(cat, ["clicks"], "uid % 4")
    for arm in (0, 1):
        c = cumulate([parts[(arm, k)] for k in range(4)])
        assert c.n == full[arm].n
        assert c.mean("clicks").sum == pytest.approx(full[arm].mean("clicks").sum)


def test_event_grain_aggregates_to_units(tmp_path: Path) -> None:
    ev = pd.DataFrame({"uid": [1, 1, 2, 3, 3, 3], "arm": [0, 0, 1, 1, 1, 1], "v": [1, 2, 3, 4, 5, 6]})
    p = tmp_path / "ev.parquet"
    ev.to_parquet(p)
    cat = MetricCatalog.model_validate(
        {
            "source": {"table": str(p), "unit": "uid", "variant": "arm", "grain": "event"},
            "metrics": [{"name": "v", "type": "mean", "expr": "v"}],
        }
    )
    assert 'GROUP BY "uid"' in compile_sql(cat, ["v"])
    vs = run(cat, ["v"])
    assert vs[0].n == 1 and vs[0].mean("v").sum == 3
    assert vs[1].n == 2 and vs[1].mean("v").sum_sq == 3**2 + 15**2


def test_metric_validation() -> None:
    with pytest.raises(ValueError):
        MetricCatalog.model_validate(
            {
                "source": {"table": "x", "unit": "u", "variant": "v"},
                "metrics": [{"name": "r", "type": "ratio", "numerator": "a"}],
            }
        )


def test_skewness_from_sums(catalog: tuple[MetricCatalog, pd.DataFrame]) -> None:
    from scipy import stats

    cat, df = catalog
    vs = run(cat, ["clicks"])
    assert vs[0].skewness("clicks") == pytest.approx(stats.skew(df[df.arm == 0].clicks.to_numpy(float)), rel=1e-9)
