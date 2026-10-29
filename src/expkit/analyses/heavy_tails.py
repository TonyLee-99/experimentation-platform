"""Heavy-tailed revenue metric: what capping (winsorization) buys and costs.

Revenue per user is zero for most users and large for a few. A handful of big spenders dominate
the variance, so tests on revenue are badly underpowered. Capping each user's value at a
threshold (Kohavi et al. 2020 ch. 22 "Variance reduction: capping"; Microsoft ExP, Netflix XP)
trades a little bias (the capped metric is a different estimand) for a lot of variance.

Data: Hillstrom MineThatData email RCT (real, 64,000 customers, 2/3 emailed). Treatment = any
email; metric = spend in the 2 weeks after the campaign. Criteo and ASOS have no unit-level
continuous metric (ASOS metric 4 looks revenue-like but only aggregates are published, so it
cannot be capped).
"""

from __future__ import annotations

import time
from typing import Any

import duckdb
import numpy as np
from scipy import stats

from expkit.analyses.common import SEED, write_json
from expkit.data import HILLSTROM_PARQUET
from expkit.stats.ttest import welch
from expkit.stats.types import MeanStats

CAP_QUANTILES = [0.75, 0.80, 0.90, 0.95, 0.99, 0.999]


def run(n_sims: int = 2000, sub_n: int = 16_000) -> dict[str, Any]:
    t0 = time.time()
    df = (
        duckdb.connect()
        .execute(
            f"SELECT (segment <> 'No E-Mail')::INT AS t, spend::DOUBLE AS y FROM read_parquet('{HILLSTROM_PARQUET}')"
        )
        .fetchnumpy()
    )
    t, y = np.asarray(df["t"]), np.asarray(df["y"])
    buyers = y[y > 0]
    out: dict[str, Any] = {
        "n": int(y.size),
        "share_buyers": float((y > 0).mean()),
        "max_spend": float(y.max()),
        "skewness_all": float(stats.skew(y)),
        "excess_kurtosis_all": float(stats.kurtosis(y)),
        "buyer_spend_quantiles": {str(q): float(np.quantile(buyers, q)) for q in [0.5, *CAP_QUANTILES]},
        "note": "buyer spend tops out at $499.0, consistent with an upstream cap in the published file",
    }
    caps: dict[str, float] = {"none": float("inf")}
    for q in CAP_QUANTILES:
        caps[f"p{q * 100:g}"] = float(np.quantile(buyers, q))  # thresholds from pooled buyers (both arms)
    full = welch(MeanStats.from_array(y[t == 0]), MeanStats.from_array(y[t == 1]), "spend")
    rows = []
    for name, c in caps.items():
        yc = np.minimum(y, c)
        e = welch(MeanStats.from_array(yc[t == 0]), MeanStats.from_array(yc[t == 1]), "spend")
        bias = e.delta - full.delta
        rows.append(
            {
                "cap": name,
                "threshold": c,
                "delta": e.delta,
                "se": e.se,
                "p_value": e.p_value,
                "variance_reduction": 1 - e.se**2 / full.se**2,
                "bias_vs_uncapped_ate": bias,
                "mse": bias**2 + e.se**2,
                "relative_bias": bias / full.delta,
            }
        )
    out["full_sample"] = rows
    # SIMULATION: power on real-data subsamples (resampling real customers, both arms)
    rng = np.random.default_rng(SEED)
    idx_t, idx_c = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    n_t = round(sub_n * t.mean())
    hits = {k: 0 for k in caps}
    for _ in range(n_sims):
        st = rng.choice(idx_t, n_t, replace=True)
        sc = rng.choice(idx_c, sub_n - n_t, replace=True)
        for k, c in caps.items():
            a, b = np.minimum(y[sc], c), np.minimum(y[st], c)
            if welch(MeanStats.from_array(a), MeanStats.from_array(b)).p_value < 0.05:
                hits[k] += 1
    out["power_simulation"] = {
        "label": f"SIMULATION: {n_sims} resamples of {sub_n:,} real customers",
        "power": {k: v / n_sims for k, v in hits.items()},
    }
    out["seconds"] = round(time.time() - t0, 1)
    write_json("heavy_tails", out)
    return out


def main() -> None:
    run()
