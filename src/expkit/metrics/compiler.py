"""Compile metric definitions into one DuckDB SQL query of per-variant sufficient statistics.

The query has two layers, mirroring how warehouse-native platforms work:

1. ``units`` — one row per randomization unit (if the source is event-level we GROUP BY unit);
2. the outer SELECT — one row per variant with only additive sums: n, Σy, Σy², and for
   regression adjustment Σx, Σxy, Σxxᵀ.

Only that tiny result (a handful of rows) leaves the warehouse. The stats engine never sees
unit-level data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np

from expkit.metrics.spec import Metric, MetricCatalog
from expkit.stats.types import CovariateStats, MeanStats, RatioStats


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class AdjustmentRequest:
    """Ask for regression-adjustment sums of ``metric`` on covariate set ``covariate_set``."""

    metric: str
    covariate_set: str


def _unit_columns(cat: MetricCatalog, metrics: list[Metric]) -> list[tuple[str, str]]:
    """(alias, expression) pairs of per-unit columns needed by the requested metrics."""
    cols: dict[str, str] = {}
    for m in metrics:
        if m.type == "mean":
            assert m.expr is not None
            cols[f"m__{m.name}"] = m.expr
        else:
            assert m.numerator is not None and m.denominator is not None
            cols[f"m__{m.name}__num"] = m.numerator
            cols[f"m__{m.name}__den"] = m.denominator
    return list(cols.items())


def compile_sql(
    cat: MetricCatalog,
    metric_names: list[str],
    adjustments: list[AdjustmentRequest] | None = None,
    where: str | None = None,
    segment: str | None = None,
) -> str:
    """``segment`` is an optional extra per-unit SQL expression grouped alongside the variant
    (e.g. a look/batch index for sequential monitoring). Its value is returned in column ``seg``."""
    src = cat.source
    metrics = [cat.get(n) for n in metric_names]
    adjustments = adjustments or []
    unit_cols = _unit_columns(cat, metrics)
    cov_cols = sorted({c for a in adjustments for c in cat.covariate_sets[a.covariate_set]})

    base = f"read_parquet('{src.table}') AS s"
    for i, j in enumerate(src.joins):
        base += f" LEFT JOIN read_parquet('{j.table}') AS j{i} USING ({_q(j.key)})"
    where_sql = f"WHERE {where}" if where else ""

    if src.grain == "event":
        if segment:
            raise NotImplementedError("segments are only supported for unit-grain sources")
        sel = [f"{_q(src.unit)}", f"ANY_VALUE({_q(src.variant)}) AS variant"]
        sel += [f"SUM(CAST({e} AS DOUBLE)) AS {_q(a)}" for a, e in unit_cols]
        sel += [f"ANY_VALUE(CAST({_q(c)} AS DOUBLE)) AS {_q('x__' + c)}" for c in cov_cols]
        units = f"SELECT {', '.join(sel)} FROM {base} {where_sql} GROUP BY {_q(src.unit)}"
    else:
        sel = [f"{_q(src.variant)} AS variant"]
        if segment:
            sel.append(f"{segment} AS seg")
        sel += [f"CAST({e} AS DOUBLE) AS {_q(a)}" for a, e in unit_cols]
        sel += [f"CAST({_q(c)} AS DOUBLE) AS {_q('x__' + c)}" for c in cov_cols]
        units = f"SELECT {', '.join(sel)} FROM {base} {where_sql}"

    agg = ["variant", *(["seg"] if segment else []), "COUNT(*) AS n"]
    for a, _ in unit_cols:
        agg.append(f"SUM({_q(a)}) AS {_q(a + '__s')}")
        agg.append(f"SUM({_q(a)} * {_q(a)}) AS {_q(a + '__ss')}")
    for m in metrics:
        if m.type == "ratio":
            num, den = f"m__{m.name}__num", f"m__{m.name}__den"
            agg.append(f"SUM({_q(num)} * {_q(den)}) AS {_q(f'm__{m.name}__xy')}")
    for i, c in enumerate(cov_cols):
        agg.append(f"SUM({_q('x__' + c)}) AS {_q(f'x__{c}__s')}")
        for c2 in cov_cols[i:]:
            agg.append(f"SUM({_q('x__' + c)} * {_q('x__' + c2)}) AS {_q(f'xx__{c}__{c2}')}")
    seen: set[tuple[str, str]] = set()
    for adj in adjustments:
        for c in cat.covariate_sets[adj.covariate_set]:
            if (adj.metric, c) in seen:
                continue
            seen.add((adj.metric, c))
            agg.append(f"SUM({_q('m__' + adj.metric)} * {_q('x__' + c)}) AS {_q(f'xy__{adj.metric}__{c}')}")
    group = "variant, seg" if segment else "variant"
    return f"WITH units AS ({units})\nSELECT {', '.join(agg)}\nFROM units GROUP BY {group} ORDER BY {group}"


@dataclass
class VariantStats:
    variant: Any
    n: float
    row: dict[str, float]

    def mean(self, metric: str) -> MeanStats:
        return MeanStats(self.n, self.row[f"m__{metric}__s"], self.row[f"m__{metric}__ss"])

    def ratio(self, metric: str) -> RatioStats:
        r = self.row
        return RatioStats(
            self.n,
            r[f"m__{metric}__num__s"],
            r[f"m__{metric}__den__s"],
            r[f"m__{metric}__num__ss"],
            r[f"m__{metric}__den__ss"],
            r[f"m__{metric}__xy"],
        )

    def covariate(self, metric: str, covariates: list[str]) -> CovariateStats:
        r = self.row
        k = len(covariates)
        sx = np.array([r[f"x__{c}__s"] for c in covariates])
        sxy = np.array([r[f"xy__{metric}__{c}"] for c in covariates])
        sxx = np.empty((k, k))
        for i, ci in enumerate(covariates):
            for j, cj in enumerate(covariates):
                a, b = sorted([ci, cj])
                key = f"xx__{a}__{b}" if f"xx__{a}__{b}" in r else f"xx__{b}__{a}"
                sxx[i, j] = r[key]
        return CovariateStats(self.n, r[f"m__{metric}__s"], r[f"m__{metric}__ss"], sx, sxy, sxx)


def run(
    cat: MetricCatalog,
    metric_names: list[str],
    adjustments: list[AdjustmentRequest] | None = None,
    where: str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[Any, VariantStats]:
    """Execute the compiled SQL and return sufficient statistics per variant."""
    sql = compile_sql(cat, metric_names, adjustments, where)
    con = con or duckdb.connect()
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    out: dict[Any, VariantStats] = {}
    for tup in cur.fetchall():
        row = dict(zip(names, tup, strict=True))
        v = row.pop("variant")
        n = float(row.pop("n"))
        out[v] = VariantStats(v, n, {k: float(val) for k, val in row.items()})
    return out


def run_segmented(
    cat: MetricCatalog,
    metric_names: list[str],
    segment: str,
    adjustments: list[AdjustmentRequest] | None = None,
    where: str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[tuple[Any, Any], VariantStats]:
    """Sufficient statistics per (variant, segment). Because every column is a plain sum,
    cumulative statistics up to look k are just the sum over segments <= k."""
    sql = compile_sql(cat, metric_names, adjustments, where=where, segment=segment)
    con = con or duckdb.connect()
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    out: dict[tuple[Any, Any], VariantStats] = {}
    for tup in cur.fetchall():
        row = dict(zip(names, tup, strict=True))
        v, g = row.pop("variant"), row.pop("seg")
        n = float(row.pop("n"))
        out[(v, g)] = VariantStats(v, n, {k: float(val) for k, val in row.items()})
    return out


def cumulate(parts: list[VariantStats]) -> VariantStats:
    """Add sufficient statistics of several segments."""
    row: dict[str, float] = {}
    for p in parts:
        for k, val in p.row.items():
            row[k] = row.get(k, 0.0) + val
    return VariantStats(parts[0].variant, sum(p.n for p in parts), row)
