"""Randomization-validity deep dive for Criteo (the "balance finding").

Global SRM passes (the split is 0.85000 to five decimals), but pre-treatment features are
imbalanced across arms. The dataset authors explain why: Criteo v2 pooled several advertiser
incrementality tests and re-sampled each to a common 85% treatment ratio (Diemert et al. 2021,
§3 and erratum). If that re-sampling is not exactly uniform within each test, treatment share
varies with user type — a Simpson's-paradox setup in which the raw difference in means is biased.

This module quantifies it and compares estimators that adjust for observed features:
* treatment share by decile of the prognostic score (cross-fitted P(visit | x));
* post-stratified ITT on 100 prognostic-score bins (a transparent, model-light adjustment);
* the regression-adjusted CUPED/CUPAC estimates from the scorecard.
"""

from __future__ import annotations

import math
from typing import Any

import duckdb

from expkit.analyses.common import CUPAC_PARQUET, write_json
from expkit.data import CRITEO_PARQUET
from expkit.db import duckdb_connect


def _post_stratified(con: duckdb.DuckDBPyConnection, y: str, score: str, bins: int) -> dict[str, float]:
    rows = con.execute(
        f"""WITH u AS (SELECT s.treatment t, s.{y} y, ntile({bins}) OVER (ORDER BY c.{score}) b
                       FROM read_parquet('{CRITEO_PARQUET}') s JOIN read_parquet('{CUPAC_PARQUET}') c USING (unit_id))
            SELECT b, count(*) n, avg(y) FILTER (t=1) m1, avg(y) FILTER (t=0) m0,
                   var_samp(y) FILTER (t=1) v1, var_samp(y) FILTER (t=0) v0,
                   count(*) FILTER (t=1) n1, count(*) FILTER (t=0) n0
            FROM u GROUP BY b"""
    ).fetchall()
    N = sum(r[1] for r in rows)
    est = sum(r[1] / N * (r[2] - r[3]) for r in rows)
    var = sum((r[1] / N) ** 2 * (r[4] / r[6] + r[5] / r[7]) for r in rows)
    base = con.execute(f"SELECT avg({y}) FROM read_parquet('{CRITEO_PARQUET}') WHERE treatment = 0").fetchone()
    assert base is not None
    return {"itt": est, "se": math.sqrt(var), "relative": est / float(base[0]), "bins": bins, "score": score}


def run() -> dict[str, Any]:
    con = duckdb_connect()
    share = con.execute(
        f"""WITH u AS (SELECT s.treatment t, s.visit, ntile(10) OVER (ORDER BY c.cupac_visit) d
                       FROM read_parquet('{CRITEO_PARQUET}') s JOIN read_parquet('{CUPAC_PARQUET}') c USING (unit_id))
            SELECT d, avg(t), count(*), avg(visit) FROM u GROUP BY d ORDER BY d"""
    ).fetchall()
    out: dict[str, Any] = {
        "treatment_share_by_prognostic_decile": [
            {"decile": int(r[0]), "treatment_share": float(r[1]), "n": int(r[2]), "visit_rate": float(r[3])}
            for r in share
        ],
        "post_stratified": {
            "visit": _post_stratified(con, "visit", "cupac_visit", 100),
            "conversion": _post_stratified(con, "conversion", "cupac_conversion", 100),
        },
        "source_note": "Diemert et al. 2021 (arXiv:2111.10106): tests re-sampled to a common treatment ratio; "
        "their C2ST used 300 samples (p = 0.137)",
    }
    write_json("balance", out)
    return out


def main() -> None:
    run()
