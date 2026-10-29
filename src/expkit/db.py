"""One place to open DuckDB connections (quiet progress bar, consistent settings)."""

from __future__ import annotations

import duckdb


def duckdb_connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    return con
