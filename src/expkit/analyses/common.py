"""Shared helpers for analyses: loading Criteo columns, writing results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from expkit.data import CRITEO_FEATURES, CRITEO_PARQUET, PROCESSED, ROOT

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
CUPAC_PARQUET = PROCESSED / "criteo_cupac.parquet"
SEED = 20261029


def load_criteo(columns: list[str], where: str | None = None, with_cupac: bool = False) -> dict[str, np.ndarray]:
    """Load Criteo columns as numpy arrays (features float32, labels int8) ordered by unit_id."""
    src = f"read_parquet('{CRITEO_PARQUET}') s"
    if with_cupac:
        src += f" JOIN read_parquet('{CUPAC_PARQUET}') c USING (unit_id)"
    sql = f"SELECT {', '.join(columns)} FROM {src} {'WHERE ' + where if where else ''} ORDER BY unit_id"
    tbl = duckdb.connect().execute(sql).fetchnumpy()
    out: dict[str, np.ndarray] = {}
    for c in columns:
        key = c.split(".")[-1].split(" AS ")[-1].strip()
        arr = np.asarray(tbl[key])
        out[key] = arr.astype(np.float32) if key in CRITEO_FEATURES or arr.dtype.kind == "f" else arr
    return out


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(_clean(payload), indent=2))
    return path


def read_json(name: str) -> dict[str, Any]:
    out: dict[str, Any] = json.loads((RESULTS / f"{name}.json").read_text())
    return out
