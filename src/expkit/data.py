"""Download and prepare the three real public datasets.

All raw files live under ``data/`` (gitignored). Every function is idempotent: if the output
already exists it is reused, so ``make data`` is cheap to re-run.

Sources
-------
* Criteo Uplift v2.1 (Diemert et al. 2018), Hugging Face Hub ``criteo/criteo-uplift``,
  license CC BY-NC-SA 4.0.
* ASOS Digital Experiments Dataset (Liu et al. 2021), OSF node ``64jsb``, license CC BY 4.0.
* Hillstrom MineThatData E-Mail Analytics challenge (2008), minethatdata.com.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

CRITEO_PARQUET = PROCESSED / "criteo.parquet"
ASOS_PARQUET = PROCESSED / "asos.parquet"
HILLSTROM_PARQUET = PROCESSED / "hillstrom.parquet"

OSF_FILES_API = "https://api.osf.io/v2/nodes/64jsb/files/osfstorage/"
HILLSTROM_URL = (
    "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
CRITEO_FEATURES = [f"f{i}" for i in range(12)]


@dataclass(frozen=True)
class DatasetSummary:
    name: str
    path: str
    rows: int
    details: dict[str, float | int | str]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.rename(dest)


def prepare_criteo() -> Path:
    """Download Criteo v2.1 from the HF Hub and write a zstd Parquet with a stable unit_id.

    ``unit_id`` is the 0-based row number of the original CSV. ``fold`` is a deterministic
    pseudo-random 2-fold split (hash of unit_id) used for CUPAC cross-fitting and HTE holdout.
    """
    if CRITEO_PARQUET.exists():
        return CRITEO_PARQUET
    from huggingface_hub import hf_hub_download

    csv_gz = Path(
        hf_hub_download(
            "criteo/criteo-uplift",
            "criteo-research-uplift-v2.1.csv.gz",
            repo_type="dataset",
            local_dir=str(RAW / "criteo"),
        )
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = true")
    con.execute("SET threads = 1")  # deterministic row_number -> unit_id
    feats = ", ".join(CRITEO_FEATURES)
    con.execute(
        f"""
        COPY (
          SELECT (row_number() OVER () - 1)::BIGINT AS unit_id,
                 {feats},
                 treatment::TINYINT AS treatment,
                 exposure::TINYINT  AS exposure,
                 visit::TINYINT     AS visit,
                 conversion::TINYINT AS conversion
          FROM read_csv_auto('{csv_gz}')
        ) TO '{CRITEO_PARQUET}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)
        """
    )
    # The CSV is no longer needed once converted; keep disk usage low.
    csv_gz.unlink(missing_ok=True)
    shutil.rmtree(RAW / "criteo" / ".cache", ignore_errors=True)
    return CRITEO_PARQUET


def _osf_download_links() -> dict[str, str]:
    with urllib.request.urlopen(OSF_FILES_API, timeout=60) as resp:
        payload = json.load(resp)
    return {d["attributes"]["name"]: d["links"]["download"] for d in payload["data"]}


def prepare_asos() -> Path:
    """Download the ASOS CSV via the OSF API (the API gives the real file download link)."""
    if ASOS_PARQUET.exists():
        return ASOS_PARQUET
    csv = RAW / "asos" / "asos_digital_experiments_dataset.csv"
    if not csv.exists():
        links = _osf_download_links()
        _download(links["asos_digital_experiments_dataset.csv"], csv)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    duckdb.connect().execute(
        f"""
        COPY (SELECT experiment_id, variant_id::INT AS variant_id, metric_id::INT AS metric_id,
                     time_since_start, count_c, count_t, mean_c, mean_t, variance_c, variance_t
              FROM read_csv_auto('{csv}')
              ORDER BY experiment_id, variant_id, metric_id, time_since_start)
        TO '{ASOS_PARQUET}' (FORMAT parquet, COMPRESSION zstd)
        """
    )
    return ASOS_PARQUET


def prepare_hillstrom() -> Path:
    if HILLSTROM_PARQUET.exists():
        return HILLSTROM_PARQUET
    csv = RAW / "hillstrom" / "hillstrom.csv"
    if not csv.exists():
        _download(HILLSTROM_URL, csv)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    duckdb.connect().execute(
        f"""
        COPY (SELECT (row_number() OVER () - 1)::BIGINT AS unit_id, *
              FROM read_csv_auto('{csv}'))
        TO '{HILLSTROM_PARQUET}' (FORMAT parquet, COMPRESSION zstd)
        """
    )
    return HILLSTROM_PARQUET


def summarize() -> list[DatasetSummary]:
    """Row counts and headline figures used for acceptance criteria D1-D3."""
    con = duckdb.connect()
    out: list[DatasetSummary] = []
    r = con.execute(
        f"SELECT count(*), avg(treatment), avg(visit), avg(conversion), avg(exposure) FROM '{CRITEO_PARQUET}'"
    ).fetchone()
    assert r is not None
    out.append(
        DatasetSummary(
            "criteo_uplift_v2.1",
            str(CRITEO_PARQUET.relative_to(ROOT)),
            int(r[0]),
            {
                "treatment_ratio": round(float(r[1]), 6),
                "visit_rate": round(float(r[2]), 6),
                "conversion_rate": round(float(r[3]), 6),
                "exposure_rate": round(float(r[4]), 6),
            },
        )
    )
    r = con.execute(
        f"SELECT count(*), count(DISTINCT experiment_id), "
        f"count(DISTINCT experiment_id || '-' || variant_id), "
        f"count(DISTINCT experiment_id || '-' || variant_id || '-' || metric_id) "
        f"FROM '{ASOS_PARQUET}'"
    ).fetchone()
    assert r is not None
    out.append(
        DatasetSummary(
            "asos_digital_experiments",
            str(ASOS_PARQUET.relative_to(ROOT)),
            int(r[0]),
            {"experiments": int(r[1]), "treatments": int(r[2]), "trajectories": int(r[3])},
        )
    )
    r = con.execute(
        f"SELECT count(*), avg((segment <> 'No E-Mail')::INT), avg(spend) FROM '{HILLSTROM_PARQUET}'"
    ).fetchone()
    assert r is not None
    out.append(
        DatasetSummary(
            "hillstrom_email",
            str(HILLSTROM_PARQUET.relative_to(ROOT)),
            int(r[0]),
            {"emailed_share": round(float(r[1]), 6), "mean_spend": round(float(r[2]), 6)},
        )
    )
    return out


def disk_usage_bytes() -> int:
    return sum(p.stat().st_size for p in DATA.rglob("*") if p.is_file())


def prepare_all() -> list[DatasetSummary]:
    prepare_criteo()
    prepare_asos()
    prepare_hillstrom()
    return summarize()
