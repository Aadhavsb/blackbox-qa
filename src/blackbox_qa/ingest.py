"""Ingest NTSB aviation data into Postgres.

Pipeline: download `avall.zip` -> extract `avall.mdb` -> read tables via
`mdbtools` (`mdb-export`) -> load the structured `reports` table.

Narrative chunking + embeddings are added in a later step and run after this.

Requires the `mdbtools` CLI on PATH (`apt install mdbtools`).
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import subprocess
import urllib.request
import zipfile
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

from blackbox_qa import db

DATA_DIR = Path(os.environ.get("BLACKBOX_DATA_DIR", "data"))
# NTSB serves the dataset through a file-handler endpoint, not a static path.
# fileID is the server-side Windows path, percent-encoded (C:\avdata\avall.zip).
AVALL_URL = os.environ.get(
    "AVALL_URL",
    "https://data.ntsb.gov/avdata/FileDirectory/DownloadFile"
    "?fileID=C%3A%5Cavdata%5Cavall.zip",
)

# NTSB column -> reports column. Pulled from the `events` table except where noted.
_EVENT_COLS = {
    "ev_id": "ev_id",
    "ntsb_no": "ntsb_no",
    "ev_type": "ev_type",
    "ev_city": "ev_city",
    "ev_state": "ev_state",
    "ev_country": "ev_country",
    "ev_highest_injury": "ev_highest_injury",
    "wx_cond_basic": "wx_cond_basic",
    "light_cond": "light_cond",
}
_EVENT_INT = {
    "ev_year": "ev_year",
    "ev_month": "ev_month",
    "inj_tot_f": "inj_tot_fatal",
    "inj_tot_s": "inj_tot_serious",
    "inj_tot_m": "inj_tot_minor",
    "inj_tot_n": "inj_tot_none",
    "inj_tot_t": "inj_tot_total",
}
_AIRCRAFT_COLS = {
    "acft_make": "acft_make",
    "acft_model": "acft_model",
    "acft_series": "acft_series",
    "acft_category": "acft_category",
    "far_part": "far_part",
    "damage": "damage",
}

_REPORT_FIELDS = [
    "ev_id", "ntsb_no", "ev_date", "ev_year", "ev_month", "ev_type",
    "ev_city", "ev_state", "ev_country", "latitude", "longitude",
    "ev_highest_injury", "inj_tot_fatal", "inj_tot_serious", "inj_tot_minor",
    "inj_tot_none", "inj_tot_total", "wx_cond_basic", "light_cond",
    "acft_make", "acft_model", "acft_series", "acft_category", "far_part", "damage",
]

# NTSB narratives table column -> chunk `source` label.
_NARRATIVE_SOURCES = {"narr_accp": "factual", "narr_cause": "cause"}


def _require_mdbtools() -> None:
    if shutil.which("mdb-export") is None:
        raise RuntimeError(
            "mdb-export not found. Install mdbtools (e.g. `sudo apt install mdbtools`)."
        )


def download_and_extract(data_dir: Path = DATA_DIR) -> Path:
    """Download avall.zip (if missing) and extract avall.mdb. Returns the .mdb path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    mdb_path = data_dir / "avall.mdb"
    if mdb_path.exists():
        return mdb_path

    zip_path = data_dir / "avall.zip"
    if not zip_path.exists():
        print(f"downloading {AVALL_URL} -> {zip_path}")
        with urllib.request.urlopen(AVALL_URL) as resp, open(zip_path, "wb") as fh:  # noqa: S310
            shutil.copyfileobj(resp, fh)

    print(f"extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".mdb"))
        with zf.open(member) as src, open(mdb_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return mdb_path


def read_table(mdb_path: Path, table: str) -> Iterator[dict[str, str]]:
    """Yield rows of an Access table as dicts via `mdb-export`."""
    _require_mdbtools()
    proc = subprocess.run(  # noqa: S603
        ["mdb-export", str(mdb_path), table],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    yield from csv.DictReader(io.StringIO(proc.stdout))


def _parse_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _parse_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    head = value.strip().split(" ")[0]
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(head, fmt).date()
        except ValueError:
            continue
    return None


def _primary_aircraft(mdb_path: Path) -> dict[str, dict[str, str]]:
    """Map ev_id -> the first aircraft row's selected fields."""
    out: dict[str, dict[str, str]] = {}
    for row in read_table(mdb_path, "aircraft"):
        ev_id = row.get("ev_id")
        if ev_id and ev_id not in out:
            out[ev_id] = {dst: row.get(src, "") for src, dst in _AIRCRAFT_COLS.items()}
    return out


def _build_report_row(event: dict[str, str], acft: dict[str, str]) -> tuple:
    rec: dict[str, object] = {f: None for f in _REPORT_FIELDS}
    for src, dst in _EVENT_COLS.items():
        rec[dst] = (event.get(src) or "").strip() or None
    for src, dst in _EVENT_INT.items():
        rec[dst] = _parse_int(event.get(src))
    rec["ev_date"] = _parse_date(event.get("ev_date"))
    rec["latitude"] = _parse_float(event.get("latitude"))
    rec["longitude"] = _parse_float(event.get("longitude"))
    for dst, val in acft.items():
        rec[dst] = (val or "").strip() or None
    return tuple(rec[f] for f in _REPORT_FIELDS)


def load_reports(mdb_path: Path, limit: int | None = None, batch_size: int = 1000) -> int:
    """Load the events table (joined to primary aircraft) into `reports`."""
    aircraft = _primary_aircraft(mdb_path)
    placeholders = ", ".join(["%s"] * len(_REPORT_FIELDS))
    cols = ", ".join(_REPORT_FIELDS)
    sql = (
        f"INSERT INTO reports ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT (ev_id) DO NOTHING"
    )

    inserted = 0
    batch: list[tuple] = []
    with db.connect() as conn, conn.cursor() as cur:
        for event in read_table(mdb_path, "events"):
            ev_id = event.get("ev_id")
            if not ev_id:
                continue
            batch.append(_build_report_row(event, aircraft.get(ev_id, {})))
            if len(batch) >= batch_size:
                cur.executemany(sql, batch)
                inserted += len(batch)
                batch.clear()
            if limit is not None and inserted + len(batch) >= limit:
                break
        if batch:
            cur.executemany(sql, batch)
            inserted += len(batch)
    return inserted


def load_chunks(mdb_path: Path, embed_batch: int = 256) -> int:
    """Chunk + embed report narratives into `report_chunks`.

    Only narratives whose ev_id already exists in `reports` are loaded, so run
    the reports stage first.
    """
    from blackbox_qa import chunking, embeddings

    sql = (
        "INSERT INTO report_chunks (ev_id, source, chunk_index, content, embedding) "
        "VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (ev_id, source, chunk_index) DO NOTHING"
    )

    with db.connect() as conn, conn.cursor() as cur:
        existing = {r[0] for r in conn.execute("SELECT ev_id FROM reports").fetchall()}
        meta: list[tuple[str, str, int]] = []
        texts: list[str] = []
        inserted = 0

        def flush() -> None:
            nonlocal inserted
            if not texts:
                return
            vecs = embeddings.embed_passages(texts)
            rows = [(m[0], m[1], m[2], t, v) for m, t, v in zip(meta, texts, vecs, strict=True)]
            cur.executemany(sql, rows)
            inserted += len(rows)
            print(f"  embedded {inserted} chunks...", flush=True)
            meta.clear()
            texts.clear()

        for row in read_table(mdb_path, "narratives"):
            ev_id = row.get("ev_id")
            if not ev_id or ev_id not in existing:
                continue
            for src_col, source in _NARRATIVE_SOURCES.items():
                text = chunking.normalize(row.get(src_col) or "")
                if not text:
                    continue
                for idx, chunk in enumerate(chunking.chunk_text(text)):
                    meta.append((ev_id, source, idx))
                    texts.append(chunk)
                    if len(texts) >= embed_batch:
                        flush()
        flush()
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest NTSB aviation data into Postgres.")
    parser.add_argument("--limit", type=int, default=None, help="max events to load")
    parser.add_argument("--mdb", type=Path, default=None, help="path to an existing avall.mdb")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--stage", choices=("reports", "chunks", "all"), default="all",
        help="which stage(s) to run",
    )
    args = parser.parse_args()

    db.apply_schema()
    mdb_path = args.mdb or (DATA_DIR / "avall.mdb" if args.skip_download else download_and_extract())
    if not mdb_path.exists():
        raise SystemExit(f"mdb not found: {mdb_path}")

    if args.stage in ("reports", "all"):
        n = load_reports(mdb_path, limit=args.limit)
        print(f"loaded {n} reports (total in db: {db.count_rows('reports')})")
    if args.stage in ("chunks", "all"):
        print("chunking + embedding narratives (CPU-heavy; progress every 256 chunks)...")
        n = load_chunks(mdb_path)
        print(f"loaded {n} chunks (total in db: {db.count_rows('report_chunks')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
