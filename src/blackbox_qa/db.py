"""Postgres connection helpers (psycopg 3 + pgvector)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from blackbox_qa.config import settings

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


@contextmanager
def connect(autocommit: bool = False) -> Iterator[psycopg.Connection]:
    """Open a connection with the pgvector adapter registered."""
    conn = psycopg.connect(settings.database_url, autocommit=autocommit)
    try:
        register_vector(conn)
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


def apply_schema() -> None:
    """Apply sql/schema.sql (idempotent)."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect(autocommit=True) as conn:
        conn.execute(sql)


def count_rows(table: str) -> int:
    if not table.isidentifier():
        raise ValueError(f"unsafe table name: {table!r}")
    with connect() as conn:
        row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


if __name__ == "__main__":
    apply_schema()
    print(f"schema applied. reports={count_rows('reports')} "
          f"chunks={count_rows('report_chunks')}")
