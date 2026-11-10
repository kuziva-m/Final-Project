"""
scripts/benchmark_store.py -- the pages kept for measuring the free reader.

Shared by scripts/claude_batch.py (adds pages) and scripts/score_free_reader.py
(scores the free reader on them). Everything lives in data/benchmark, which is
never committed -- it holds real business records:

    pages/<page id>.<ext>   each page at full size
    claude/<page id>.json   Claude's first reading of it
    index.json              {"pages": {page id: {...}}}
    runs/<time>/            each scoring run's results

The correct contents of a page are its document in the app ("Benchmark: ..."),
as last saved: a document saved after Claude's reading has been checked.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCH_DIR = Path(os.environ.get("LEDGER_BENCHMARK_DIR") or ROOT / "data" / "benchmark")
PAGES_DIR = BENCH_DIR / "pages"
CLAUDE_DIR = BENCH_DIR / "claude"
RUNS_DIR = BENCH_DIR / "runs"
INDEX_PATH = BENCH_DIR / "index.json"


def load_index() -> dict:
    try:
        index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        index = {}
    index.setdefault("pages", {})
    return index


def save_index(index: dict) -> None:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(INDEX_PATH)


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def correct_tables(conn: sqlite3.Connection, entry: dict) -> tuple[list[dict] | None, bool]:
    """A page's correct tables from its document, and whether someone has
    saved it since Claude read it (checked). ``(None, False)`` if the
    document was deleted."""
    row = conn.execute("SELECT tables_json, updated_at FROM scans WHERE id = ?", (entry["scan_id"],)).fetchone()
    if row is None:
        return None, False
    return json.loads(row["tables_json"]), row["updated_at"] is not None
