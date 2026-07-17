"""SQLite job store. API process owns status transitions; the worker process
writes progress + its own terminal transition. WAL handles the two writers."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  original_name TEXT NOT NULL,
  status TEXT NOT NULL,
  total INTEGER NOT NULL DEFAULT 0,
  done INTEGER NOT NULL DEFAULT 0,
  current_file TEXT,
  warnings TEXT NOT NULL DEFAULT '[]',
  error TEXT,
  started_at TEXT,
  finished_at TEXT,
  results_count INTEGER,
  errors_count INTEGER,
  worker_pid INTEGER
);
"""

TERMINAL = {"completed", "failed", "interrupted"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # autocommit; check_same_thread=False because FastAPI sync endpoints run in a
    # threadpool — sqlite3's default serialized mode makes per-execute calls safe.
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(_SCHEMA)
    return conn


def _to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    d["warnings"] = json.loads(d["warnings"])
    return d


def create_job(db: sqlite3.Connection, job_id: str, original_name: str) -> None:
    db.execute(
        "INSERT INTO jobs (id, created_at, original_name, status) VALUES (?, ?, ?, 'validating')",
        (job_id, _now(), original_name),
    )


def get_job(db: sqlite3.Connection, job_id: str) -> dict | None:
    return _to_dict(db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())


def list_jobs(db: sqlite3.Connection) -> list[dict]:
    rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC").fetchall()
    return [_to_dict(r) for r in rows]


def set_status(db: sqlite3.Connection, job_id: str, status: str, *, error: str | None = None) -> None:
    sets, params = ["status = ?"], [status]
    if error is not None:
        sets.append("error = ?"); params.append(error)
    if status == "running":
        sets.append("started_at = ?"); params.append(_now())
    if status in TERMINAL:
        sets.append("finished_at = ?"); params.append(_now())
    params.append(job_id)
    db.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params)


def set_validation(db: sqlite3.Connection, job_id: str, total: int, warnings: list[str]) -> None:
    db.execute(
        "UPDATE jobs SET total = ?, warnings = ?, status = 'awaiting_confirmation' WHERE id = ?",
        (total, json.dumps(warnings), job_id),
    )


def update_progress(db: sqlite3.Connection, job_id: str, done: int, current_file: str) -> None:
    db.execute(
        "UPDATE jobs SET done = ?, current_file = ? WHERE id = ?", (done, current_file, job_id)
    )


def finish(
    db: sqlite3.Connection, job_id: str, results_count: int, errors_count: int,
    extra_warnings: list[str],
) -> None:
    row = db.execute("SELECT warnings FROM jobs WHERE id = ?", (job_id,)).fetchone()
    merged = json.loads(row["warnings"]) + list(extra_warnings)
    db.execute(
        "UPDATE jobs SET status = 'completed', finished_at = ?, results_count = ?, "
        "errors_count = ?, warnings = ? WHERE id = ?",
        (_now(), results_count, errors_count, json.dumps(merged), job_id),
    )


def set_worker_pid(db: sqlite3.Connection, job_id: str, pid: int) -> None:
    db.execute("UPDATE jobs SET worker_pid = ? WHERE id = ?", (pid, job_id))


def delete_job(db: sqlite3.Connection, job_id: str) -> None:
    db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
