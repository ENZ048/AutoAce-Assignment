"""SQLite job store. API process owns status transitions; the worker process
writes progress + its own terminal transition. WAL handles the two writers."""

import json
import sqlite3
from datetime import UTC, datetime
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
  worker_pid INTEGER,
  failed_files TEXT NOT NULL DEFAULT '[]'
);
"""

TERMINAL = {"completed", "failed", "interrupted"}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # autocommit; check_same_thread=False because FastAPI sync endpoints run in a
    # threadpool — sqlite3's default serialized mode makes per-execute calls safe.
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(_SCHEMA)
    # Additive migration for DBs created before failed_files existed.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    if "failed_files" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN failed_files TEXT NOT NULL DEFAULT '[]'")
    return conn


def _to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    d["warnings"] = json.loads(d["warnings"])
    d["failed_files"] = json.loads(d["failed_files"])
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


def set_status(
    db: sqlite3.Connection, job_id: str, status: str, *, error: str | None = None
) -> None:
    sets, params = ["status = ?"], [status]
    if error is not None:
        sets.append("error = ?")
        params.append(error)
    if status == "running":
        sets.append("started_at = ?")
        params.append(_now())
    if status in TERMINAL:
        sets.append("finished_at = ?")
        params.append(_now())
    params.append(job_id)
    db.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params)


def set_validation(db: sqlite3.Connection, job_id: str, total: int, warnings: list[str]) -> None:
    db.execute(
        "UPDATE jobs SET total = ?, warnings = ?, status = 'awaiting_confirmation' WHERE id = ?",
        (total, json.dumps(warnings), job_id),
    )


def update_progress(
    db: sqlite3.Connection,
    job_id: str,
    done: int,
    current_file: str,
    failed: str | None = None,
) -> None:
    db.execute(
        "UPDATE jobs SET done = ?, current_file = ? WHERE id = ?", (done, current_file, job_id)
    )
    if failed:
        # The worker is the sole writer of failed_files, so read-modify-write is safe.
        row = db.execute("SELECT failed_files FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is not None:
            names = json.loads(row["failed_files"])
            if current_file not in names:
                names.append(current_file)
                db.execute(
                    "UPDATE jobs SET failed_files = ? WHERE id = ?", (json.dumps(names), job_id)
                )


def finish(
    db: sqlite3.Connection,
    job_id: str,
    results_count: int,
    errors_count: int,
    extra_warnings: list[str],
) -> None:
    row = db.execute("SELECT warnings FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:  # job deleted while the worker was still running — nothing to finish
        return
    merged = json.loads(row["warnings"]) + list(extra_warnings)
    # status='running' guard: a stale/adopted worker that finishes after the dispatcher
    # or sweep_orphans already moved this row to interrupted/failed must not resurrect it
    # back to completed. 0 rows updated in that case is an expected no-op, not an error.
    db.execute(
        "UPDATE jobs SET status = 'completed', finished_at = ?, results_count = ?, "
        "errors_count = ?, warnings = ? WHERE id = ? AND status = 'running'",
        (_now(), results_count, errors_count, json.dumps(merged), job_id),
    )


def set_worker_pid(db: sqlite3.Connection, job_id: str, pid: int) -> None:
    db.execute("UPDATE jobs SET worker_pid = ? WHERE id = ?", (pid, job_id))


def delete_job(db: sqlite3.Connection, job_id: str) -> None:
    db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def requeue(db: sqlite3.Connection, job_id: str) -> None:
    db.execute(
        "UPDATE jobs SET status = 'queued', done = 0, current_file = NULL, error = NULL, "
        "started_at = NULL, finished_at = NULL, results_count = NULL, errors_count = NULL, "
        "worker_pid = NULL, "  # a re-run must never inherit the dead attempt's (reusable) pid
        "failed_files = '[]' "
        "WHERE id = ?",
        (job_id,),
    )
