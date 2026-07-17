"""Append-only audit log of every batch's predictions.

A durable, delete-proof record: job directories and DB rows are removed when a
batch is deleted from the UI, but this file lives at the data-dir root (a
sibling of jobs/), so nothing in the delete path ever touches it. One JSON line
per file — a successful result carries the 9 schema fields; a failed file
carries its error. Deliberately torch-free and dependency-light so writing it
never adds cost to the worker's completion path.
"""

import json
from datetime import UTC, datetime
from pathlib import Path


def audit_path(data_dir: Path) -> Path:
    return data_dir / "audit.jsonl"


def record_batch(
    data_dir: Path, job_id: str, batch_name: str, report, ts: str | None = None
) -> None:
    """Append one line per file in `report` (duck-typed: `.results` name->result
    with model_dump(), `.errors` list of objects with .name/.error)."""
    ts = ts or datetime.now(UTC).isoformat(timespec="seconds")
    path = audit_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name, result in report.results.items():
        rows.append(
            {
                "ts": ts,
                "job_id": job_id,
                "batch": batch_name,
                "file": name,
                "status": "ok",
                **result.model_dump(mode="json"),
            }
        )
    for err in report.errors:
        rows.append(
            {
                "ts": ts,
                "job_id": job_id,
                "batch": batch_name,
                "file": err.name,
                "status": "error",
                "error": err.error,
            }
        )
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
