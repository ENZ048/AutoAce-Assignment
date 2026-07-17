# AutoAce Evaluation Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hosted evaluation dashboard for the AutoAce trial — login, batch upload, validation-before-processing, reload-safe progress, results review, downloads — wrapped around the existing `autoace_audio` pipeline.

**Architecture:** One FastAPI process (`src/dashboard/`) serves `/api/*` plus the built React SPA (`webapp/dist`). Jobs live in SQLite (`data/dashboard.db`, WAL) + job dirs (`data/jobs/<id>/`). Batch processing runs in one spawned worker process at a time calling `autoace_audio.batch.run_batch(extracted_dir, out_dir, progress_cb=…)`; a dispatcher promotes queued jobs and sweeps orphans on startup.

**Tech Stack:** FastAPI, uvicorn, PyJWT, bcrypt, python-multipart, sqlite3 (stdlib), multiprocessing (spawn); Vite + React 18 + Tailwind + react-router + axios + react-hot-toast + lucide-react + `@fontsource/sora`.

**Spec:** `docs/superpowers/specs/2026-07-17-dashboard-design.md` — the requirements source. Read it before starting.

## Global Constraints

- Repo `/Users/kishorrane/Test/AutoAce-Dashboard`, branch `dashboard` only; push `git push -u origin dashboard`.
- **Never modify `src/autoace_audio/`** (backend requests go in `docs/DASHBOARD-BACKEND-REQUESTS.md`).
- Fully self-contained project: no code/assets/references from any other project; no CDN assets; no new third-party services.
- Python venv is prebuilt at `.venv/` — run tests with `.venv/bin/pytest`, install with `.venv/bin/pip`.
- Every commit: conventional message ending `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; before EVERY commit `git status` must show nothing from `data/`, `.env`, `.superpowers/`, `out/`, `models_cache/`.
- No verbatim customer speech in committed files or logs; logs carry filenames/ids/statuses only.
- All new pytest tests must be fast (no models, no network) so they join `make test` (which runs `-m "not slow and not network"`).
- `transformers` pin and all existing deps unchanged.
- Job states (exact strings): `validating`, `awaiting_confirmation`, `queued`, `running`, `completed`, `failed`, `interrupted`.
- The 9 result fields (exact order): `emotional_tone`, `emotional_intensity`, `background_noise_present`, `background_noise_type`, `background_noise_severity`, `audio_quality`, `speaker_overlap_present`, `long_silence_present`, `confidence`.

## File Structure

```
src/dashboard/
  __init__.py        # empty marker
  config.py          # DashboardSettings (pydantic-settings) + get_dashboard_settings()
  hash_password.py   # python -m dashboard.hash_password → bcrypt hash for .env
  store.py           # SQLite job store: schema + CRUD + progress writes
  auth.py            # bcrypt verify, JWT issue/verify, require_auth dependency
  zipsafe.py         # zip-slip-safe extraction + batch-root resolution
  runner.py          # worker_main (runs run_batch), stub analyze, dispatch_once, orphan sweep
  api.py             # all /api routes
  app.py             # create_app(): routes + lifespan dispatcher + static SPA serving
tests/web/
  __init__.py, conftest.py           # tmp data dir, settings env, TestClient, auth header
  test_config.py, test_store.py, test_auth.py, test_zipsafe.py,
  test_upload.py, test_runner.py, test_job_routes.py
webapp/                               # Vite React SPA (Task 9-12)
  src/{api.js, auth.js, main.jsx, App.jsx, pages/*.jsx, components/*.jsx, index.css}
Makefile             # + webapp-build, web, web-dev targets
.env.example         # + DASHBOARD_* keys
pyproject.toml       # + [project.optional-dependencies] web
```

---

### Task 1: Dashboard settings + password hash helper

**Files:**
- Modify: `pyproject.toml` (add `web` extra after the `dev` extra)
- Create: `src/dashboard/__init__.py`, `src/dashboard/config.py`, `src/dashboard/hash_password.py`
- Test: `tests/web/__init__.py`, `tests/web/test_config.py`

**Interfaces:**
- Produces: `dashboard.config.DashboardSettings` with fields `admin_user: str`, `admin_password_hash: str`, `jwt_secret: str`, `max_upload_mb: int = 1024`, `stub_analyze: bool = False`, `data_dir: Path = Path("data")` (env prefix `DASHBOARD_`); `get_dashboard_settings()` cached accessor; `clear_settings_cache()` for tests.
- Produces: `python -m dashboard.hash_password <plaintext>` prints a bcrypt hash.

- [ ] **Step 1: Add the `web` extra and install**

In `pyproject.toml`, extend the optional dependencies:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-timeout>=2.3", "ruff>=0.5"]
web = [
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "python-multipart>=0.0.9",
    "bcrypt>=4.1",
    "pyjwt>=2.8",
    "httpx>=0.27",
]
```

(`httpx` is required by FastAPI's TestClient; keeping it in `web` keeps `dev` core-only.)

Run: `.venv/bin/pip install -e '.[web,dev]' -q && .venv/bin/python -c "import fastapi, jwt, bcrypt, multipart; print('ok')"`
Expected: `ok`

- [ ] **Step 2: Write the failing config test**

`tests/web/__init__.py` — empty file. `tests/web/test_config.py`:

```python
import pytest

from dashboard.config import DashboardSettings, clear_settings_cache, get_dashboard_settings


def _set_required(monkeypatch):
    monkeypatch.setenv("DASHBOARD_ADMIN_USER", "autoace")
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_HASH", "$2b$12$abcdefghijklmnopqrstuv")
    monkeypatch.setenv("DASHBOARD_JWT_SECRET", "s3cret")


def test_settings_load_from_env(monkeypatch, tmp_path):
    _set_required(monkeypatch)
    monkeypatch.setenv("DASHBOARD_DATA_DIR", str(tmp_path))
    clear_settings_cache()
    s = get_dashboard_settings()
    assert s.admin_user == "autoace"
    assert s.max_upload_mb == 1024  # default
    assert s.stub_analyze is False  # default
    assert str(s.data_dir) == str(tmp_path)


def test_missing_required_key_fails_fast(monkeypatch):
    monkeypatch.delenv("DASHBOARD_ADMIN_USER", raising=False)
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_HASH", "$2b$12$abcdefghijklmnopqrstuv")
    monkeypatch.setenv("DASHBOARD_JWT_SECRET", "s3cret")
    clear_settings_cache()
    with pytest.raises(Exception, match="(?i)admin_user"):
        get_dashboard_settings()


def test_settings_cached_until_cleared(monkeypatch, tmp_path):
    _set_required(monkeypatch)
    clear_settings_cache()
    first = get_dashboard_settings()
    monkeypatch.setenv("DASHBOARD_ADMIN_USER", "other")
    assert get_dashboard_settings() is first
    clear_settings_cache()
    assert get_dashboard_settings().admin_user == "other"
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/web/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard'`

- [ ] **Step 4: Implement config**

`src/dashboard/__init__.py` — empty file. `src/dashboard/config.py`:

```python
"""Dashboard-only settings. Deliberately separate from autoace_audio.config."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DASHBOARD_", env_file=".env", extra="ignore")

    admin_user: str
    admin_password_hash: str
    jwt_secret: str
    max_upload_mb: int = 1024
    stub_analyze: bool = False  # dev/test only: canned analyze, no models/keys
    data_dir: Path = Path("data")


_cached: DashboardSettings | None = None


def get_dashboard_settings() -> DashboardSettings:
    global _cached
    if _cached is None:
        _cached = DashboardSettings()  # raises with the missing field named — fail fast
    return _cached


def clear_settings_cache() -> None:
    global _cached
    _cached = None
```

- [ ] **Step 5: Run config tests**

Run: `.venv/bin/pytest tests/web/test_config.py -v`
Expected: 3 PASS

- [ ] **Step 6: Write failing hash-helper test** (append to `tests/web/test_config.py`)

```python
def test_hash_password_roundtrip():
    import bcrypt

    from dashboard.hash_password import make_hash

    h = make_hash("Trial#2026")
    assert h.startswith("$2b$")
    assert bcrypt.checkpw(b"Trial#2026", h.encode())
```

Run: `.venv/bin/pytest tests/web/test_config.py::test_hash_password_roundtrip -v`
Expected: FAIL — no module `dashboard.hash_password`

- [ ] **Step 7: Implement `src/dashboard/hash_password.py`**

```python
"""Generate the bcrypt hash for DASHBOARD_ADMIN_PASSWORD_HASH.

Usage: .venv/bin/python -m dashboard.hash_password '<plaintext>'
"""

import sys

import bcrypt


def make_hash(plaintext: str) -> str:
    return bcrypt.hashpw(plaintext.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m dashboard.hash_password '<plaintext>'")
    print(make_hash(sys.argv[1]))
```

- [ ] **Step 8: Run all Task 1 tests + full suite**

Run: `.venv/bin/pytest tests/web/ -v && make test`
Expected: 4 PASS; existing 120 still pass.

- [ ] **Step 9: Update `.env.example`** (append)

```
# --- dashboard (web UI) ---
DASHBOARD_ADMIN_USER=            # login username handed to AutoAce
DASHBOARD_ADMIN_PASSWORD_HASH=   # bcrypt hash: .venv/bin/python -m dashboard.hash_password '<plaintext>'
DASHBOARD_JWT_SECRET=            # long random string, e.g. `openssl rand -hex 32`
# DASHBOARD_MAX_UPLOAD_MB=1024
# DASHBOARD_STUB_ANALYZE=0      # dev/test ONLY: canned results, no models/keys
```

- [ ] **Step 10: Commit**

```bash
git status   # confirm nothing from data/ .env .superpowers/ out/ models_cache/
git add pyproject.toml src/dashboard/ tests/web/ .env.example
git commit -m "feat(dashboard): settings, web deps, password-hash helper

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: SQLite job store

**Files:**
- Create: `src/dashboard/store.py`
- Test: `tests/web/test_store.py`

**Interfaces:**
- Consumes: `dashboard.config.get_dashboard_settings()` (for nothing — store takes an explicit db path; keeps it unit-testable).
- Produces (all functions take `db: sqlite3.Connection` as first arg):
  - `connect(db_path: Path) -> sqlite3.Connection` (WAL, row_factory=Row, autocommit)
  - `create_job(db, job_id: str, original_name: str) -> None` (status `validating`)
  - `get_job(db, job_id) -> dict | None`, `list_jobs(db) -> list[dict]` (newest first)
  - `set_status(db, job_id, status: str, *, error: str | None = None) -> None` (stamps `started_at` on `running`, `finished_at` on terminal)
  - `set_validation(db, job_id, total: int, warnings: list[str]) -> None` (also sets status `awaiting_confirmation`)
  - `update_progress(db, job_id, done: int, current_file: str) -> None`
  - `finish(db, job_id, results_count: int, errors_count: int, extra_warnings: list[str]) -> None` (appends warnings, sets counts + status `completed`)
  - `set_worker_pid(db, job_id, pid: int) -> None`
  - `delete_job(db, job_id) -> None`
  - Job dict keys: `id, created_at, original_name, status, total, done, current_file, warnings (list), error, started_at, finished_at, results_count, errors_count, worker_pid`.

- [ ] **Step 1: Write failing store tests** — `tests/web/test_store.py`:

```python
import sqlite3

import pytest

from dashboard import store


@pytest.fixture()
def db(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    yield conn
    conn.close()


def test_create_and_get_job(db):
    store.create_job(db, "abc123", "batch.zip")
    job = store.get_job(db, "abc123")
    assert job["id"] == "abc123"
    assert job["original_name"] == "batch.zip"
    assert job["status"] == "validating"
    assert job["warnings"] == []
    assert job["done"] == 0 and job["total"] == 0


def test_get_missing_job_returns_none(db):
    assert store.get_job(db, "nope") is None


def test_validation_sets_total_warnings_and_status(db):
    store.create_job(db, "j1", "b.zip")
    store.set_validation(db, "j1", total=3, warnings=["manifest row has no file on disk: x.wav"])
    job = store.get_job(db, "j1")
    assert job["status"] == "awaiting_confirmation"
    assert job["total"] == 3
    assert job["warnings"] == ["manifest row has no file on disk: x.wav"]


def test_status_transitions_stamp_times(db):
    store.create_job(db, "j1", "b.zip")
    store.set_status(db, "j1", "queued")
    assert store.get_job(db, "j1")["started_at"] is None
    store.set_status(db, "j1", "running")
    assert store.get_job(db, "j1")["started_at"] is not None
    store.set_status(db, "j1", "failed", error="boom")
    job = store.get_job(db, "j1")
    assert job["finished_at"] is not None and job["error"] == "boom"


def test_progress_and_finish(db):
    store.create_job(db, "j1", "b.zip")
    store.set_validation(db, "j1", total=2, warnings=[])
    store.update_progress(db, "j1", done=1, current_file="a.wav")
    job = store.get_job(db, "j1")
    assert job["done"] == 1 and job["current_file"] == "a.wav"
    store.finish(db, "j1", results_count=1, errors_count=1, extra_warnings=["1/2 files fell back"])
    job = store.get_job(db, "j1")
    assert job["status"] == "completed"
    assert job["results_count"] == 1 and job["errors_count"] == 1
    assert job["warnings"] == ["1/2 files fell back"]


def test_list_jobs_newest_first_and_delete(db):
    store.create_job(db, "old", "a.zip")
    store.create_job(db, "new", "b.zip")
    ids = [j["id"] for j in store.list_jobs(db)]
    assert ids == ["new", "old"]
    store.delete_job(db, "old")
    assert store.get_job(db, "old") is None


def test_two_connections_see_each_other(tmp_path):
    a = store.connect(tmp_path / "t.db")
    b = store.connect(tmp_path / "t.db")
    store.create_job(a, "j1", "b.zip")
    store.update_progress(b, "j1", done=1, current_file="x.wav")
    assert store.get_job(a, "j1")["done"] == 1
    a.close(); b.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/web/test_store.py -v`
Expected: FAIL — `cannot import name 'store'`

- [ ] **Step 3: Implement `src/dashboard/store.py`**

```python
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
    rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC, id DESC").fetchall()
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
```

- [ ] **Step 4: Run store tests**

Run: `.venv/bin/pytest tests/web/test_store.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git status
git add src/dashboard/store.py tests/web/test_store.py
git commit -m "feat(dashboard): SQLite job store with WAL + progress writes

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Auth — bcrypt login, JWT issue/verify, FastAPI dependency

**Files:**
- Create: `src/dashboard/auth.py`
- Test: `tests/web/test_auth.py`

**Interfaces:**
- Consumes: `DashboardSettings` (admin_user, admin_password_hash, jwt_secret).
- Produces: `verify_login(username: str, password: str, settings) -> bool`; `create_token(settings) -> str` (HS256, `sub`=admin user, `exp`=now+24 h); `decode_token(token: str, settings) -> dict | None`; `require_auth` FastAPI dependency (returns the `sub` string, raises 401 otherwise); `TOKEN_TTL_S = 86400`.

- [ ] **Step 1: Write failing auth tests** — `tests/web/test_auth.py`:

```python
import time

import jwt as pyjwt

from dashboard.auth import TOKEN_TTL_S, create_token, decode_token, verify_login
from dashboard.config import DashboardSettings
from dashboard.hash_password import make_hash


def make_settings(**over):
    base = dict(
        admin_user="autoace",
        admin_password_hash=make_hash("Right#Pass1"),
        jwt_secret="test-secret",
    )
    base.update(over)
    return DashboardSettings(**base)


def test_verify_login_correct():
    assert verify_login("autoace", "Right#Pass1", make_settings()) is True


def test_verify_login_wrong_password_or_user():
    s = make_settings()
    assert verify_login("autoace", "wrong", s) is False
    assert verify_login("intruder", "Right#Pass1", s) is False


def test_token_roundtrip():
    s = make_settings()
    payload = decode_token(create_token(s), s)
    assert payload["sub"] == "autoace"
    assert payload["exp"] > time.time() + TOKEN_TTL_S - 120


def test_expired_token_rejected():
    s = make_settings()
    stale = pyjwt.encode(
        {"sub": "autoace", "exp": int(time.time()) - 10}, s.jwt_secret, algorithm="HS256"
    )
    assert decode_token(stale, s) is None


def test_garbage_and_wrong_secret_rejected():
    s = make_settings()
    assert decode_token("not.a.token", s) is None
    other = pyjwt.encode(
        {"sub": "autoace", "exp": int(time.time()) + 600}, "other-secret", algorithm="HS256"
    )
    assert decode_token(other, s) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/web/test_auth.py -v`
Expected: FAIL — `cannot import name 'auth'` / module not found

- [ ] **Step 3: Implement `src/dashboard/auth.py`**

```python
"""Single-admin auth: bcrypt against the env-provisioned hash, stateless HS256 JWT."""

import time

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from dashboard.config import DashboardSettings, get_dashboard_settings

TOKEN_TTL_S = 24 * 3600

_bearer = HTTPBearer(auto_error=False)


def verify_login(username: str, password: str, settings: DashboardSettings) -> bool:
    if username != settings.admin_user:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), settings.admin_password_hash.encode("utf-8"))
    except ValueError:  # malformed stored hash — treat as auth failure, never a 500
        return False


def create_token(settings: DashboardSettings) -> str:
    payload = {"sub": settings.admin_user, "exp": int(time.time()) + TOKEN_TTL_S}
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_token(token: str, settings: DashboardSettings) -> dict | None:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None


def require_auth(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(creds.credentials, get_dashboard_settings())
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload["sub"]
```

- [ ] **Step 4: Run auth tests**

Run: `.venv/bin/pytest tests/web/test_auth.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git status
git add src/dashboard/auth.py tests/web/test_auth.py
git commit -m "feat(dashboard): bcrypt login + stateless JWT auth dependency

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Zip-slip-safe extraction + batch-root resolution

**Files:**
- Create: `src/dashboard/zipsafe.py`
- Test: `tests/web/test_zipsafe.py`

**Interfaces:**
- Produces: `extract_zip(zip_path: Path, dest: Path) -> Path` — extracts into `dest`, returns the resolved batch root (`dest` itself, or its single subdir per the backend's `_unzip_if_needed` semantics with root CSVs moved into that subdir); raises `UnsafeZipError(ValueError)` before extracting anything if any member path is absolute or contains `..`.
- Consumed by: Task 6 upload route.

- [ ] **Step 1: Write failing tests** — `tests/web/test_zipsafe.py`:

```python
import zipfile

import pytest

from dashboard.zipsafe import UnsafeZipError, extract_zip


def make_zip(path, entries: dict[str, bytes]):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def test_flat_zip_extracts_to_dest_root(tmp_path):
    z = make_zip(tmp_path / "b.zip", {"call_001.wav": b"RIFF", "labels.csv": b"name,result_json\n"})
    root = extract_zip(z, tmp_path / "x")
    assert root == tmp_path / "x"
    assert (root / "call_001.wav").read_bytes() == b"RIFF"
    assert (root / "labels.csv").exists()


def test_single_subdir_becomes_root_and_csv_moves_in(tmp_path):
    z = make_zip(
        tmp_path / "b.zip",
        {"batch/call_001.wav": b"RIFF", "labels.csv": b"name,result_json\n"},
    )
    root = extract_zip(z, tmp_path / "x")
    assert root == tmp_path / "x" / "batch"
    assert (root / "call_001.wav").exists()
    assert (root / "labels.csv").exists()          # moved into the subdir
    assert not (tmp_path / "x" / "labels.csv").exists()


@pytest.mark.parametrize("evil", ["../evil.txt", "a/../../evil.txt", "/abs.txt"])
def test_hostile_members_rejected_before_extraction(tmp_path, evil):
    z = make_zip(tmp_path / "b.zip", {"ok.wav": b"RIFF", evil: b"x"})
    dest = tmp_path / "x"
    with pytest.raises(UnsafeZipError):
        extract_zip(z, dest)
    assert not (dest / "ok.wav").exists()          # nothing extracted at all
    assert not (tmp_path / "evil.txt").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/web/test_zipsafe.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `src/dashboard/zipsafe.py`**

```python
"""Zip-slip-safe extraction + batch-root resolution.

Mirrors autoace_audio.batch._unzip_if_needed's root-resolution semantics.
Exists because ZipFile.extractall lacks member-path safety and we extract into
the job directory, not a tempdir. Backend code is never modified."""

import shutil
import zipfile
from pathlib import Path


class UnsafeZipError(ValueError):
    """A zip member path would escape the extraction directory."""


def _validate_members(zf: zipfile.ZipFile, dest: Path) -> None:
    base = dest.resolve()
    for info in zf.infolist():
        name = info.filename
        if name.startswith(("/", "\\")) or ".." in Path(name).parts:
            raise UnsafeZipError(f"unsafe path in zip: {name!r}")
        if not (dest / name).resolve().is_relative_to(base):
            raise UnsafeZipError(f"unsafe path in zip: {name!r}")


def extract_zip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        _validate_members(zf, dest)  # all-or-nothing: validate before extracting anything
        for info in zf.infolist():
            if info.is_dir():
                (dest / info.filename).mkdir(parents=True, exist_ok=True)
                continue
            target = dest / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    return _resolve_batch_root(dest)


def _resolve_batch_root(dest: Path) -> Path:
    """Same rules as the backend: root wins if it has non-CSV files; else a single
    subdir becomes the batch root and root-level CSVs move into it."""
    non_csv = [p for p in dest.iterdir() if p.is_file() and p.suffix.lower() != ".csv"]
    if non_csv:
        return dest
    subdirs = [d for d in dest.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        for csv_file in dest.glob("*.csv"):
            shutil.move(str(csv_file), str(subdirs[0] / csv_file.name))
        return subdirs[0]
    return dest
```

- [ ] **Step 4: Run zipsafe tests**

Run: `.venv/bin/pytest tests/web/test_zipsafe.py -v`
Expected: 5 PASS (3 parametrized hostile cases)

- [ ] **Step 5: Commit**

```bash
git status
git add src/dashboard/zipsafe.py tests/web/test_zipsafe.py
git commit -m "feat(dashboard): zip-slip-safe extraction with batch-root resolution

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: App factory, test fixtures, login route, authenticated jobs list

**Files:**
- Create: `src/dashboard/app.py`, `src/dashboard/api.py`, `tests/web/conftest.py`
- Test: append to `tests/web/test_auth.py`

**Interfaces:**
- Consumes: `store.connect/list_jobs`, `auth.verify_login/create_token/require_auth`, `config.get_dashboard_settings`.
- Produces: `dashboard.app.create_app() -> FastAPI` with `app.state.db` (open store connection) and `app.state.jobs_dir` (Path); `dashboard.api.router` (`/api` prefix); routes `POST /api/auth/login` → `{"access_token": str}` and `GET /api/jobs` → `list[dict]` (job dicts from `store.list_jobs`).
- Produces (test infra): conftest fixtures `app_env` (env + tmp data dir + `DASHBOARD_STUB_ANALYZE=1`), `client` (TestClient with lifespan), `auth_header` (dict with Bearer token); helper `make_batch_zip(path, names=..., manifest_rows=None, manifest=True)`; constant `PASSWORD = "Right#Pass1"`.

- [ ] **Step 1: Write conftest** — `tests/web/conftest.py`:

```python
import zipfile

import pytest
from fastapi.testclient import TestClient

from dashboard.config import clear_settings_cache
from dashboard.hash_password import make_hash

PASSWORD = "Right#Pass1"
PASSWORD_HASH = make_hash(PASSWORD)  # once per session — bcrypt is deliberately slow


@pytest.fixture()
def app_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHBOARD_ADMIN_USER", "autoace")
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_HASH", PASSWORD_HASH)
    monkeypatch.setenv("DASHBOARD_JWT_SECRET", "test-secret-0123456789abcdef-pad-to-32b")
    monkeypatch.setenv("DASHBOARD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DASHBOARD_STUB_ANALYZE", "1")
    clear_settings_cache()
    yield tmp_path
    clear_settings_cache()


@pytest.fixture()
def client(app_env):
    from dashboard.app import create_app

    with TestClient(create_app()) as c:
        yield c


@pytest.fixture()
def auth_header(client):
    r = client.post("/api/auth/login", json={"username": "autoace", "password": PASSWORD})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def make_batch_zip(path, names=("call_001.wav", "call_002.wav"), manifest_rows=None, manifest=True):
    """Tiny fake batch: content is never decoded before processing, so b'RIFF' suffices."""
    with zipfile.ZipFile(path, "w") as zf:
        for n in names:
            zf.writestr(n, b"RIFF0000WAVEfake")
        if manifest:
            rows = manifest_rows if manifest_rows is not None else [f"{n}," for n in names]
            zf.writestr("labels.csv", "name,result_json\n" + "\n".join(rows) + "\n")
    return path
```

- [ ] **Step 2: Write failing route tests** (append to `tests/web/test_auth.py`):

```python
def test_login_route_returns_token_and_gates_jobs(client):
    assert client.get("/api/jobs").status_code == 401
    r = client.post("/api/auth/login", json={"username": "autoace", "password": "Right#Pass1"})
    assert r.status_code == 200
    token = r.json()["access_token"]
    r2 = client.get("/api/jobs", headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 200
    assert r2.json() == []


def test_login_route_rejects_wrong_password(client):
    r = client.post("/api/auth/login", json={"username": "autoace", "password": "nope"})
    assert r.status_code == 401
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/web/test_auth.py -v`
Expected: the two new tests FAIL — `No module named 'dashboard.app'`; the 5 unit tests still pass.

- [ ] **Step 4: Implement `src/dashboard/api.py`** (login + list only; later tasks append):

```python
"""All /api routes."""

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from dashboard import store
from dashboard.auth import create_token, require_auth, verify_login
from dashboard.config import get_dashboard_settings

router = APIRouter(prefix="/api")


class LoginBody(BaseModel):
    username: str
    password: str


@router.post("/auth/login")
def login(body: LoginBody):
    settings = get_dashboard_settings()
    if not verify_login(body.username, body.password, settings):
        time.sleep(0.5)  # basic brute-force friction on failures only
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_token(settings)}


@router.get("/jobs")
def list_jobs(request: Request, user: str = Depends(require_auth)):
    return store.list_jobs(request.app.state.db)
```

- [ ] **Step 5: Implement `src/dashboard/app.py`**:

```python
"""FastAPI app factory. Static SPA serving is added in Task 12."""

from fastapi import FastAPI

from dashboard import api, store
from dashboard.config import get_dashboard_settings


def create_app() -> FastAPI:
    settings = get_dashboard_settings()
    app = FastAPI(
        title="AutoAce Evaluation Dashboard",
        docs_url=None, redoc_url=None, openapi_url=None,  # no public API browser
    )
    app.state.db = store.connect(settings.data_dir / "dashboard.db")
    app.state.jobs_dir = settings.data_dir / "jobs"
    app.include_router(api.router)
    return app
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/web/ -v`
Expected: all PASS (config 4, store 7, auth 7, zipsafe 5)

- [ ] **Step 7: Commit**

```bash
git status
git add src/dashboard/api.py src/dashboard/app.py tests/web/conftest.py tests/web/test_auth.py
git commit -m "feat(dashboard): app factory, login route, authenticated job list

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Upload + validate-before-processing route

**Files:**
- Modify: `src/dashboard/api.py` (append route + helpers)
- Test: `tests/web/test_upload.py`

**Interfaces:**
- Consumes: `zipsafe.extract_zip/UnsafeZipError`, `store.create_job/set_validation/get_job/delete_job`, `autoace_audio.batch.validate_batch`.
- Produces: `POST /api/jobs` (multipart field name `files`, one or many) → 201 + job dict in `awaiting_confirmation`. Writes `data/jobs/<id>/upload/<name>.zip` (zip path only), `extracted/`, `batch_root.txt` (absolute path of the resolved batch root — consumed by Task 7's runner), and `files.json` (ordered audio filenames from `validate_batch` — consumed by the UI's live queue). Failure modes: single non-zip → 400; hostile zip → 400; corrupt zip → 400; cumulative bytes > cap → 413. All failures remove the job row + directory.

- [ ] **Step 1: Write failing upload tests** — `tests/web/test_upload.py`:

```python
import io
import shutil

from tests.web.conftest import make_batch_zip


def _upload_zip(client, auth_header, zip_path, name="batch.zip"):
    return client.post(
        "/api/jobs",
        headers=auth_header,
        files=[("files", (name, zip_path.read_bytes(), "application/zip"))],
    )


def test_zip_upload_validates_and_awaits_confirmation(client, auth_header, tmp_path, app_env):
    z = make_batch_zip(tmp_path / "b.zip")
    r = _upload_zip(client, auth_header, z)
    assert r.status_code == 201
    job = r.json()
    assert job["status"] == "awaiting_confirmation"
    assert job["total"] == 2
    assert job["warnings"] == []
    assert job["original_name"] == "batch.zip"
    root = (app_env / "data" / "jobs" / job["id"] / "batch_root.txt").read_text()
    assert (app_env / "data" / "jobs" / job["id"] / "upload" / "batch.zip").exists()
    from pathlib import Path
    assert (Path(root) / "call_001.wav").exists()


def test_manifest_mismatch_surfaces_backend_warning(client, auth_header, tmp_path):
    z = make_batch_zip(
        tmp_path / "b.zip",
        manifest_rows=["call_001.wav,", "call_002.wav,", "ghost.wav,"],
    )
    job = _upload_zip(client, auth_header, z).json()
    assert "manifest row has no file on disk: ghost.wav" in job["warnings"]


def test_missing_manifest_is_warning_not_error(client, auth_header, tmp_path):
    z = make_batch_zip(tmp_path / "b.zip", manifest=False)
    job = _upload_zip(client, auth_header, z).json()
    assert job["status"] == "awaiting_confirmation"
    assert any("no CSV manifest found" in w for w in job["warnings"])


def test_folder_upload_streams_into_extracted(client, auth_header):
    r = client.post(
        "/api/jobs",
        headers=auth_header,
        files=[
            ("files", ("call_001.wav", b"RIFF0000WAVEfake", "audio/wav")),
            ("files", ("labels.csv", b"name,result_json\ncall_001.wav,\n", "text/csv")),
        ],
    )
    assert r.status_code == 201
    job = r.json()
    assert job["total"] == 1
    assert job["original_name"] == "folder upload (2 files)"


def test_single_non_zip_rejected(client, auth_header):
    r = client.post(
        "/api/jobs", headers=auth_header,
        files=[("files", ("call.wav", b"RIFF", "audio/wav"))],
    )
    assert r.status_code == 400


def test_hostile_zip_rejected_and_no_job_left(client, auth_header, tmp_path):
    import zipfile
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.txt", b"x")
    r = _upload_zip(client, auth_header, z, name="evil.zip")
    assert r.status_code == 400
    assert client.get("/api/jobs", headers=auth_header).json() == []


def test_oversized_upload_413(client, auth_header, tmp_path, monkeypatch):
    from dashboard.config import clear_settings_cache
    monkeypatch.setenv("DASHBOARD_MAX_UPLOAD_MB", "1")
    clear_settings_cache()
    big = tmp_path / "big.zip"
    import zipfile
    with zipfile.ZipFile(big, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("call_001.wav", b"\x00" * (2 * 1024 * 1024))
    r = _upload_zip(client, auth_header, big, name="big.zip")
    assert r.status_code == 413
    assert client.get("/api/jobs", headers=auth_header).json() == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/web/test_upload.py -v`
Expected: FAIL — 404/405 on `POST /api/jobs` (route doesn't exist yet)

- [ ] **Step 3: Implement the upload route** (append to `src/dashboard/api.py`):

```python
import json
import shutil
import uuid
import zipfile
from pathlib import Path

from fastapi import File, UploadFile

from autoace_audio.batch import validate_batch
from dashboard.zipsafe import UnsafeZipError, extract_zip


class _TooLarge(Exception):
    pass


def _stream_to(dst: Path, upload: UploadFile, used: list[int], cap_bytes: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            used[0] += len(chunk)
            if used[0] > cap_bytes:
                raise _TooLarge()
            out.write(chunk)


@router.post("/jobs", status_code=201)
def create_job_route(
    request: Request,
    files: list[UploadFile] = File(...),
    user: str = Depends(require_auth),
):
    settings = get_dashboard_settings()
    db = request.app.state.db
    cap_bytes = settings.max_upload_mb * 1024 * 1024
    is_zip = len(files) == 1 and (files[0].filename or "").lower().endswith(".zip")
    if len(files) == 1 and not is_zip:
        raise HTTPException(400, "Upload one ZIP archive, or select a folder (audio files + one CSV manifest).")

    job_id = uuid.uuid4().hex
    job_dir: Path = request.app.state.jobs_dir / job_id
    extracted = job_dir / "extracted"
    original_name = files[0].filename if is_zip else f"folder upload ({len(files)} files)"
    store.create_job(db, job_id, original_name)
    used = [0]
    try:
        if is_zip:
            upload_path = job_dir / "upload" / Path(files[0].filename).name
            _stream_to(upload_path, files[0], used, cap_bytes)
            root = extract_zip(upload_path, extracted)
        else:
            extracted.mkdir(parents=True, exist_ok=True)
            for f in files:
                # brief: audio files at the batch root — flatten any folder paths
                _stream_to(extracted / Path(f.filename).name, f, used, cap_bytes)
            root = extracted
        (job_dir / "batch_root.txt").write_text(str(root), encoding="utf-8")
        file_list, warnings = validate_batch(root)
        (job_dir / "files.json").write_text(
            json.dumps([p.name for p in file_list]), encoding="utf-8"
        )
        store.set_validation(db, job_id, total=len(file_list), warnings=warnings)
        return store.get_job(db, job_id)
    except _TooLarge:
        _discard(db, job_id, job_dir)
        raise HTTPException(413, f"Upload exceeds the {settings.max_upload_mb} MB limit.") from None
    except UnsafeZipError as e:
        _discard(db, job_id, job_dir)
        raise HTTPException(400, f"Rejected ZIP: {e}") from None
    except zipfile.BadZipFile:
        _discard(db, job_id, job_dir)
        raise HTTPException(400, "The uploaded file is not a valid ZIP archive.") from None


def _discard(db, job_id: str, job_dir: Path) -> None:
    store.delete_job(db, job_id)
    shutil.rmtree(job_dir, ignore_errors=True)
```

- [ ] **Step 4: Run upload tests**

Run: `.venv/bin/pytest tests/web/test_upload.py -v`
Expected: 7 PASS

- [ ] **Step 5: Run whole web suite + core suite**

Run: `.venv/bin/pytest tests/web/ -q && make test`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git status
git add src/dashboard/api.py tests/web/test_upload.py
git commit -m "feat(dashboard): streaming batch upload with validate-before-processing

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Runner — worker process, dispatcher, orphan sweep, start route

**Files:**
- Create: `src/dashboard/runner.py`
- Modify: `src/dashboard/app.py` (lifespan), `src/dashboard/api.py` (append `start` + `get` routes)
- Test: `tests/web/test_runner.py`

**Interfaces:**
- Consumes: `store.*`, `autoace_audio.batch.run_batch` (and its `analyze_fn` injection point), `batch_root.txt` from Task 6.
- Produces: `runner.stub_analyze(path, tone_arm=None)`; `runner.worker_main(job_id, db_path: str, batch_root: str, out_dir: str, stub: bool)`; `runner.dispatch_once(db, db_path: Path, jobs_dir: Path, stub: bool) -> bool`; `runner.sweep_orphans(db)`; routes `POST /api/jobs/{id}/start` (409 unless `awaiting_confirmation`) and `GET /api/jobs/{id}` (job dict + `files: list[str]` from `files.json`).
- Note: the worker imports `autoace_audio.batch` → `pipeline` → torch at process start even when stubbed (~few seconds). No models load with the stub. Lifecycle tests poll with generous timeouts; expect `tests/web/` to take ~30–60 s wall total.

- [ ] **Step 1: Write failing tests** — `tests/web/test_runner.py`:

```python
import time

import pytest

from dashboard import runner, store
from tests.web.conftest import make_batch_zip


def test_stub_analyze_returns_valid_result_and_fails_on_bad(tmp_path):
    out = runner.stub_analyze(tmp_path / "call_001.wav")
    d = out.result.model_dump(mode="json")
    assert d["emotional_tone"] == "neutral" and d["confidence"] == 0.9
    with pytest.raises(ValueError):
        runner.stub_analyze(tmp_path / "call_bad.wav")


def test_sweep_orphans_marks_stale_jobs(tmp_path):
    db = store.connect(tmp_path / "t.db")
    for jid, status in [("r1", "running"), ("q1", "queued"), ("v1", "validating"), ("c1", "completed")]:
        store.create_job(db, jid, "b.zip")
        store.set_status(db, jid, status)
    runner.sweep_orphans(db)
    assert store.get_job(db, "r1")["status"] == "interrupted"
    assert store.get_job(db, "q1")["status"] == "interrupted"
    assert store.get_job(db, "v1")["status"] == "failed"
    assert store.get_job(db, "c1")["status"] == "completed"
    db.close()


class _FakeProc:
    pid = 9999
    def __init__(self, *a, **k): self._alive = True
    def start(self): pass
    def is_alive(self): return self._alive
    @property
    def exitcode(self): return None


def test_dispatch_once_runs_one_job_at_a_time(tmp_path, monkeypatch):
    class FakeCtx:
        Process = staticmethod(lambda **kw: _FakeProc())
    monkeypatch.setattr(runner, "_ctx", FakeCtx())
    runner._processes.clear()
    db = store.connect(tmp_path / "t.db")
    for jid in ("older", "newer"):
        store.create_job(db, jid, "b.zip")
        (tmp_path / jid).mkdir()
        (tmp_path / jid / "batch_root.txt").write_text(str(tmp_path / jid))
        store.set_status(db, jid, "queued")
        time.sleep(1.1)  # created_at has second resolution; keep ordering unambiguous
    assert runner.dispatch_once(db, tmp_path / "t.db", tmp_path, stub=True) is True
    assert store.get_job(db, "older")["status"] == "running"     # oldest queued first
    assert runner.dispatch_once(db, tmp_path / "t.db", tmp_path, stub=True) is False
    assert store.get_job(db, "newer")["status"] == "queued"      # waits its turn
    runner._processes.clear()
    db.close()


def _wait_for(client, auth_header, job_id, status, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}", headers=auth_header).json()
        if job["status"] == status:
            return job
        assert job["status"] not in ("failed",), f"unexpected failure: {job['error']}"
        time.sleep(0.5)
    raise AssertionError(f"job never reached {status}")


def test_full_lifecycle_with_stub(client, auth_header, tmp_path):
    z = make_batch_zip(tmp_path / "b.zip")
    job = client.post("/api/jobs", headers=auth_header,
                      files=[("files", ("b.zip", z.read_bytes(), "application/zip"))]).json()
    r = client.post(f"/api/jobs/{job['id']}/start", headers=auth_header)
    assert r.status_code == 200 and r.json()["status"] == "queued"
    done = _wait_for(client, auth_header, job["id"], "completed")
    assert done["done"] == 2 and done["total"] == 2
    assert done["results_count"] == 2 and done["errors_count"] == 0
    assert done["started_at"] and done["finished_at"]


def test_per_file_failure_isolation_surfaces_in_counts(client, auth_header, tmp_path):
    z = make_batch_zip(tmp_path / "b.zip", names=("call_001.wav", "call_bad.wav"))
    job = client.post("/api/jobs", headers=auth_header,
                      files=[("files", ("b.zip", z.read_bytes(), "application/zip"))]).json()
    client.post(f"/api/jobs/{job['id']}/start", headers=auth_header)
    done = _wait_for(client, auth_header, job["id"], "completed")
    assert done["results_count"] == 1 and done["errors_count"] == 1


def test_start_requires_awaiting_confirmation(client, auth_header, tmp_path):
    z = make_batch_zip(tmp_path / "b.zip")
    job = client.post("/api/jobs", headers=auth_header,
                      files=[("files", ("b.zip", z.read_bytes(), "application/zip"))]).json()
    client.post(f"/api/jobs/{job['id']}/start", headers=auth_header)
    r = client.post(f"/api/jobs/{job['id']}/start", headers=auth_header)
    assert r.status_code == 409
    assert client.post("/api/jobs/nope/start", headers=auth_header).status_code == 404
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/web/test_runner.py -v`
Expected: FAIL — `No module named 'dashboard.runner'`

- [ ] **Step 3: Implement `src/dashboard/runner.py`**:

```python
"""Job execution: one spawned worker process at a time runs run_batch.
The API process owns queueing (dispatch_once); the worker owns progress
writes and its own terminal transition."""

import multiprocessing as mp
from pathlib import Path

from dashboard import store

_ctx = mp.get_context("spawn")
_processes: dict[str, object] = {}  # job_id -> live Process handle (this server process only)


def stub_analyze(path, tone_arm=None):
    """DASHBOARD_STUB_ANALYZE=1 only: canned result, no models/keys/network.
    Raises for files with 'bad' in the name to exercise failure isolation."""
    from autoace_audio.pipeline import PipelineOutput
    from autoace_audio.schema import AnalysisResult

    p = Path(path)
    if "bad" in p.name.lower():
        raise ValueError("stub: simulated per-file analysis failure")
    return PipelineOutput(
        result=AnalysisResult(
            emotional_tone="neutral",
            emotional_intensity="low",
            background_noise_present=False,
            background_noise_type="",
            background_noise_severity="none",
            audio_quality="clear",
            speaker_overlap_present=False,
            long_silence_present=False,
            confidence=0.9,
        ),
        diagnostics={"stub": True},
    )


def worker_main(job_id: str, db_path: str, batch_root: str, out_dir: str, stub: bool) -> None:
    """Spawned-process entry point."""
    db = store.connect(Path(db_path))
    try:
        from autoace_audio.batch import run_batch  # imports torch — inside the worker only

        def progress(done: int, total: int, name: str) -> None:
            store.update_progress(db, job_id, done=done, current_file=name)

        kwargs = {"analyze_fn": stub_analyze} if stub else {}
        report = run_batch(Path(batch_root), Path(out_dir), progress_cb=progress, **kwargs)
        already = set(store.get_job(db, job_id)["warnings"])
        extra = [w for w in report.warnings if w not in already]  # validation warnings repeat
        store.finish(db, job_id, results_count=len(report.results),
                     errors_count=len(report.errors), extra_warnings=extra)
    except Exception as e:  # noqa: BLE001 — a worker must always leave a terminal status
        store.set_status(db, job_id, "failed", error=f"{type(e).__name__}: {e}")
    finally:
        db.close()


def sweep_orphans(db) -> None:
    """Startup: previous server process left these behind."""
    for job in store.list_jobs(db):
        if job["status"] in ("running", "queued"):
            store.set_status(db, job["id"], "interrupted", error="interrupted by server restart")
        elif job["status"] == "validating":
            store.set_status(db, job["id"], "failed", error="interrupted during validation")


def dispatch_once(db, db_path: Path, jobs_dir: Path, stub: bool) -> bool:
    """Reap finished workers; start the oldest queued job if nothing runs. True if started one."""
    for job in store.list_jobs(db):
        if job["status"] != "running":
            continue
        proc = _processes.get(job["id"])
        if proc is None:  # running row without a handle → predates a restart
            store.set_status(db, job["id"], "interrupted", error="interrupted by server restart")
            continue
        if proc.is_alive():
            return False  # a job is genuinely running — nothing else may start
        _processes.pop(job["id"], None)
        if store.get_job(db, job["id"])["status"] == "running":  # died without terminal write
            store.set_status(db, job["id"], "failed",
                             error=f"worker process died (exit code {proc.exitcode})")
    queued = [j for j in store.list_jobs(db) if j["status"] == "queued"]
    if not queued:
        return False
    job = queued[-1]  # list_jobs is newest-first → last is the oldest queued
    job_dir = jobs_dir / job["id"]
    batch_root = (job_dir / "batch_root.txt").read_text(encoding="utf-8").strip()
    proc = _ctx.Process(
        target=worker_main,
        kwargs=dict(job_id=job["id"], db_path=str(db_path), batch_root=batch_root,
                    out_dir=str(job_dir / "out"), stub=stub),
        daemon=True,
    )
    store.set_status(db, job["id"], "running")
    proc.start()
    _processes[job["id"]] = proc
    store.set_worker_pid(db, job["id"], proc.pid)
    return True
```

- [ ] **Step 4: Wire the dispatcher into the app** — replace `src/dashboard/app.py` with:

```python
"""FastAPI app factory + background dispatcher. Static SPA serving arrives in Task 12."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from dashboard import api, runner, store
from dashboard.config import get_dashboard_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings = get_dashboard_settings()
    runner.sweep_orphans(app.state.db)
    stop = asyncio.Event()

    async def _dispatcher():
        db_path = settings.data_dir / "dashboard.db"
        while not stop.is_set():
            try:
                runner.dispatch_once(app.state.db, db_path, app.state.jobs_dir,
                                     settings.stub_analyze)
            except Exception:  # noqa: BLE001 — the dispatcher must never die
                logger.exception("dispatcher tick failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    task = asyncio.create_task(_dispatcher())
    yield
    stop.set()
    await task


def create_app() -> FastAPI:
    settings = get_dashboard_settings()
    app = FastAPI(
        title="AutoAce Evaluation Dashboard",
        docs_url=None, redoc_url=None, openapi_url=None,
        lifespan=_lifespan,
    )
    app.state.db = store.connect(settings.data_dir / "dashboard.db")
    app.state.jobs_dir = settings.data_dir / "jobs"
    app.include_router(api.router)
    return app
```

- [ ] **Step 5: Add start/get routes** (append to `src/dashboard/api.py`):

```python
@router.get("/jobs/{job_id}")
def get_job_route(job_id: str, request: Request, user: str = Depends(require_auth)):
    job = store.get_job(request.app.state.db, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    files_path = request.app.state.jobs_dir / job_id / "files.json"
    job["files"] = json.loads(files_path.read_text(encoding="utf-8")) if files_path.exists() else []
    return job


@router.post("/jobs/{job_id}/start")
def start_job(job_id: str, request: Request, user: str = Depends(require_auth)):
    db = request.app.state.db
    job = store.get_job(db, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] != "awaiting_confirmation":
        raise HTTPException(409, f"Job is {job['status']}; only a validated batch can start.")
    store.set_status(db, job_id, "queued")
    return store.get_job(db, job_id)
```

- [ ] **Step 6: Run runner tests, then everything**

Run: `.venv/bin/pytest tests/web/test_runner.py -v` then `.venv/bin/pytest tests/web/ -q && make test`
Expected: all PASS (lifecycle tests take tens of seconds — torch imports in the worker; no models download)

- [ ] **Step 7: Commit**

```bash
git status
git add src/dashboard/runner.py src/dashboard/app.py src/dashboard/api.py tests/web/test_runner.py
git commit -m "feat(dashboard): worker-process runner, one-at-a-time dispatcher, start route

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Job management routes — rerun, delete, results, errors, downloads

**Files:**
- Modify: `src/dashboard/store.py` (add `requeue`), `src/dashboard/api.py` (append routes)
- Test: `tests/web/test_job_routes.py`, one test appended to `tests/web/test_store.py`

**Interfaces:**
- Consumes: job dirs (`out/results.json`, `out/errors.csv`) written by `run_batch`.
- Produces: `store.requeue(db, job_id)` (status `queued`; resets `done`, `current_file`, `error`, `started_at`, `finished_at`, `results_count`, `errors_count`); routes `POST /api/jobs/{id}/rerun` (409 unless `failed`/`interrupted`), `DELETE /api/jobs/{id}` → 204 (409 while `queued`/`running`), `GET /api/jobs/{id}/results` → `[{name, ...9 fields}]`, `GET /api/jobs/{id}/errors` → `[{name, error}]`, `GET /api/jobs/{id}/download/{artifact}` for `results.csv | results.json | errors.csv`.

- [ ] **Step 1: Failing store test** (append to `tests/web/test_store.py`):

```python
def test_requeue_resets_run_state(db):
    store.create_job(db, "j1", "b.zip")
    store.set_validation(db, "j1", total=2, warnings=["w"])
    store.set_status(db, "j1", "running")
    store.update_progress(db, "j1", done=2, current_file="x.wav")
    store.set_status(db, "j1", "failed", error="boom")
    store.requeue(db, "j1")
    job = store.get_job(db, "j1")
    assert job["status"] == "queued"
    assert job["done"] == 0 and job["current_file"] is None
    assert job["error"] is None and job["finished_at"] is None and job["started_at"] is None
    assert job["results_count"] is None and job["errors_count"] is None
    assert job["total"] == 2 and job["warnings"] == ["w"]   # validation facts survive
```

Run: `.venv/bin/pytest tests/web/test_store.py::test_requeue_resets_run_state -v` → FAIL (no `requeue`).

- [ ] **Step 2: Implement `store.requeue`** (append to `src/dashboard/store.py`):

```python
def requeue(db: sqlite3.Connection, job_id: str) -> None:
    db.execute(
        "UPDATE jobs SET status = 'queued', done = 0, current_file = NULL, error = NULL, "
        "started_at = NULL, finished_at = NULL, results_count = NULL, errors_count = NULL "
        "WHERE id = ?",
        (job_id,),
    )
```

Run the test again → PASS.

- [ ] **Step 3: Write failing route tests** — `tests/web/test_job_routes.py`:

```python
import json

from dashboard import store
from tests.web.conftest import make_batch_zip

RESULT_FIELDS = [
    "emotional_tone", "emotional_intensity", "background_noise_present",
    "background_noise_type", "background_noise_severity", "audio_quality",
    "speaker_overlap_present", "long_silence_present", "confidence",
]


def _job_with_artifacts(client, auth_header, app_env, tmp_path):
    """Upload a real batch, then hand-write out/ + completed status (no worker spawn)."""
    z = make_batch_zip(tmp_path / "b.zip")
    job = client.post("/api/jobs", headers=auth_header,
                      files=[("files", ("b.zip", z.read_bytes(), "application/zip"))]).json()
    out = app_env / "data" / "jobs" / job["id"] / "out"
    out.mkdir(parents=True)
    row = {f: ("neutral" if f == "emotional_tone" else "low" if f == "emotional_intensity"
               else False if f.endswith("_present") else "" if f == "background_noise_type"
               else "none" if f == "background_noise_severity" else "clear" if f == "audio_quality"
               else 0.9) for f in RESULT_FIELDS}
    (out / "results.json").write_text(json.dumps({"call_001.wav": row}), encoding="utf-8")
    (out / "results.csv").write_text(
        'name,result_json\ncall_001.wav,"{""emotional_tone"": ""neutral""}"\n', encoding="utf-8")
    (out / "errors.csv").write_text("name,error\ncall_002.wav,decode: fake\n", encoding="utf-8")
    db = store.connect(app_env / "data" / "dashboard.db")
    store.set_status(db, job["id"], "completed")
    db.close()
    return job


def test_results_and_errors_endpoints(client, auth_header, app_env, tmp_path):
    job = _job_with_artifacts(client, auth_header, app_env, tmp_path)
    rows = client.get(f"/api/jobs/{job['id']}/results", headers=auth_header).json()
    assert rows[0]["name"] == "call_001.wav"
    assert all(f in rows[0] for f in RESULT_FIELDS)
    errs = client.get(f"/api/jobs/{job['id']}/errors", headers=auth_header).json()
    assert errs == [{"name": "call_002.wav", "error": "decode: fake"}]


def test_results_409_until_available(client, auth_header, tmp_path):
    z = make_batch_zip(tmp_path / "b.zip")
    job = client.post("/api/jobs", headers=auth_header,
                      files=[("files", ("b.zip", z.read_bytes(), "application/zip"))]).json()
    assert client.get(f"/api/jobs/{job['id']}/results", headers=auth_header).status_code == 409


def test_download_serves_artifacts_verbatim(client, auth_header, app_env, tmp_path):
    job = _job_with_artifacts(client, auth_header, app_env, tmp_path)
    disk = (app_env / "data" / "jobs" / job["id"] / "out" / "results.csv").read_bytes()
    r = client.get(f"/api/jobs/{job['id']}/download/results.csv", headers=auth_header)
    assert r.status_code == 200 and r.content == disk
    assert client.get(f"/api/jobs/{job['id']}/download/evil.txt", headers=auth_header).status_code == 404


def test_rerun_only_from_failed_or_interrupted(client, auth_header, app_env, tmp_path):
    job = _job_with_artifacts(client, auth_header, app_env, tmp_path)  # completed
    assert client.post(f"/api/jobs/{job['id']}/rerun", headers=auth_header).status_code == 409
    db = store.connect(app_env / "data" / "dashboard.db")
    store.set_status(db, job["id"], "interrupted", error="interrupted by server restart")
    db.close()
    r = client.post(f"/api/jobs/{job['id']}/rerun", headers=auth_header)
    assert r.status_code == 200 and r.json()["status"] in ("queued", "running", "completed")


def test_delete_blocked_while_active_then_removes_everything(client, auth_header, app_env, tmp_path):
    job = _job_with_artifacts(client, auth_header, app_env, tmp_path)
    db = store.connect(app_env / "data" / "dashboard.db")
    store.set_status(db, job["id"], "running")
    db.close()
    assert client.delete(f"/api/jobs/{job['id']}", headers=auth_header).status_code == 409
    db = store.connect(app_env / "data" / "dashboard.db")
    store.set_status(db, job["id"], "completed")
    db.close()
    assert client.delete(f"/api/jobs/{job['id']}", headers=auth_header).status_code == 204
    assert client.get(f"/api/jobs/{job['id']}", headers=auth_header).status_code == 404
    assert not (app_env / "data" / "jobs" / job["id"]).exists()
```

Run: `.venv/bin/pytest tests/web/test_job_routes.py -v` → FAIL (routes missing).

- [ ] **Step 4: Implement the routes** (append to `src/dashboard/api.py`):

```python
import csv  # json/shutil/Path already imported by the Task 6 block above

from fastapi.responses import FileResponse

_ARTIFACTS = {"results.csv": "text/csv", "results.json": "application/json", "errors.csv": "text/csv"}


def _job_or_404(db, job_id: str) -> dict:
    job = store.get_job(db, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


def _out_path(request: Request, job_id: str, artifact: str) -> Path:
    path = request.app.state.jobs_dir / job_id / "out" / artifact
    if not path.exists():
        raise HTTPException(409, "Not available yet — the batch has not finished processing.")
    return path


@router.post("/jobs/{job_id}/rerun")
def rerun_job(job_id: str, request: Request, user: str = Depends(require_auth)):
    db = request.app.state.db
    job = _job_or_404(db, job_id)
    if job["status"] not in ("failed", "interrupted"):
        raise HTTPException(409, f"Job is {job['status']}; only failed or interrupted jobs re-run.")
    store.requeue(db, job_id)
    return store.get_job(db, job_id)


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job_route(job_id: str, request: Request, user: str = Depends(require_auth)):
    db = request.app.state.db
    job = _job_or_404(db, job_id)
    if job["status"] in ("queued", "running"):
        raise HTTPException(409, "Job is active; wait for it to finish before deleting.")
    store.delete_job(db, job_id)
    shutil.rmtree(request.app.state.jobs_dir / job_id, ignore_errors=True)


@router.get("/jobs/{job_id}/results")
def job_results(job_id: str, request: Request, user: str = Depends(require_auth)):
    _job_or_404(request.app.state.db, job_id)
    data = json.loads(_out_path(request, job_id, "results.json").read_text(encoding="utf-8"))
    return [{"name": name, **fields} for name, fields in data.items()]


@router.get("/jobs/{job_id}/errors")
def job_errors(job_id: str, request: Request, user: str = Depends(require_auth)):
    _job_or_404(request.app.state.db, job_id)
    with open(_out_path(request, job_id, "errors.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@router.get("/jobs/{job_id}/download/{artifact}")
def download_artifact(job_id: str, artifact: str, request: Request,
                      user: str = Depends(require_auth)):
    if artifact not in _ARTIFACTS:
        raise HTTPException(404, "Unknown artifact")
    _job_or_404(request.app.state.db, job_id)
    return FileResponse(_out_path(request, job_id, artifact),
                        media_type=_ARTIFACTS[artifact], filename=artifact)
```

- [ ] **Step 5: Run everything**

Run: `.venv/bin/pytest tests/web/ -q && make test`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git status
git add src/dashboard/store.py src/dashboard/api.py tests/web/test_store.py tests/web/test_job_routes.py
git commit -m "feat(dashboard): rerun/delete/results/errors/download routes

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Webapp scaffold — Vite + React + Tailwind brand tokens, API client, login

**Files:**
- Create: `webapp/package.json`, `webapp/vite.config.js`, `webapp/tailwind.config.js`, `webapp/postcss.config.js`, `webapp/index.html`, `webapp/.gitignore`, `webapp/src/main.jsx`, `webapp/src/index.css`, `webapp/src/api.js`, `webapp/src/App.jsx`, `webapp/src/pages/LoginPage.jsx`

**Interfaces:**
- Produces: `src/api.js` exports `getToken/setToken/clearToken`, `login(username, password)`, `listJobs()`, `getJob(id)`, `createJob(formData, onUploadProgress)`, `startJob(id)`, `rerunJob(id)`, `deleteJob(id)`, `getResults(id)`, `getErrors(id)`, `downloadArtifact(id, name)` — all returning response data promises; 401s (except login) clear the token and redirect to `/login`.
- Produces: Tailwind tokens `wash`, `ink`, `body`, `navy`, `accent`, `accent-bright`; font families `display` (Sora) and `body`; route shell with auth guard.
- Consumes: the `/api` routes from Tasks 5–8 (via Vite dev proxy to `127.0.0.1:8000`).

- [ ] **Step 1: Write the scaffold files**

`webapp/package.json`:

```json
{
  "name": "autoace-dashboard-webapp",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run"
  },
  "dependencies": {
    "@fontsource/sora": "^5.1.0",
    "axios": "^1.7.7",
    "lucide-react": "^0.460.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-hot-toast": "^2.4.1",
    "react-router-dom": "^6.28.0"
  },
  "devDependencies": {
    "@playwright/test": "^1.48.2",
    "@vitejs/plugin-react": "^4.3.3",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.47",
    "tailwindcss": "^3.4.14",
    "vite": "^5.4.10",
    "vitest": "^2.1.4"
  }
}
```

`webapp/.gitignore`:

```
node_modules/
dist/
test-results/
playwright-report/
```

`webapp/vite.config.js`:

```js
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
})
```

`webapp/tailwind.config.js` (brand tokens from the spec's visual-design section):

```js
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        wash: '#EFF4FA',
        ink: '#0F172A',
        body: '#475569',
        navy: '#020817',
        accent: { DEFAULT: '#2563EB', bright: '#3B82F6' },
      },
      fontFamily: {
        display: ['Sora', 'system-ui', 'sans-serif'],
        body: ['system-ui', '-apple-system', '"Segoe UI"', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
```

`webapp/postcss.config.js`:

```js
export default { plugins: { tailwindcss: {}, autoprefixer: {} } }
```

`webapp/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AutoAce Evaluation Dashboard</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

`webapp/src/index.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  @apply bg-wash text-body font-body antialiased;
}

h1, h2, h3 {
  @apply font-display text-ink;
}

:focus-visible {
  @apply outline-none ring-2 ring-accent ring-offset-2 ring-offset-wash;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

`webapp/src/main.jsx`:

```jsx
import '@fontsource/sora/600.css'
import '@fontsource/sora/700.css'
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

- [ ] **Step 2: Write the API client** — `webapp/src/api.js`:

```js
import axios from 'axios'

const TOKEN_KEY = 'autoace_dashboard_token'
export const getToken = () => localStorage.getItem(TOKEN_KEY)
export const setToken = (t) => localStorage.setItem(TOKEN_KEY, t)
export const clearToken = () => localStorage.removeItem(TOKEN_KEY)

const client = axios.create({ baseURL: '' })

client.interceptors.request.use((config) => {
  const t = getToken()
  if (t) config.headers.Authorization = `Bearer ${t}`
  return config
})

client.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401 && !err.config?.url?.includes('/auth/login')) {
      clearToken()
      window.location.assign('/login')
    }
    return Promise.reject(err)
  },
)

export const login = (username, password) =>
  client.post('/api/auth/login', { username, password }).then((r) => r.data)
export const listJobs = () => client.get('/api/jobs').then((r) => r.data)
export const getJob = (id) => client.get(`/api/jobs/${id}`).then((r) => r.data)
export const createJob = (formData, onUploadProgress) =>
  client.post('/api/jobs', formData, { onUploadProgress }).then((r) => r.data)
export const startJob = (id) => client.post(`/api/jobs/${id}/start`).then((r) => r.data)
export const rerunJob = (id) => client.post(`/api/jobs/${id}/rerun`).then((r) => r.data)
export const deleteJob = (id) => client.delete(`/api/jobs/${id}`)
export const getResults = (id) => client.get(`/api/jobs/${id}/results`).then((r) => r.data)
export const getErrors = (id) => client.get(`/api/jobs/${id}/errors`).then((r) => r.data)

export const downloadArtifact = async (id, name) => {
  const r = await client.get(`/api/jobs/${id}/download/${name}`, { responseType: 'blob' })
  const url = URL.createObjectURL(r.data)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  URL.revokeObjectURL(url)
}
```

- [ ] **Step 3: Route shell with auth guard** — `webapp/src/App.jsx`:

```jsx
import { Toaster } from 'react-hot-toast'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { getToken } from './api'
import JobPage from './pages/JobPage'
import JobsPage from './pages/JobsPage'
import LoginPage from './pages/LoginPage'

function RequireAuth({ children }) {
  return getToken() ? children : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <BrowserRouter>
      <Toaster position="top-right" />
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/" element={<RequireAuth><JobsPage /></RequireAuth>} />
        <Route path="/jobs/:id" element={<RequireAuth><JobPage /></RequireAuth>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
```

(`JobsPage`/`JobPage` don't exist yet — create placeholder files so the build passes; Tasks 10–11 replace them:)

`webapp/src/pages/JobsPage.jsx` (placeholder): `export default function JobsPage() { return null }`
`webapp/src/pages/JobPage.jsx` (placeholder): `export default function JobPage() { return null }`

- [ ] **Step 4: Login page** — `webapp/src/pages/LoginPage.jsx`:

```jsx
import { useState } from 'react'
import toast from 'react-hot-toast'
import { useNavigate } from 'react-router-dom'
import { login, setToken } from '../api'

export default function LoginPage() {
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    try {
      const { access_token } = await login(username, password)
      setToken(access_token)
      navigate('/')
    } catch (err) {
      toast.error(err.response?.status === 401
        ? 'Wrong username or password'
        : 'Could not reach the server')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-sm rounded-xl border border-gray-200 bg-white p-8 shadow-sm">
        <h1 className="text-2xl font-bold">
          <span className="text-accent">AutoAce</span> Evaluation
        </h1>
        <p className="mt-1 text-sm">Sign in to upload and review analysis batches.</p>
        <form onSubmit={submit} className="mt-6 space-y-4">
          <label className="block text-sm">
            Username
            <input value={username} onChange={(e) => setUsername(e.target.value)} required
              autoComplete="username"
              className="mt-1 w-full rounded-lg border border-gray-200 px-3 py-2 text-ink" />
          </label>
          <label className="block text-sm">
            Password
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)}
              required autoComplete="current-password"
              className="mt-1 w-full rounded-lg border border-gray-200 px-3 py-2 text-ink" />
          </label>
          <button type="submit" disabled={busy}
            className="w-full rounded-lg bg-navy px-4 py-2.5 font-medium text-white disabled:opacity-60">
            {busy ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
      </div>
    </main>
  )
}
```

- [ ] **Step 5: Install and build**

Run: `cd webapp && npm install && npm run build`
Expected: `dist/` produced with no errors (placeholders render nothing yet). `git status` must show `webapp/node_modules` and `webapp/dist` ignored (via `webapp/.gitignore`).

- [ ] **Step 6: Manual smoke (dev)**

Terminal A: set env (`export $(grep -v '^#' .env | xargs)` or a real `.env`) plus `DASHBOARD_STUB_ANALYZE=1`, then `.venv/bin/uvicorn --factory dashboard.app:create_app --port 8000`.
Terminal B: `cd webapp && npm run dev` → open the printed URL, log in with the dev credentials, expect redirect to a blank jobs page (placeholder) with no console errors.

- [ ] **Step 7: Commit**

```bash
git status
git add webapp
git commit -m "feat(webapp): Vite+React+Tailwind scaffold, brand tokens, API client, login

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Status helpers (vitest) + jobs list page with upload

**Files:**
- Create: `webapp/src/lib/status.js`, `webapp/src/lib/status.test.js`, `webapp/src/components/StatusChip.jsx`, `webapp/src/components/UploadCard.jsx`
- Modify: `webapp/src/pages/JobsPage.jsx` (replace placeholder)

**Interfaces:**
- Produces: `STATUS_META` (label + Tailwind chip classes for all 7 states), `isActive(status)`, `shortId(id)`, `fmtTime(iso)`, `buildQueueRows(files, done, total) -> [{name, state}]` with `state ∈ completed|analyzing|pending` (consumed by Task 11);
  `<StatusChip status />`; `<UploadCard onCreated(job) />` posting `FormData` with field name `files`.
- Consumes: `api.listJobs/createJob`.

- [ ] **Step 1: Write failing helper tests** — `webapp/src/lib/status.test.js`:

```js
import { describe, expect, it } from 'vitest'
import { STATUS_META, buildQueueRows, isActive, shortId } from './status'

const ALL = ['validating', 'awaiting_confirmation', 'queued', 'running',
  'completed', 'failed', 'interrupted']

describe('status helpers', () => {
  it('covers every job state with label and chip classes', () => {
    for (const s of ALL) {
      expect(STATUS_META[s].label).toBeTruthy()
      expect(STATUS_META[s].chip).toContain('bg-')
    }
  })
  it('isActive only for in-flight states', () => {
    expect(ALL.filter(isActive)).toEqual(['validating', 'queued', 'running'])
  })
  it('shortId takes 8 chars', () => {
    expect(shortId('abcdef0123456789')).toBe('abcdef01')
  })
  it('buildQueueRows marks completed/analyzing/pending', () => {
    const files = ['a.wav', 'b.wav', 'c.wav']
    expect(buildQueueRows(files, 0, 3).map((r) => r.state))
      .toEqual(['analyzing', 'pending', 'pending'])
    expect(buildQueueRows(files, 1, 3).map((r) => r.state))
      .toEqual(['completed', 'analyzing', 'pending'])
    expect(buildQueueRows(files, 3, 3).map((r) => r.state))
      .toEqual(['completed', 'completed', 'completed'])
  })
})
```

Run: `cd webapp && npx vitest run` → FAIL (module missing).

- [ ] **Step 2: Implement** — `webapp/src/lib/status.js`:

```js
export const STATUS_META = {
  validating: { label: 'Validating', chip: 'bg-blue-100 text-blue-700' },
  awaiting_confirmation: { label: 'Awaiting confirmation', chip: 'bg-amber-100 text-amber-700' },
  queued: { label: 'Queued', chip: 'bg-blue-100 text-blue-700' },
  running: { label: 'Analyzing', chip: 'bg-blue-100 text-blue-700' },
  completed: { label: 'Completed', chip: 'bg-green-100 text-green-700' },
  failed: { label: 'Failed', chip: 'bg-red-100 text-red-700' },
  interrupted: { label: 'Interrupted', chip: 'bg-red-100 text-red-700' },
}

export const isActive = (s) => ['validating', 'queued', 'running'].includes(s)

export const shortId = (id) => (id || '').slice(0, 8)

export const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString() : '—')

export const buildQueueRows = (files, done, total) =>
  files.map((name, i) => ({
    name,
    state: i < done ? 'completed' : i === done && done < total ? 'analyzing' : 'pending',
  }))
```

Run: `npx vitest run` → 4 PASS.

- [ ] **Step 3: StatusChip** — `webapp/src/components/StatusChip.jsx`:

```jsx
import { STATUS_META } from '../lib/status'

export default function StatusChip({ status }) {
  const meta = STATUS_META[status] ?? { label: status, chip: 'bg-gray-100 text-gray-700' }
  return (
    <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${meta.chip}`}>
      {meta.label}
    </span>
  )
}
```

- [ ] **Step 4: UploadCard** — `webapp/src/components/UploadCard.jsx`:

```jsx
import { useRef, useState } from 'react'
import toast from 'react-hot-toast'
import { createJob } from '../api'

export default function UploadCard({ onCreated }) {
  const zipRef = useRef()
  const folderRef = useRef()
  const [progress, setProgress] = useState(null) // null | 0..100

  const send = async (files) => {
    if (!files.length) return
    const form = new FormData()
    for (const f of files) form.append('files', f, f.name)
    setProgress(0)
    try {
      const job = await createJob(form, (e) =>
        setProgress(e.total ? Math.round((e.loaded / e.total) * 100) : 0))
      onCreated(job)
    } catch (err) {
      toast.error(err.response?.data?.detail ?? 'Upload failed')
    } finally {
      setProgress(null)
    }
  }

  return (
    <section
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => { e.preventDefault(); send([...e.dataTransfer.files]) }}
      className="rounded-xl border-2 border-dashed border-gray-200 bg-white p-6 text-center"
    >
      <h2 className="text-lg font-semibold">Upload a batch</h2>
      <p className="mx-auto mt-1 max-w-md text-sm">
        Drop a ZIP here, or choose a ZIP / folder. A batch is audio files plus one CSV manifest
        (<code className="font-mono">name,result_json</code>) at the folder root.
      </p>
      {progress === null ? (
        <div className="mt-4 flex justify-center gap-3">
          <button onClick={() => zipRef.current.click()}
            className="rounded-lg bg-navy px-4 py-2 text-sm font-medium text-white">
            Choose ZIP
          </button>
          <button onClick={() => folderRef.current.click()}
            className="rounded-lg border border-gray-200 px-4 py-2 text-sm font-medium text-ink">
            Choose folder
          </button>
          <input ref={zipRef} type="file" accept=".zip" hidden
            onChange={(e) => send([...e.target.files])} />
          <input ref={folderRef} type="file" webkitdirectory="" hidden
            onChange={(e) => send([...e.target.files])} />
        </div>
      ) : (
        <div className="mx-auto mt-4 max-w-md">
          <div className="h-2 overflow-hidden rounded-full bg-gray-200">
            <div className="h-full rounded-full bg-accent transition-all"
              style={{ width: `${progress}%` }} />
          </div>
          <p className="mt-1 text-xs">Uploading… {progress}%</p>
        </div>
      )}
    </section>
  )
}
```

- [ ] **Step 5: Jobs page** — replace `webapp/src/pages/JobsPage.jsx`:

```jsx
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { clearToken, listJobs } from '../api'
import StatusChip from '../components/StatusChip'
import UploadCard from '../components/UploadCard'
import { fmtTime, shortId } from '../lib/status'

export default function JobsPage() {
  const navigate = useNavigate()
  const [jobs, setJobs] = useState([])

  useEffect(() => {
    let alive = true
    const load = () => listJobs().then((j) => alive && setJobs(j)).catch(() => {})
    load()
    const t = setInterval(load, 3000)
    return () => { alive = false; clearInterval(t) }
  }, [])

  return (
    <main className="mx-auto max-w-5xl px-4 py-8">
      <header className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold">
          <span className="text-accent">Batch</span> analysis
        </h1>
        <button onClick={() => { clearToken(); navigate('/login') }}
          className="text-sm underline">Sign out</button>
      </header>
      <UploadCard onCreated={(job) => navigate(`/jobs/${job.id}`)} />
      <section className="mt-8 overflow-x-auto rounded-xl border border-gray-200 bg-white">
        {jobs.length === 0 ? (
          <p className="p-6 text-center text-sm">
            No batches yet. Upload one above to see validation, progress and results here.
          </p>
        ) : (
          <table className="w-full text-left text-sm">
            <thead className="border-b border-gray-200 text-xs uppercase tracking-wide">
              <tr>
                <th className="px-4 py-3">Batch</th><th className="px-4 py-3">Uploaded</th>
                <th className="px-4 py-3">Status</th><th className="px-4 py-3">Progress</th>
                <th className="px-4 py-3">Files</th><th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id} className="border-b border-gray-100 last:border-0">
                  <td className="px-4 py-3">
                    <div className="font-medium text-ink">{j.original_name}</div>
                    <div className="font-mono text-xs text-gray-400">{shortId(j.id)}</div>
                  </td>
                  <td className="px-4 py-3">{fmtTime(j.created_at)}</td>
                  <td className="px-4 py-3"><StatusChip status={j.status} /></td>
                  <td className="px-4 py-3 font-mono">
                    {j.status === 'running' ? `${j.done}/${j.total}` : '—'}
                  </td>
                  <td className="px-4 py-3">{j.total}</td>
                  <td className="px-4 py-3 text-right">
                    <button onClick={() => navigate(`/jobs/${j.id}`)}
                      className="font-medium text-accent">Open</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </main>
  )
}
```

- [ ] **Step 6: Verify build + tests**

Run: `cd webapp && npx vitest run && npm run build`
Expected: 4 tests pass; build succeeds.

- [ ] **Step 7: Commit**

```bash
git status
git add webapp/src
git commit -m "feat(webapp): jobs list with polling + ZIP/folder upload card

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Job detail page — validation report, live analysis queue, results, downloads

**Files:**
- Create: `webapp/src/components/ValidationReport.jsx`, `webapp/src/components/LiveQueue.jsx`, `webapp/src/components/ResultsSection.jsx`
- Modify: `webapp/src/pages/JobPage.jsx` (replace placeholder), `webapp/src/index.css` (pulse animation)

**Interfaces:**
- Consumes: `api.getJob/startJob/rerunJob/deleteJob/getResults/getErrors/downloadArtifact`, `lib/status.buildQueueRows/isActive/STATUS_META`, job dict incl. `files: list[str]`.
- Produces: `/jobs/:id` renders each state per the spec; the **signature element** is `<LiveQueue>` — three Sora stat cards + per-file queue with pulsing current row (brand: AutoAce's live-call-queue metaphor).

- [ ] **Step 1: Pulse animation** (append to `webapp/src/index.css`):

```css
@keyframes queue-pulse {
  0%, 100% { background-color: rgb(219 234 254); }   /* blue-100 */
  50% { background-color: rgb(191 219 254); }        /* blue-200 */
}
.queue-analyzing { animation: queue-pulse 1.6s ease-in-out infinite; }
```

(The global `prefers-reduced-motion` rule from Task 9 already collapses this to a static highlight.)

- [ ] **Step 2: ValidationReport** — `webapp/src/components/ValidationReport.jsx`:

```jsx
export default function ValidationReport({ job, onStart, onDiscard }) {
  const hasManifest = !job.warnings.some((w) => w.includes('no CSV manifest found'))
  return (
    <section className="rounded-xl border border-gray-200 bg-white p-6">
      <h2 className="text-lg font-semibold">Validation report</h2>
      <p className="mt-1 text-sm">
        <strong className="text-ink">{job.total}</strong> audio file{job.total === 1 ? '' : 's'} ready ·
        manifest {hasManifest ? 'found' : 'not found'}
      </p>
      {job.warnings.length > 0 && (
        <ul className="mt-3 space-y-1">
          {job.warnings.map((w) => (
            <li key={w} className="rounded-lg bg-amber-100 px-3 py-2 font-mono text-xs text-amber-700">
              {w}
            </li>
          ))}
        </ul>
      )}
      <p className="mt-3 text-xs text-gray-400">
        Nothing has been processed yet. Review the report, then start the analysis.
      </p>
      <div className="mt-4 flex gap-3">
        <button onClick={onStart}
          className="rounded-lg bg-navy px-4 py-2 text-sm font-medium text-white">
          Start processing
        </button>
        <button onClick={onDiscard}
          className="rounded-lg border border-gray-200 px-4 py-2 text-sm font-medium text-ink">
          Discard batch
        </button>
      </div>
    </section>
  )
}
```

- [ ] **Step 3: LiveQueue (signature element)** — `webapp/src/components/LiveQueue.jsx`:

```jsx
import { buildQueueRows } from '../lib/status'

function Stat({ value, label }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 text-center">
      <div className="font-display text-3xl font-bold text-accent">{value}</div>
      <div className="mt-1 text-xs uppercase tracking-wide">{label}</div>
    </div>
  )
}

function elapsed(startedAt) {
  if (!startedAt) return '0:00'
  const s = Math.max(0, Math.floor((Date.now() - new Date(startedAt)) / 1000))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

export default function LiveQueue({ job }) {
  const rows = buildQueueRows(job.files, job.done, job.total)
  return (
    <section>
      <div className="grid grid-cols-3 gap-3">
        <Stat value={job.done} label="Analyzed" />
        <Stat value={job.total - job.done} label="Remaining" />
        <Stat value={elapsed(job.started_at)} label="Elapsed" />
      </div>
      <div className="mt-3 h-2 overflow-hidden rounded-full bg-gray-200">
        <div className="h-full rounded-full bg-accent transition-all"
          style={{ width: `${job.total ? (job.done / job.total) * 100 : 0}%` }} />
      </div>
      <ul className="mt-4 max-h-96 space-y-1 overflow-y-auto rounded-xl border border-gray-200 bg-white p-3">
        {rows.map((r) => (
          <li key={r.name}
            className={`flex items-center justify-between rounded-lg px-3 py-1.5 font-mono text-xs ${
              r.state === 'analyzing' ? 'queue-analyzing text-blue-700'
              : r.state === 'completed' ? 'text-gray-400 line-through decoration-green-700/40'
              : 'text-body'}`}>
            <span>{r.name}</span>
            <span>{r.state === 'completed' ? '✓' : r.state === 'analyzing' ? 'analyzing…' : 'queued'}</span>
          </li>
        ))}
      </ul>
      {job.status === 'queued' && (
        <p className="mt-2 text-xs">Waiting for the current batch to finish — one batch runs at a time.</p>
      )}
      <p className="mt-2 text-xs text-gray-400">
        Safe to close or reload this page — progress is saved on the server.
      </p>
    </section>
  )
}
```

- [ ] **Step 4: ResultsSection** — `webapp/src/components/ResultsSection.jsx`:

```jsx
import { useEffect, useState } from 'react'
import { downloadArtifact, getErrors, getResults } from '../api'

const FIELDS = ['emotional_tone', 'emotional_intensity', 'background_noise_present',
  'background_noise_type', 'background_noise_severity', 'audio_quality',
  'speaker_overlap_present', 'long_silence_present', 'confidence']

const ENUM_CHIP = {
  neutral: 'bg-gray-100 text-gray-700', satisfied: 'bg-green-100 text-green-700',
  frustrated: 'bg-amber-100 text-amber-700', upset: 'bg-red-100 text-red-700',
  distressed: 'bg-red-100 text-red-700', low: 'bg-gray-100 text-gray-700',
  medium: 'bg-amber-100 text-amber-700', high: 'bg-red-100 text-red-700',
  none: 'bg-gray-100 text-gray-700', clear: 'bg-green-100 text-green-700',
  slightly_impaired: 'bg-amber-100 text-amber-700', severely_impaired: 'bg-red-100 text-red-700',
}

function Cell({ value }) {
  if (typeof value === 'boolean') return <span>{value ? '✓' : '—'}</span>
  if (typeof value === 'number') return <span className="font-mono">{value.toFixed(2)}</span>
  if (value === '') return <span className="text-gray-300">—</span>
  const chip = ENUM_CHIP[value]
  return chip
    ? <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${chip}`}>{value}</span>
    : <span className="font-mono text-xs">{value}</span>
}

export default function ResultsSection({ job }) {
  const [rows, setRows] = useState([])
  const [errors, setErrors] = useState([])
  const [filter, setFilter] = useState('')

  useEffect(() => {
    getResults(job.id).then(setRows).catch(() => {})
    getErrors(job.id).then(setErrors).catch(() => {})
  }, [job.id])

  const shown = rows.filter((r) => r.name.toLowerCase().includes(filter.toLowerCase()))

  return (
    <section className="space-y-6">
      <div className="grid grid-cols-3 gap-3 text-center">
        {[[job.results_count, 'Succeeded', 'text-green-700'],
          [job.errors_count, 'Failed', job.errors_count ? 'text-red-700' : 'text-gray-400'],
          [job.warnings.length, 'Warnings', job.warnings.length ? 'text-amber-700' : 'text-gray-400'],
        ].map(([v, label, tone]) => (
          <div key={label} className="rounded-xl border border-gray-200 bg-white p-4">
            <div className={`font-display text-3xl font-bold ${tone}`}>{v ?? 0}</div>
            <div className="mt-1 text-xs uppercase tracking-wide">{label}</div>
          </div>
        ))}
      </div>

      {job.warnings.length > 0 && (
        <ul className="space-y-1">
          {job.warnings.map((w) => (
            <li key={w} className="rounded-lg bg-amber-100 px-3 py-2 font-mono text-xs text-amber-700">{w}</li>
          ))}
        </ul>
      )}

      {errors.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-red-200 bg-white">
          <table className="w-full text-left text-sm">
            <thead className="bg-red-100 text-xs uppercase tracking-wide text-red-700">
              <tr><th className="px-4 py-2">Failed file</th><th className="px-4 py-2">Reason</th></tr>
            </thead>
            <tbody>
              {errors.map((e) => (
                <tr key={e.name} className="border-t border-red-100">
                  <td className="px-4 py-2 font-mono text-xs">{e.name}</td>
                  <td className="px-4 py-2 font-mono text-xs">{e.error}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="flex items-center justify-between gap-3">
        <input value={filter} onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter by filename…"
          className="w-56 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm" />
        <div className="flex gap-2">
          {['results.csv', 'results.json', 'errors.csv'].map((a) => (
            <button key={a} onClick={() => downloadArtifact(job.id, a)}
              className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 font-mono text-xs text-ink">
              ⬇ {a}
            </button>
          ))}
        </div>
      </div>

      <div className="max-h-[32rem] overflow-auto rounded-xl border border-gray-200 bg-white">
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 bg-white text-xs uppercase tracking-wide shadow-sm">
            <tr>
              <th className="px-3 py-2">name</th>
              {FIELDS.map((f) => <th key={f} className="px-3 py-2">{f.replaceAll('_', ' ')}</th>)}
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.name} className="border-t border-gray-100">
                <td className="px-3 py-2 font-mono text-xs text-ink">{r.name}</td>
                {FIELDS.map((f) => <td key={f} className="px-3 py-2"><Cell value={r[f]} /></td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
```

- [ ] **Step 5: JobPage** — replace `webapp/src/pages/JobPage.jsx`:

```jsx
import { useCallback, useEffect, useState } from 'react'
import toast from 'react-hot-toast'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { deleteJob, getJob, rerunJob, startJob } from '../api'
import LiveQueue from '../components/LiveQueue'
import ResultsSection from '../components/ResultsSection'
import StatusChip from '../components/StatusChip'
import ValidationReport from '../components/ValidationReport'
import { isActive } from '../lib/status'

export default function JobPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [job, setJob] = useState(null)
  const [missing, setMissing] = useState(false)

  const refresh = useCallback(() =>
    getJob(id).then(setJob).catch((e) => e.response?.status === 404 && setMissing(true)), [id])

  useEffect(() => {
    refresh()
    const t = setInterval(() => {
      setJob((j) => { if (!j || isActive(j.status)) refresh(); return j })
    }, 2000)
    return () => clearInterval(t)
  }, [refresh])

  if (missing) return <Shell><p className="text-sm">This batch no longer exists.</p></Shell>
  if (!job) return <Shell><p className="text-sm">Loading…</p></Shell>

  const act = (fn, okMsg) => () =>
    fn(job.id).then(() => { okMsg && toast.success(okMsg); refresh() })
      .catch((e) => toast.error(e.response?.data?.detail ?? 'Action failed'))

  return (
    <Shell>
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">
            <span className="text-accent">Batch</span> {job.original_name}
          </h1>
          <p className="mt-1 font-mono text-xs text-gray-400">{job.id}</p>
        </div>
        <StatusChip status={job.status} />
      </header>

      {job.status === 'awaiting_confirmation' && (
        <ValidationReport job={job}
          onStart={act(startJob, 'Batch queued')}
          onDiscard={() => deleteJob(job.id).then(() => navigate('/'))} />
      )}
      {(job.status === 'queued' || job.status === 'running') && <LiveQueue job={job} />}
      {job.status === 'completed' && <ResultsSection job={job} />}
      {(job.status === 'failed' || job.status === 'interrupted') && (
        <section className="rounded-xl border border-red-200 bg-white p-6">
          <h2 className="text-lg font-semibold text-red-700">
            {job.status === 'failed' ? 'Batch failed' : 'Batch interrupted'}
          </h2>
          <p className="mt-2 rounded-lg bg-red-100 px-3 py-2 font-mono text-xs text-red-700">
            {job.error ?? 'No details recorded.'}
          </p>
          <button onClick={act(rerunJob, 'Batch re-queued')}
            className="mt-4 rounded-lg bg-navy px-4 py-2 text-sm font-medium text-white">
            Re-run batch
          </button>
        </section>
      )}
    </Shell>
  )
}

function Shell({ children }) {
  return (
    <main className="mx-auto max-w-6xl px-4 py-8">
      <Link to="/" className="text-sm text-accent">← All batches</Link>
      <div className="mt-4">{children}</div>
    </main>
  )
}
```

- [ ] **Step 6: Verify build + tests**

Run: `cd webapp && npx vitest run && npm run build`
Expected: tests pass; build succeeds.

- [ ] **Step 7: Manual smoke against the stub server**

With the Task 9 dev setup running (`DASHBOARD_STUB_ANALYZE=1`): upload a small ZIP (e.g. zip two copies of any `.wav` plus a `labels.csv`), see the validation report, press **Start processing**, watch the live queue tick both files, land on the results table, download all three artifacts. Reload mid-run — the queue view must restore from server state.

- [ ] **Step 8: Commit**

```bash
git status
git add webapp/src
git commit -m "feat(webapp): job detail — validation report, live analysis queue, results review

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: Single-origin serving, Make targets, Playwright smoke, real-pipeline acceptance

**Files:**
- Modify: `src/dashboard/app.py` (serve `webapp/dist`), `Makefile` (new targets)
- Create: `webapp/playwright.config.js`, `webapp/e2e/smoke.spec.js`, `webapp/e2e/fixtures/` (generated ZIP)
- Test: `tests/web/test_static.py`

**Interfaces:**
- Consumes: everything prior.
- Produces: `GET /` serves the built SPA (client-side routes fall back to `index.html`; `/api/*` unaffected); `make webapp-build`, `make web`, `make web-dev`.

- [ ] **Step 1: Failing static test** — `tests/web/test_static.py`:

```python
from pathlib import Path

import pytest

DIST = Path(__file__).resolve().parents[2] / "webapp" / "dist"


@pytest.mark.skipif(not DIST.exists(), reason="webapp not built")
def test_spa_served_with_client_route_fallback(client):
    for path in ("/", "/jobs/deadbeef"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")


def test_api_404_stays_json_404(client, auth_header):
    r = client.get("/api/jobs/nope", headers=auth_header)
    assert r.status_code == 404
    assert r.json()["detail"] == "Job not found"


def test_security_headers_on_every_response(client):
    r = client.post("/api/auth/login", json={"username": "x", "password": "y"})
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
```

Run: `.venv/bin/pytest tests/web/test_static.py -v` → SPA test FAILS (404 for `/`), API test passes.

- [ ] **Step 2: Security headers + serve the SPA** (append inside `create_app()` in `src/dashboard/app.py`, after `include_router`):

```python
    @app.middleware("http")
    async def _security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    from pathlib import Path

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    dist = Path(__file__).resolve().parents[2] / "webapp" / "dist"
    if dist.exists():  # dev API-only mode works without a build
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            candidate = dist / path
            if path and ".." not in path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app
```

(Route registration order makes `/api/*` win; the catch-all only sees everything else.)

Run: `cd webapp && npm run build && cd .. && .venv/bin/pytest tests/web/test_static.py -v` → 2 PASS.

- [ ] **Step 3: Make targets** (append to `Makefile`, matching its existing style):

```make
webapp-build:  ## build the React SPA into webapp/dist
	cd webapp && npm install && npm run build

web: webapp-build  ## serve API + built SPA on :8000
	.venv/bin/uvicorn --factory dashboard.app:create_app --host 0.0.0.0 --port 8000

web-dev:  ## API with reload; pair with `cd webapp && npm run dev` for HMR
	.venv/bin/uvicorn --factory dashboard.app:create_app --reload --port 8000
```

Run: `make web-dev` briefly → uvicorn boots, Ctrl-C.

- [ ] **Step 4: Playwright smoke** — `webapp/playwright.config.js`:

```js
import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  use: { baseURL: 'http://127.0.0.1:8000' },
  timeout: 120000,
})
```

Generate the fixture ZIP (commit it — ~300 bytes):

```bash
.venv/bin/python - <<'EOF'
import zipfile, pathlib
p = pathlib.Path("webapp/e2e/fixtures"); p.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(p / "batch.zip", "w") as z:
    for n in ("call_001.wav", "call_002.wav"):
        z.writestr(n, b"RIFF0000WAVEfake")
    z.writestr("labels.csv", "name,result_json\ncall_001.wav,\ncall_002.wav,\n")
EOF
```

`webapp/e2e/smoke.spec.js`:

```js
import { expect, test } from '@playwright/test'

// Server must be running with DASHBOARD_STUB_ANALYZE=1 and the SPA built.
// Credentials come from env so real ones never land in the repo.
const USER = process.env.E2E_USER
const PASS = process.env.E2E_PASS

test('login → upload → confirm → live queue → results → download', async ({ page }) => {
  await page.goto('/login')
  await page.getByLabel('Username').fill(USER)
  await page.getByLabel('Password').fill(PASS)
  await page.getByRole('button', { name: 'Sign in' }).click()

  await page.getByRole('button', { name: 'Choose ZIP' }).waitFor()
  const chooser = page.waitForEvent('filechooser')
  await page.getByRole('button', { name: 'Choose ZIP' }).click()
  await (await chooser).setFiles('e2e/fixtures/batch.zip')

  await expect(page.getByText('Validation report')).toBeVisible()
  await expect(page.getByText('2 audio files ready')).toBeVisible()
  await page.getByRole('button', { name: 'Start processing' }).click()

  await expect(page.getByText('call_001.wav')).toBeVisible()
  await expect(page.getByText('Succeeded')).toBeVisible({ timeout: 90000 })

  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: '⬇ results.csv' }).click()
  expect((await download).suggestedFilename()).toBe('results.csv')
})
```

Run once locally:

```bash
cd webapp && npx playwright install chromium && cd ..
E2E_USER=<dev user> E2E_PASS=<dev password> — with the stub server from Task 9 Step 6 running:
cd webapp && E2E_USER=... E2E_PASS=... npx playwright test
```
Expected: 1 passed.

- [ ] **Step 5: Full-suite gate**

Run: `.venv/bin/pytest tests/web/ -q && make test && cd webapp && npx vitest run && npm run build`
Expected: everything green.

- [ ] **Step 6: Commit**

```bash
git status
git add src/dashboard/app.py Makefile tests/web/test_static.py webapp/playwright.config.js webapp/e2e
git commit -m "feat(dashboard): single-origin SPA serving, make targets, e2e smoke

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Real-pipeline acceptance (manual, no stub)**

1. `.env` filled (real `GEMINI_API_KEY`, `DASHBOARD_*` set, `DASHBOARD_STUB_ANALYZE` unset).
2. `make web` → open `http://127.0.0.1:8000`, sign in.
3. Zip the three real sample calls from `data/test_recordings/` together with their `labels.csv` manifest; upload through the browser.
4. Confirm: validation report → start → live queue advances per file (first file is slow: models load once per worker) → completed.
5. Cross-check the 9-field rows against the pipeline's known outputs for these calls (`out/validation_report.md` history); download all three artifacts and open them.
6. Reload the page mid-run once — the queue must restore from server state.
7. Report the result to the user; push `git push -u origin dashboard` and tell them "dashboard ready for merge review".

---

## Deviations & backend requests

Any needed change to `src/autoace_audio/` goes into `docs/DASHBOARD-BACKEND-REQUESTS.md` (committed) instead of being made here — none are anticipated by this plan.
