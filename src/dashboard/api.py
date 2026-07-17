"""All /api routes."""

import csv
import json
import shutil
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from autoace_audio.batch import validate_batch
from dashboard import store
from dashboard.auth import create_token, require_auth, verify_login
from dashboard.config import get_dashboard_settings
from dashboard.zipsafe import UnsafeZipError, extract_zip

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
        raise HTTPException(
            400,
            "Upload one ZIP archive, or select a folder (audio files + one CSV manifest).",
        )

    # Folder uploads flatten each part to its basename (brief: audio files live at
    # the batch root). Reject blank/duplicate names up front, before any job row or
    # directory exists, so a bad request never needs cleanup and never orphans state.
    names: list[str] = []
    if not is_zip:
        seen: set[str] = set()
        for f in files:
            name = Path(f.filename or "").name
            if not name:
                raise HTTPException(400, "Every uploaded file must have a filename.")
            if name in seen:
                raise HTTPException(400, f"Duplicate filename in upload: {name}")
            seen.add(name)
            names.append(name)

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
            root = extract_zip(
                upload_path, extracted, max_extracted_bytes=settings.max_extract_mb * 1024 * 1024
            )
        else:
            extracted.mkdir(parents=True, exist_ok=True)
            for f, name in zip(files, names, strict=True):
                _stream_to(extracted / name, f, used, cap_bytes)
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
    except HTTPException:
        # Any HTTPException raised from inside the try (not one of the specific
        # cases above) still leaves a row + directory behind unless we clean up.
        _discard(db, job_id, job_dir)
        raise
    except Exception:
        # Safety net: any other failure (e.g. validate_batch raising unexpectedly)
        # must not strand the job row or its directory. Clean up and propagate.
        _discard(db, job_id, job_dir)
        raise


def _discard(db, job_id: str, job_dir: Path) -> None:
    store.delete_job(db, job_id)
    shutil.rmtree(job_dir, ignore_errors=True)


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


_ARTIFACTS = {
    "results.csv": "text/csv",
    "results.json": "application/json",
    "errors.csv": "text/csv",
}


def _job_or_404(db, job_id: str) -> dict:
    job = store.get_job(db, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


def _out_path(job: dict, request: Request, job_id: str, artifact: str) -> Path:
    # Status gate first: after a rerun, stale artifacts from a prior attempt can still
    # sit on disk while the job is queued/running again — gate on the job's own status
    # before ever looking at the filesystem, so a "running" job card can't serve torn
    # or superseded data. Existence stays as the second layer for the completed case
    # where out/ hasn't been written yet (shouldn't happen, but cheap to keep).
    if job["status"] != "completed":
        raise HTTPException(409, "Results are available once the batch has completed.")
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
    # Clear the previous attempt's artifacts before requeueing — otherwise they remain
    # servable through the read routes while the rerun is queued/running.
    shutil.rmtree(request.app.state.jobs_dir / job_id / "out", ignore_errors=True)
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
    job = _job_or_404(request.app.state.db, job_id)
    path = _out_path(job, request, job_id, "results.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(
            500, "The results file for this batch is unreadable; re-run the batch."
        ) from e
    return [{"name": name, **fields} for name, fields in data.items()]


@router.get("/jobs/{job_id}/errors")
def job_errors(job_id: str, request: Request, user: str = Depends(require_auth)):
    job = _job_or_404(request.app.state.db, job_id)
    with open(_out_path(job, request, job_id, "errors.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@router.get("/jobs/{job_id}/download/{artifact}")
def download_artifact(
    job_id: str, artifact: str, request: Request, user: str = Depends(require_auth)
):
    if artifact not in _ARTIFACTS:
        raise HTTPException(404, "Unknown artifact")
    job = _job_or_404(request.app.state.db, job_id)
    return FileResponse(
        _out_path(job, request, job_id, artifact),
        media_type=_ARTIFACTS[artifact],
        filename=artifact,
    )
