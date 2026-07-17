"""All /api routes."""

import json
import shutil
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
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
            root = extract_zip(upload_path, extracted)
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
