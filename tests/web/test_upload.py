from fastapi.testclient import TestClient
from tests.web.conftest import PASSWORD, make_batch_zip


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
        "/api/jobs",
        headers=auth_header,
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


def test_duplicate_folder_filenames_rejected(client, auth_header):
    r = client.post(
        "/api/jobs",
        headers=auth_header,
        files=[
            ("files", ("call_001.wav", b"RIFF0000WAVEfake1", "audio/wav")),
            ("files", ("call_001.wav", b"RIFF0000WAVEfake2", "audio/wav")),
        ],
    )
    assert r.status_code == 400
    assert "call_001.wav" in r.json()["detail"]
    assert client.get("/api/jobs", headers=auth_header).json() == []


def test_unexpected_validation_error_leaves_no_orphan(app_env, tmp_path, monkeypatch):
    """The shared `client` fixture uses TestClient's default
    raise_server_exceptions=True, which re-raises an unhandled exception out of
    client.post(...) instead of handing back a 500 response. A local client with
    raise_server_exceptions=False is built here instead so the assertions can run
    against a normal response (see task-6-report.md for the tradeoff notes)."""
    import dashboard.api as api_mod
    from dashboard.app import create_app

    def boom(root):
        raise RuntimeError("validation exploded")

    monkeypatch.setattr(api_mod, "validate_batch", boom)

    with TestClient(create_app(), raise_server_exceptions=False) as local_client:
        login = local_client.post(
            "/api/auth/login", json={"username": "autoace", "password": PASSWORD}
        )
        auth = {"Authorization": f"Bearer {login.json()['access_token']}"}
        z = make_batch_zip(tmp_path / "b.zip")
        r = local_client.post(
            "/api/jobs",
            headers=auth,
            files=[("files", ("b.zip", z.read_bytes(), "application/zip"))],
        )
        assert r.status_code == 500
        assert local_client.get("/api/jobs", headers=auth).json() == []

    jobs_root = app_env / "data" / "jobs"
    remaining = list(jobs_root.iterdir()) if jobs_root.exists() else []
    assert remaining == []


def test_blank_filename_part_rejected_before_job_created(app_env):
    """httpx's own files= builder silently drops an empty filename (the part
    becomes a plain form field, not a file, and FastAPI 422s before our route
    even runs) so this crafts the raw multipart body to get a genuine
    UploadFile(filename="") past FastAPI's parsing and into our validation —
    reproducing the crash Finding 1 describes."""
    from dashboard.app import create_app

    with TestClient(create_app(), raise_server_exceptions=False) as local_client:
        login = local_client.post(
            "/api/auth/login", json={"username": "autoace", "password": PASSWORD}
        )
        auth = f"Bearer {login.json()['access_token']}"
        body = (
            b"--XBOUND\r\n"
            b'Content-Disposition: form-data; name="files"; filename=""\r\n'
            b"Content-Type: audio/wav\r\n\r\n"
            b"RIFF0000WAVEfake\r\n"
            b"--XBOUND\r\n"
            b'Content-Disposition: form-data; name="files"; filename="labels.csv"\r\n'
            b"Content-Type: text/csv\r\n\r\n"
            b"name,result_json\n\r\n"
            b"--XBOUND--\r\n"
        )
        headers = {
            "Authorization": auth,
            "Content-Type": "multipart/form-data; boundary=XBOUND",
        }
        r = local_client.post("/api/jobs", headers=headers, content=body)
        assert r.status_code == 400
        assert "filename" in r.json()["detail"]
        assert local_client.get("/api/jobs", headers={"Authorization": auth}).json() == []
