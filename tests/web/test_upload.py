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
