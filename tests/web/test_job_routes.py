import json

from tests.web.conftest import make_batch_zip

from dashboard import store

RESULT_FIELDS = [
    "emotional_tone",
    "emotional_intensity",
    "background_noise_present",
    "background_noise_type",
    "background_noise_severity",
    "audio_quality",
    "speaker_overlap_present",
    "long_silence_present",
    "confidence",
]


def _job_with_artifacts(client, auth_header, app_env, tmp_path):
    """Upload a real batch, then hand-write out/ + completed status (no worker spawn)."""
    z = make_batch_zip(tmp_path / "b.zip")
    job = client.post(
        "/api/jobs",
        headers=auth_header,
        files=[("files", ("b.zip", z.read_bytes(), "application/zip"))],
    ).json()
    out = app_env / "data" / "jobs" / job["id"] / "out"
    out.mkdir(parents=True)
    row = {
        f: (
            "neutral"
            if f == "emotional_tone"
            else "low"
            if f == "emotional_intensity"
            else False
            if f.endswith("_present")
            else ""
            if f == "background_noise_type"
            else "none"
            if f == "background_noise_severity"
            else "clear"
            if f == "audio_quality"
            else 0.9
        )
        for f in RESULT_FIELDS
    }
    (out / "results.json").write_text(json.dumps({"call_001.wav": row}), encoding="utf-8")
    (out / "results.csv").write_text(
        'name,result_json\ncall_001.wav,"{""emotional_tone"": ""neutral""}"\n', encoding="utf-8"
    )
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
    job = client.post(
        "/api/jobs",
        headers=auth_header,
        files=[("files", ("b.zip", z.read_bytes(), "application/zip"))],
    ).json()
    assert client.get(f"/api/jobs/{job['id']}/results", headers=auth_header).status_code == 409


def test_download_serves_artifacts_verbatim(client, auth_header, app_env, tmp_path):
    job = _job_with_artifacts(client, auth_header, app_env, tmp_path)
    disk = (app_env / "data" / "jobs" / job["id"] / "out" / "results.csv").read_bytes()
    r = client.get(f"/api/jobs/{job['id']}/download/results.csv", headers=auth_header)
    assert r.status_code == 200 and r.content == disk
    r_evil = client.get(f"/api/jobs/{job['id']}/download/evil.txt", headers=auth_header)
    assert r_evil.status_code == 404


def test_rerun_only_from_failed_or_interrupted(client, auth_header, app_env, tmp_path):
    job = _job_with_artifacts(client, auth_header, app_env, tmp_path)  # completed
    assert client.post(f"/api/jobs/{job['id']}/rerun", headers=auth_header).status_code == 409
    db = store.connect(app_env / "data" / "dashboard.db")
    store.set_status(db, job["id"], "interrupted", error="interrupted by server restart")
    db.close()
    r = client.post(f"/api/jobs/{job['id']}/rerun", headers=auth_header)
    assert r.status_code == 200 and r.json()["status"] in ("queued", "running", "completed")


def test_delete_blocked_while_active_then_removes_everything(
    client, auth_header, app_env, tmp_path
):
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
