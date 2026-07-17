import time

import pytest
from tests.web.conftest import make_batch_zip

from dashboard import runner, store


def test_stub_analyze_returns_valid_result_and_fails_on_bad(tmp_path):
    out = runner.stub_analyze(tmp_path / "call_001.wav")
    d = out.result.model_dump(mode="json")
    assert d["emotional_tone"] == "neutral" and d["confidence"] == 0.9
    with pytest.raises(ValueError):
        runner.stub_analyze(tmp_path / "call_bad.wav")


def test_sweep_orphans_marks_stale_jobs(tmp_path):
    db = store.connect(tmp_path / "t.db")
    statuses = [("r1", "running"), ("q1", "queued"), ("v1", "validating"), ("c1", "completed")]
    for jid, status in statuses:
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

    def __init__(self, *a, **k):
        self._alive = True

    def start(self):
        pass

    def is_alive(self):
        return self._alive

    @property
    def exitcode(self):
        return None


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
    assert store.get_job(db, "older")["status"] == "running"  # oldest queued first
    assert runner.dispatch_once(db, tmp_path / "t.db", tmp_path, stub=True) is False
    assert store.get_job(db, "newer")["status"] == "queued"  # waits its turn
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
    job = client.post(
        "/api/jobs",
        headers=auth_header,
        files=[("files", ("b.zip", z.read_bytes(), "application/zip"))],
    ).json()
    r = client.post(f"/api/jobs/{job['id']}/start", headers=auth_header)
    assert r.status_code == 200 and r.json()["status"] == "queued"
    done = _wait_for(client, auth_header, job["id"], "completed")
    assert done["done"] == 2 and done["total"] == 2
    assert done["results_count"] == 2 and done["errors_count"] == 0
    assert done["started_at"] and done["finished_at"]


def test_per_file_failure_isolation_surfaces_in_counts(client, auth_header, tmp_path):
    z = make_batch_zip(tmp_path / "b.zip", names=("call_001.wav", "call_bad.wav"))
    job = client.post(
        "/api/jobs",
        headers=auth_header,
        files=[("files", ("b.zip", z.read_bytes(), "application/zip"))],
    ).json()
    client.post(f"/api/jobs/{job['id']}/start", headers=auth_header)
    done = _wait_for(client, auth_header, job["id"], "completed")
    assert done["results_count"] == 1 and done["errors_count"] == 1


def test_start_requires_awaiting_confirmation(client, auth_header, tmp_path):
    z = make_batch_zip(tmp_path / "b.zip")
    job = client.post(
        "/api/jobs",
        headers=auth_header,
        files=[("files", ("b.zip", z.read_bytes(), "application/zip"))],
    ).json()
    client.post(f"/api/jobs/{job['id']}/start", headers=auth_header)
    r = client.post(f"/api/jobs/{job['id']}/start", headers=auth_header)
    assert r.status_code == 409
    assert client.post("/api/jobs/nope/start", headers=auth_header).status_code == 404
