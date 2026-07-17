import os
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


def test_spawn_worker_is_a_detached_subprocess(monkeypatch):
    """The worker must be a session-detached subprocess, not a multiprocessing
    child: multiprocessing's atexit either terminates daemonic children or
    joins non-daemonic ones on clean interpreter exit, so either flavor makes
    a routine server restart kill (or block on) an in-flight batch. A detached
    Popen child survives, which is what sweep_orphans' adoption path expects."""
    captured = {}

    class FakePopen:
        pid = 111

        def __init__(self, args, **kw):
            captured["args"] = list(args)
            captured["kw"] = kw

        def poll(self):
            return None

        returncode = None

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    h = runner._spawn_worker("j1", "db.sqlite", "/tmp/root", "/tmp/out", stub=True)
    assert captured["kw"]["start_new_session"] is True
    assert captured["args"][1:3] == ["-m", "dashboard.worker"]
    assert captured["args"][3:] == ["j1", "db.sqlite", "/tmp/root", "/tmp/out", "1"]
    assert h.pid == 111 and h.is_alive() and h.exitcode is None


def _fake_report():
    class R:
        warnings = []
        results = {}
        errors = []

    return R()


def test_worker_preloads_models_before_run_batch(tmp_path, monkeypatch):
    """Real (non-stub) workers load the model singletons up front, and mark
    the job with the loading sentinel while doing so, so model-load time
    never masquerades as first-file analysis time in the UI."""
    import autoace_audio.batch as batch_mod
    import autoace_audio.pipeline as pipeline_mod

    calls = []
    sentinel_seen = {}

    db = store.connect(tmp_path / "t.db")
    store.create_job(db, "j1", "b.zip")
    store.set_status(db, "j1", "running")

    def fake_preload():
        calls.append("preload")
        sentinel_seen["during"] = store.get_job(db, "j1")["current_file"]

    def fake_run_batch(*a, **k):
        calls.append("run")
        return _fake_report()

    monkeypatch.setattr(pipeline_mod, "preload_models", fake_preload)
    monkeypatch.setattr(batch_mod, "run_batch", fake_run_batch)
    runner.worker_main("j1", str(tmp_path / "t.db"), str(tmp_path), str(tmp_path / "out"), False)
    assert calls == ["preload", "run"]
    assert sentinel_seen["during"] == store.MODEL_LOADING
    assert store.get_job(db, "j1")["current_file"] != store.MODEL_LOADING  # cleared after
    db.close()


def test_worker_skips_preload_in_stub_mode(tmp_path, monkeypatch):
    import autoace_audio.batch as batch_mod
    import autoace_audio.pipeline as pipeline_mod

    calls = []
    monkeypatch.setattr(pipeline_mod, "preload_models", lambda: calls.append("preload"))
    monkeypatch.setattr(
        batch_mod, "run_batch", lambda *a, **k: (calls.append("run"), _fake_report())[1]
    )
    db = store.connect(tmp_path / "t.db")
    store.create_job(db, "j1", "b.zip")
    store.set_status(db, "j1", "running")
    runner.worker_main("j1", str(tmp_path / "t.db"), str(tmp_path), str(tmp_path / "out"), True)
    assert calls == ["run"]
    db.close()


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


def test_sweep_leaves_running_row_with_live_pid(tmp_path):
    db = store.connect(tmp_path / "t.db")
    store.create_job(db, "r1", "b.zip")
    store.set_status(db, "r1", "running")
    store.set_worker_pid(db, "r1", os.getpid())  # our own pid — definitely alive
    runner.sweep_orphans(db)
    assert store.get_job(db, "r1")["status"] == "running"
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
    monkeypatch.setattr(runner, "_spawn_worker", lambda *a, **k: _FakeProc())
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


def test_dispatch_treats_adopted_live_worker_as_busy(tmp_path):
    runner._processes.clear()
    db = store.connect(tmp_path / "t.db")
    store.create_job(db, "adopted", "a.zip")
    store.set_status(db, "adopted", "running")
    store.set_worker_pid(db, "adopted", os.getpid())
    store.create_job(db, "waiting", "b.zip")
    store.set_status(db, "waiting", "queued")
    assert runner.dispatch_once(db, tmp_path / "t.db", tmp_path, stub=True) is False
    assert store.get_job(db, "adopted")["status"] == "running"
    assert store.get_job(db, "waiting")["status"] == "queued"
    db.close()


def _noop():
    pass


def _ctx_dead_pid():
    """Spawn a real short-lived process and return its pid after join — guaranteed dead,
    not reused within the test (unlike a made-up integer, which risks colliding with a
    live pid on the test host)."""
    import multiprocessing as mp

    p = mp.get_context("spawn").Process(target=_noop)
    p.start()
    p.join()
    return p.pid


def test_dead_pid_running_row_is_interrupted_and_queue_resumes(tmp_path, monkeypatch):
    # fake Process handle so no real worker spawns when the queue resumes
    class _FakeProc:
        pid = 4242

        def is_alive(self):
            return True

        @property
        def exitcode(self):
            return None

    monkeypatch.setattr(runner, "_spawn_worker", lambda *a, **k: _FakeProc())
    runner._processes.clear()
    db = store.connect(tmp_path / "t.db")
    store.create_job(db, "stale", "a.zip")
    store.set_status(db, "stale", "running")
    dead = _ctx_dead_pid()
    store.set_worker_pid(db, "stale", dead)
    store.create_job(db, "waiting", "b.zip")
    (tmp_path / "waiting").mkdir()
    (tmp_path / "waiting" / "batch_root.txt").write_text(str(tmp_path / "waiting"))
    store.set_status(db, "waiting", "queued")
    assert runner.dispatch_once(db, tmp_path / "t.db", tmp_path, stub=True) is True
    stale = store.get_job(db, "stale")
    assert stale["status"] == "interrupted"
    # Distinct from sweep_orphans' restart message: this row had no process handle in
    # THIS server process and a dead pid discovered mid-tick, which (post-startup-sweep)
    # only happens to an adopted worker that has since died — not a server restart.
    assert stale["error"] == "worker process died unexpectedly"
    assert store.get_job(db, "waiting")["status"] == "running"
    runner._processes.clear()
    db.close()


def test_wedge_guard_missing_batch_root_fails_job_and_queue_continues(tmp_path, monkeypatch):
    """A queued job whose batch_root.txt is missing/unreadable must not permanently
    stall the dispatcher: it gets marked failed and the next queued job still starts,
    in the same dispatch_once tick."""

    monkeypatch.setattr(runner, "_spawn_worker", lambda *a, **k: _FakeProc())
    runner._processes.clear()
    db = store.connect(tmp_path / "t.db")
    store.create_job(db, "wedged", "a.zip")
    (tmp_path / "wedged").mkdir()  # batch_root.txt deliberately absent
    store.set_status(db, "wedged", "queued")
    time.sleep(1.1)  # created_at has second resolution; keep ordering unambiguous
    store.create_job(db, "healthy", "b.zip")
    (tmp_path / "healthy").mkdir()
    (tmp_path / "healthy" / "batch_root.txt").write_text(str(tmp_path / "healthy"))
    store.set_status(db, "healthy", "queued")
    assert runner.dispatch_once(db, tmp_path / "t.db", tmp_path, stub=True) is True
    wedged = store.get_job(db, "wedged")
    assert wedged["status"] == "failed"
    assert wedged["error"]
    assert store.get_job(db, "healthy")["status"] == "running"
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
    assert done["failed_files"] == ["call_bad.wav"]  # live per-file failure marker


def test_completed_batch_is_audit_logged_and_survives_delete(
    client, auth_header, tmp_path, app_env
):
    """After a real batch completes, its predictions are in the durable audit
    log; deleting the batch from the UI must not remove that record."""
    import json

    from dashboard.config import get_dashboard_settings

    z = make_batch_zip(tmp_path / "b.zip")
    job = client.post(
        "/api/jobs",
        headers=auth_header,
        files=[("files", ("b.zip", z.read_bytes(), "application/zip"))],
    ).json()
    client.post(f"/api/jobs/{job['id']}/start", headers=auth_header)
    _wait_for(client, auth_header, job["id"], "completed")

    audit_file = get_dashboard_settings().data_dir / "audit.jsonl"
    before = [json.loads(x) for x in audit_file.read_text().strip().splitlines()]
    assert {r["file"] for r in before} == {"call_001.wav", "call_002.wav"}
    assert all(r["job_id"] == job["id"] for r in before)

    r = client.delete(f"/api/jobs/{job['id']}", headers=auth_header)
    assert r.status_code == 204
    assert client.get(f"/api/jobs/{job['id']}", headers=auth_header).status_code == 404
    # the batch is gone from the dashboard, but the audit record persists
    after = [json.loads(x) for x in audit_file.read_text().strip().splitlines()]
    assert after == before


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
