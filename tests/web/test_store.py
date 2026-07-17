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
    store.set_status(db, "j1", "running")  # finish() only completes a still-running row
    store.finish(db, "j1", results_count=1, errors_count=1, extra_warnings=["1/2 files fell back"])
    job = store.get_job(db, "j1")
    assert job["status"] == "completed"
    assert job["results_count"] == 1 and job["errors_count"] == 1
    assert job["warnings"] == ["1/2 files fell back"]


def test_finish_only_completes_from_running(db):
    store.create_job(db, "j1", "b.zip")
    store.set_status(db, "j1", "interrupted", error="interrupted by server restart")
    store.finish(db, "j1", results_count=2, errors_count=0, extra_warnings=[])
    job = store.get_job(db, "j1")
    assert job["status"] == "interrupted"
    assert job["results_count"] is None


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
    a.close()
    b.close()


def test_finish_on_deleted_job_is_a_noop(db):
    store.create_job(db, "j1", "b.zip")
    store.delete_job(db, "j1")
    store.finish(db, "j1", results_count=1, errors_count=0, extra_warnings=["w"])  # must not raise
    assert store.get_job(db, "j1") is None
