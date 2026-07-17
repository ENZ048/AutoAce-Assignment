import json
from types import SimpleNamespace

from dashboard import audit


class _Result:
    """Duck-typed AnalysisResult: only model_dump is used by the audit writer."""

    def __init__(self, tone):
        self._tone = tone

    def model_dump(self, mode="json"):
        return {"emotional_tone": self._tone, "confidence": 0.8}


def _report(results, errors=()):
    return SimpleNamespace(
        results={n: _Result(t) for n, t in results.items()},
        errors=[SimpleNamespace(name=n, error=e) for n, e in errors],
    )


def test_record_batch_writes_one_line_per_file(tmp_path):
    audit.record_batch(
        tmp_path,
        "job1",
        "batch.zip",
        _report({"a.wav": "upset", "b.wav": "neutral"}, errors=[("c.wav", "decode: boom")]),
        ts="2026-07-18T00:00:00+00:00",
    )
    lines = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    recs = [json.loads(x) for x in lines]
    ok = {r["file"]: r for r in recs if r["status"] == "ok"}
    assert ok["a.wav"]["emotional_tone"] == "upset"
    assert ok["a.wav"]["job_id"] == "job1" and ok["a.wav"]["batch"] == "batch.zip"
    err = next(r for r in recs if r["status"] == "error")
    assert err["file"] == "c.wav" and "decode" in err["error"]


def test_record_batch_appends_across_batches(tmp_path):
    audit.record_batch(
        tmp_path, "j1", "b1", _report({"a.wav": "upset"}), ts="2026-07-18T00:00:00+00:00"
    )
    audit.record_batch(
        tmp_path, "j2", "b2", _report({"x.wav": "satisfied"}), ts="2026-07-18T00:01:00+00:00"
    )
    lines = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2  # second call appended, did not overwrite
    assert {json.loads(x)["job_id"] for x in lines} == {"j1", "j2"}


def test_audit_survives_job_dir_deletion(tmp_path):
    """The audit file is a sibling of jobs/, so deleting a job dir cannot remove it."""
    import shutil

    (tmp_path / "jobs" / "j1").mkdir(parents=True)
    audit.record_batch(
        tmp_path, "j1", "b1", _report({"a.wav": "upset"}), ts="2026-07-18T00:00:00+00:00"
    )
    shutil.rmtree(tmp_path / "jobs" / "j1")  # what the delete route does
    assert (tmp_path / "audit.jsonl").exists()
    assert json.loads((tmp_path / "audit.jsonl").read_text().strip())["file"] == "a.wav"
