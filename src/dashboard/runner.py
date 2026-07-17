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
        store.finish(
            db,
            job_id,
            results_count=len(report.results),
            errors_count=len(report.errors),
            extra_warnings=extra,
        )
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
            store.set_status(
                db, job["id"], "failed", error=f"worker process died (exit code {proc.exitcode})"
            )
    queued = [j for j in store.list_jobs(db) if j["status"] == "queued"]
    if not queued:
        return False
    job = queued[-1]  # list_jobs is newest-first → last is the oldest queued
    job_dir = jobs_dir / job["id"]
    batch_root = (job_dir / "batch_root.txt").read_text(encoding="utf-8").strip()
    proc = _ctx.Process(
        target=worker_main,
        kwargs=dict(
            job_id=job["id"],
            db_path=str(db_path),
            batch_root=batch_root,
            out_dir=str(job_dir / "out"),
            stub=stub,
        ),
        daemon=True,
    )
    store.set_status(db, job["id"], "running")
    proc.start()
    _processes[job["id"]] = proc
    store.set_worker_pid(db, job["id"], proc.pid)
    return True
