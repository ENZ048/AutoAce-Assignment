"""Job execution: one spawned worker process at a time runs run_batch.
The API process owns queueing (dispatch_once); the worker owns progress
writes and its own terminal transition."""

import logging
import os
import subprocess
import sys
from pathlib import Path

from dashboard import store

logger = logging.getLogger(__name__)

_processes: dict[str, object] = {}  # job_id -> live worker handle (this server process only)


class _WorkerHandle:
    """A session-detached worker subprocess.

    Deliberately subprocess.Popen, not multiprocessing: multiprocessing's
    atexit handler terminates daemonic children and joins non-daemonic ones
    on clean interpreter exit, so either flavor makes a routine server
    restart kill (or block for the length of) an in-flight batch. A detached
    Popen child keeps running; on restart sweep_orphans adopts it via its
    recorded pid — the lifecycle that machinery was built for."""

    def __init__(self, args: list[str]):
        self._proc = subprocess.Popen(args, start_new_session=True)

    @property
    def pid(self) -> int:
        return self._proc.pid

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    @property
    def exitcode(self) -> int | None:
        return self._proc.returncode


def _spawn_worker(job_id: str, db_path: str, batch_root: str, out_dir: str, stub: bool):
    return _WorkerHandle(
        [
            sys.executable,
            "-m",
            "dashboard.worker",
            job_id,
            db_path,
            batch_root,
            out_dir,
            "1" if stub else "0",
        ]
    )


def _pid_alive(pid: int | None) -> bool:
    """Best-effort liveness check: signal 0 probes for delivery, sends nothing.

    Known limitation (documented, not solved here): PID reuse can make an
    unrelated process look like our worker, notably after a host reboot on
    bare metal. In the containerized deployment target, a fresh PID namespace
    per container start makes a stale pid reliably dead, so the check is sound
    there. The failure mode with a false-alive pid is a conservatively stuck
    'running' row — safer than the double-running-workers bug this guards
    against.
    """
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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

        def progress(done: int, total: int, name: str, failed: str | None = None) -> None:
            store.update_progress(db, job_id, done=done, current_file=name, failed=failed)

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
        # Guarded like store.finish: only overwrite if the row is still 'running', so a
        # stale/adopted worker whose job was already superseded (interrupted/failed by
        # the dispatcher or startup sweep) can't resurrect it. Reading status then writing
        # is racy in principle, but the only other writers (dispatch_once, sweep_orphans)
        # apply the same status='running' precondition before moving a row off 'running',
        # so this narrow window can't produce a wrong terminal state.
        current = store.get_job(db, job_id)
        if current is not None and current["status"] == "running":
            store.set_status(db, job_id, "failed", error=f"{type(e).__name__}: {e}")
    finally:
        db.close()


def sweep_orphans(db) -> None:
    """Startup: previous server process left these behind — unless a 'running' row's
    worker_pid is still alive, meaning the server restarted but the spawned worker
    (a separate OS process) survived and is still working. Adopt it rather than
    marking it interrupted, otherwise dispatch_once would start a second worker
    alongside it: two concurrent ~4GB workers, the exact thing this module prevents."""
    for job in store.list_jobs(db):
        if job["status"] == "running":
            if _pid_alive(job["worker_pid"]):
                logger.warning(
                    "job %s: adopted worker (pid %s) appears alive; leaving it running, "
                    "will be monitored by the dispatcher",
                    job["id"],
                    job["worker_pid"],
                )
                continue
            store.set_status(db, job["id"], "interrupted", error="interrupted by server restart")
        elif job["status"] == "queued":
            store.set_status(db, job["id"], "interrupted", error="interrupted by server restart")
        elif job["status"] == "validating":
            store.set_status(db, job["id"], "failed", error="interrupted during validation")


def dispatch_once(db, db_path: Path, jobs_dir: Path, stub: bool) -> bool:
    """Reap finished workers; start the oldest queued job if nothing runs. True if started one."""
    for job in store.list_jobs(db):
        if job["status"] != "running":
            continue
        proc = _processes.get(job["id"])
        if proc is None:  # running row without a handle → predates a restart, or an adopted orphan
            if _pid_alive(job["worker_pid"]):
                return False  # adopted orphan still working — treat as busy, start nothing
            # sweep_orphans already resolves every dead-pid 'running' row it finds at
            # startup, so by the time dispatch_once observes a handle-less 'running' row
            # with a dead pid, it can only be an adopted orphan whose worker has since
            # died mid-tick — not a restart. Distinct message from sweep_orphans'.
            store.set_status(db, job["id"], "interrupted", error="worker process died unexpectedly")
            continue
        if proc.is_alive():
            return False  # a job is genuinely running — nothing else may start
        _processes.pop(job["id"], None)
        if store.get_job(db, job["id"])["status"] == "running":  # died without terminal write
            store.set_status(
                db, job["id"], "failed", error=f"worker process died (exit code {proc.exitcode})"
            )
    queued = [j for j in store.list_jobs(db) if j["status"] == "queued"]
    # list_jobs is newest-first, so iterate in reverse (oldest first). A queued job
    # whose batch_root.txt is missing/unreadable must not wedge the whole queue: mark
    # THAT job failed and move on to the next queued job in the same tick, instead of
    # letting the read raise out of dispatch_once (which would re-raise on every
    # subsequent tick forever, since the same oldest-queued job is picked each time).
    for job in reversed(queued):
        job_dir = jobs_dir / job["id"]
        try:
            batch_root = (job_dir / "batch_root.txt").read_text(encoding="utf-8").strip()
        except OSError as e:
            store.set_status(
                db,
                job["id"],
                "failed",
                error=f"could not read batch_root.txt: {type(e).__name__}: {e}",
            )
            continue
        store.set_status(db, job["id"], "running")
        proc = _spawn_worker(job["id"], str(db_path), batch_root, str(job_dir / "out"), stub=stub)
        _processes[job["id"]] = proc
        store.set_worker_pid(db, job["id"], proc.pid)
        return True
    return False
