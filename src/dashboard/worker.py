"""Detached worker entry point: python -m dashboard.worker <job_id> <db_path>
<batch_root> <out_dir> <0|1 stub>. Spawned by runner._spawn_worker as its own
session so it survives a server restart (see runner._WorkerHandle)."""

import sys

from dashboard.runner import worker_main


def main(argv: list[str]) -> None:
    job_id, db_path, batch_root, out_dir, stub = argv
    worker_main(job_id, db_path, batch_root, out_dir, stub == "1")


if __name__ == "__main__":
    main(sys.argv[1:])
