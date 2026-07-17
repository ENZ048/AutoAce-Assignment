"""Run the whole study in order, resumably: an experiment whose run logs
already exist is skipped unless --force. Sequential by design (16GB
machine: Global Constraints says never run two experiments concurrently).

The brief's own draft had two resumability bugs, fixed here (see
study-task-8-report.md for the full write-up and the regression tests that
catch them):
  - `--force` on an already-complete stage was silently a no-op: the resume
    loop started at `done + 1`, so when `done == runs` already (the stage
    fully exists) `range(done + 1, runs + 1)` is empty *regardless* of
    force -- `--force` never actually redid anything for a finished stage.
    Fixed by treating a forced stage as having 0 done runs, so it always
    redoes the full `runs` count from 1 (log_run overwrites in place).
  - the identical bug existed a second time, independently, in a
    special-cased block that ran `combined` after the main STAGES loop with
    its own copy of the same (buggy) logic. Fixed by routing both the
    exp0-5 stages AND combined through one shared `_run_stage` helper
    instead of duplicating the loop.
"""

import argparse

from eval.experiments import (
    combined,
    exp0_baseline,
    exp1_gap_noise,
    exp2_fewshot,
    exp3_advocate,
    exp4_flash,
    exp5_overlap,
)
from eval.experiments.common import SpendGuard, read_runs

STAGES = [
    ("exp0_baseline", exp0_baseline, 3),
    ("exp1_gap_noise", exp1_gap_noise, 3),
    ("exp2_fewshot", exp2_fewshot, 3),
    ("exp3_advocate", exp3_advocate, 3),
    ("exp4_flash", exp4_flash, 3),
    ("exp5_overlap", exp5_overlap, 1),
]


def _run_stage(name: str, run_fn, target_runs: int, force: bool) -> None:
    """run_fn(i) executes run i. Resumable: skips entirely once target_runs
    logs already exist, unless force -- which redoes ALL target_runs runs
    from 1 (not just the tail), since log_run overwrites existing files in
    place rather than appending."""
    done = 0 if force else len(read_runs(name))
    if done >= target_runs:
        print(f"{name}: {done} runs exist, skipping")
        return
    for i in range(done + 1, target_runs + 1):
        run_fn(i)
        print(f"{name} run {i} done; study total ${SpendGuard().total():.2f}")


def run(force: bool = False) -> None:
    for name, mod, runs in STAGES:
        _run_stage(name, mod.run_once, runs, force)
    stack = combined.decide_stack()
    print("stack:", stack)
    # Fail loudly BEFORE spending live money if a fresh computation ever
    # disagrees with the controller's recorded binding determination.
    combined.verify_stack_matches_determination(stack)
    _run_stage("combined", lambda i: combined.run_once(i, stack), 3, force)
    print(f"STUDY COMPLETE. total spend ${SpendGuard().total():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="redo every stage, ignoring existing logs")
    args = ap.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
