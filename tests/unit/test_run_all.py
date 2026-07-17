"""Mock-level tests for run_all.py: resumable orchestration across exp0-5 +
combined. Every stage module and combined itself is monkeypatched to a fake
recording collaborator -- these tests exercise run_all's OWN orchestration
logic (skip/resume/force, stage order, spend summary), not any real
experiment's behavior (that belongs to exp0-5's own test files). Real
tmp-dir run logs (via common.log_run/read_runs) drive the resumability
checks, per Task 8's Job item 3 ("unit-test the resume logic with tmp
dirs, do NOT re-run the other experiments live")."""

from types import SimpleNamespace

import pytest
from eval.experiments import common, run_all


class _FakeGuard:
    """Never touches the real out/experiments/spend.json (its default
    state_path is bound at class-definition time, before any per-test
    OUT_DIR monkeypatch could reach it) -- same isolation convention as
    every other exp module's tests (test_exp4_flash.py's _OrderedGuard)."""

    def __init__(self, *a, **kw):
        pass

    def total(self):
        return 0.0

    def check(self, projected_usd):
        pass

    def add(self, cost_usd):
        pass


def _fake_stage(name, calls):
    def run_once(i):
        calls.append((name, i))
        common.log_run(name, i, {"cost_usd": 0.0, "run": i})

    return SimpleNamespace(run_once=run_once)


def _fake_combined(calls, stack=None, verify_raises=None):
    stack = stack if stack is not None else {"gap_noise": True}

    def decide_stack():
        calls.append(("decide_stack",))
        return stack

    def verify_stack_matches_determination(s):
        calls.append(("verify", s))
        if verify_raises:
            raise verify_raises

    def run_once(i, s):
        calls.append(("combined_run_once", i, s))
        common.log_run("combined", i, {"cost_usd": 0.0, "run": i})

    return SimpleNamespace(
        decide_stack=decide_stack,
        verify_stack_matches_determination=verify_stack_matches_determination,
        run_once=run_once,
    )


def _install(monkeypatch, tmp_path, stages, combined_fake):
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    monkeypatch.setattr(run_all, "STAGES", stages)
    monkeypatch.setattr(run_all, "combined", combined_fake)
    monkeypatch.setattr(run_all, "SpendGuard", _FakeGuard)


# ---------------------------------------------------------------------------
# Fresh run: nothing on disk, every stage executes, combined runs last with
# the decided stack.
# ---------------------------------------------------------------------------


def test_run_all_executes_every_stage_in_order_when_no_logs_exist(monkeypatch, tmp_path):
    calls: list = []
    stages = [
        ("stage_a", _fake_stage("stage_a", calls), 2),
        ("stage_b", _fake_stage("stage_b", calls), 1),
    ]
    _install(monkeypatch, tmp_path, stages, _fake_combined(calls))

    run_all.run(force=False)

    assert calls == [
        ("stage_a", 1),
        ("stage_a", 2),
        ("stage_b", 1),
        ("decide_stack",),
        ("verify", {"gap_noise": True}),
        ("combined_run_once", 1, {"gap_noise": True}),
        ("combined_run_once", 2, {"gap_noise": True}),
        ("combined_run_once", 3, {"gap_noise": True}),
    ]


# ---------------------------------------------------------------------------
# Resumability: skip complete stages, resume partial ones from the next
# index, never re-run what already has logs.
# ---------------------------------------------------------------------------


def test_run_all_skips_a_stage_whose_logs_already_fully_exist(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    for i in range(1, 4):
        common.log_run("stage_a", i, {"cost_usd": 0.0, "run": i})

    def _must_not_run(i):
        raise AssertionError("stage_a.run_once must not be called -- its 3 logs already exist")

    stages = [("stage_a", SimpleNamespace(run_once=_must_not_run), 3)]
    _install(monkeypatch, tmp_path, stages, _fake_combined(calls))

    run_all.run(force=False)  # must not raise -- the assertion above is the real check


def test_run_all_resumes_a_partial_stage_from_the_next_index(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    common.log_run("stage_a", 1, {"cost_usd": 0.0, "run": 1})  # only run 1 exists

    stages = [("stage_a", _fake_stage("stage_a", calls), 3)]
    _install(monkeypatch, tmp_path, stages, _fake_combined(calls))

    run_all.run(force=False)

    stage_a_calls = [c for c in calls if c[0] == "stage_a"]
    assert stage_a_calls == [("stage_a", 2), ("stage_a", 3)]  # run 1 skipped, not redone


# ---------------------------------------------------------------------------
# --force: regression tests for the brief's own bug (force on an
# already-complete stage was silently a no-op, since the loop started at
# done+1 == runs+1, an empty range, even when force was requested).
# ---------------------------------------------------------------------------


def test_run_all_force_redoes_a_fully_complete_stage(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    for i in range(1, 4):
        common.log_run("stage_a", i, {"cost_usd": 0.0, "run": i})

    stages = [("stage_a", _fake_stage("stage_a", calls), 3)]
    _install(monkeypatch, tmp_path, stages, _fake_combined(calls))

    run_all.run(force=True)

    stage_a_calls = [c for c in calls if c[0] == "stage_a"]
    assert stage_a_calls == [("stage_a", 1), ("stage_a", 2), ("stage_a", 3)]


def test_run_all_force_also_redoes_combined_when_already_complete(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    for i in range(1, 4):
        common.log_run("combined", i, {"cost_usd": 0.0, "run": i})

    _install(monkeypatch, tmp_path, [], _fake_combined(calls))

    run_all.run(force=True)

    combined_calls = [c for c in calls if c[0] == "combined_run_once"]
    assert [c[1] for c in combined_calls] == [1, 2, 3]


def test_run_all_without_force_does_not_redo_a_fully_complete_stage(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    for i in range(1, 4):
        common.log_run("stage_a", i, {"cost_usd": 0.0, "run": i})

    stages = [("stage_a", _fake_stage("stage_a", calls), 3)]
    _install(monkeypatch, tmp_path, stages, _fake_combined(calls))

    run_all.run(force=False)

    assert [c for c in calls if c[0] == "stage_a"] == []


# ---------------------------------------------------------------------------
# combined receives the decided stack, and a mismatched stack must abort
# BEFORE any live combined run (verify_stack_matches_determination's whole
# job).
# ---------------------------------------------------------------------------


def test_run_all_calls_combined_with_the_decided_stack_after_all_stages(monkeypatch, tmp_path):
    calls: list = []
    stack = {"gap_noise": True, "flash": False}
    stages = [("stage_a", _fake_stage("stage_a", calls), 1)]
    _install(monkeypatch, tmp_path, stages, _fake_combined(calls, stack=stack))

    run_all.run(force=False)

    assert ("verify", stack) in calls
    combined_calls = [c for c in calls if c[0] == "combined_run_once"]
    assert combined_calls and all(c[2] == stack for c in combined_calls)


def test_run_all_propagates_verify_mismatch_instead_of_running_combined_live(monkeypatch, tmp_path):
    calls: list = []
    combined_fake = _fake_combined(calls, verify_raises=RuntimeError("stack mismatch"))
    _install(monkeypatch, tmp_path, [], combined_fake)

    with pytest.raises(RuntimeError, match="stack mismatch"):
        run_all.run(force=False)

    assert not any(c[0] == "combined_run_once" for c in calls)  # never reached


# ---------------------------------------------------------------------------
# Spend summary + CLI wiring.
# ---------------------------------------------------------------------------


def test_run_all_prints_study_complete_with_final_spend(monkeypatch, tmp_path, capsys):
    calls: list = []
    _install(monkeypatch, tmp_path, [], _fake_combined(calls))

    run_all.run(force=False)

    out = capsys.readouterr().out
    assert "STUDY COMPLETE" in out
    assert "$" in out


def test_main_wires_the_force_flag_from_argv(monkeypatch, tmp_path):
    calls: list = []
    _install(monkeypatch, tmp_path, [], _fake_combined(calls))
    monkeypatch.setattr("sys.argv", ["run_all.py", "--force"])
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    for i in range(1, 4):
        common.log_run("combined", i, {"cost_usd": 0.0, "run": i})

    run_all.main()

    # force=True redoes combined even though 3 logs already existed
    combined_calls = [c for c in calls if c[0] == "combined_run_once"]
    assert [c[1] for c in combined_calls] == [1, 2, 3]


def test_main_defaults_force_to_false(monkeypatch, tmp_path):
    calls: list = []
    _install(monkeypatch, tmp_path, [], _fake_combined(calls))
    monkeypatch.setattr("sys.argv", ["run_all.py"])
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    for i in range(1, 4):
        common.log_run("combined", i, {"cost_usd": 0.0, "run": i})

    run_all.main()

    assert not any(c[0] == "combined_run_once" for c in calls)  # already 3/3, no force -> skip
