import json

import pytest
from eval.experiments import common
from eval.experiments.common import (
    BudgetExceeded,
    SpendGuard,
    field_compare,
    gemini_cost,
    log_run,
    loses_field,
    wins_field,
)


def test_gemini_cost_math():
    # 1M in + 1M out at default rates = $2.00; None counts as zero
    assert gemini_cost(1_000_000, 1_000_000) == pytest.approx(2.00)
    assert gemini_cost(1509, 102) == pytest.approx(1509 * 0.50 / 1e6 + 102 * 1.50 / 1e6)
    assert gemini_cost(None, None) == 0.0


def test_spend_guard_caps_at_10_and_persists(tmp_path):
    state = tmp_path / "spend.json"
    g = SpendGuard(state_path=state, cap_usd=10.0, warn_usd=7.0)
    g.add(6.0)
    g.check(1.0)  # 7.0 projected: fine (warning only past warn threshold)
    g.add(3.5)
    with pytest.raises(BudgetExceeded):
        g.check(1.0)  # 9.5 + 1.0 > 10.0
    # persisted: a fresh instance sees the same total
    g2 = SpendGuard(state_path=state, cap_usd=10.0, warn_usd=7.0)
    assert g2.total() == pytest.approx(9.5)
    assert json.loads(state.read_text())["total_usd"] == pytest.approx(9.5)


def test_field_compare():
    pred = {"a": 1, "b": "x", "c": True}
    truth = {"a": 1, "b": "y", "c": True}
    assert field_compare(pred, truth, ["a", "b", "c"]) == {"a": True, "b": False, "c": True}


def _runs(correct_flags: list[bool], clip="call_002.ogg", field="emotional_tone"):
    return [{"per_clip": {clip: {"pred": {}, "correct": {field: flag}}}} for flag in correct_flags]


def test_wins_field_requires_2of3_flip():
    base = _runs([False, False, False])
    good = _runs([True, True, False])
    bad = _runs([True, False, False])
    assert wins_field(base, good, "emotional_tone", "call_002.ogg") is True
    assert wins_field(base, bad, "emotional_tone", "call_002.ogg") is False
    # baseline already mostly right -> no win even if lever is right
    base_ok = _runs([True, True, False])
    assert wins_field(base_ok, good, "emotional_tone", "call_002.ogg") is False


def test_loses_field_requires_solid_baseline_flipped_solidly_wrong():
    # baseline solidly right (2/3), lever solidly wrong (0/3) -> regression
    base = _runs([True, True, False])
    flipped = _runs([False, False, False])
    assert loses_field(base, flipped, "emotional_tone", "call_002.ogg") is True


def test_loses_field_false_when_lever_only_partially_wrong():
    # lever right 2/3 is NOT "solidly wrong" (threshold is <= n//3 = 1 for
    # n=3, so a 2/3 partial wobble must not count as a full regression --
    # mirrors the real E4 call_001-overlap wobble: 3/3 baseline -> 2/3
    # lever, correctly NOT counted as a regression)
    base = _runs([True, True, True])
    wobble = _runs([True, True, False])
    assert loses_field(base, wobble, "emotional_tone", "call_002.ogg") is False


def test_loses_field_false_when_baseline_is_not_solid():
    # baseline itself only 1/3 right -> nothing solid to regress FROM, even
    # if the lever is solidly wrong too (mirrors wins_field's own baseline
    # eligibility gate)
    base_weak = _runs([True, False, False])
    flipped = _runs([False, False, False])
    assert loses_field(base_weak, flipped, "emotional_tone", "call_002.ogg") is False


def test_loses_field_false_when_unchanged_both_wrong():
    # a clip that's simply unchanged (both baseline and lever wrong) is
    # neither a win nor a regression -- confirms loses_field is a genuine
    # mirror of wins_field, not just "not a win"
    base = _runs([False, False, False])
    same = _runs([False, False, False])
    assert wins_field(base, same, "emotional_tone", "call_002.ogg") is False
    assert loses_field(base, same, "emotional_tone", "call_002.ogg") is False


def test_loses_field_empty_runs_is_false():
    assert loses_field([], _runs([False, False, False]), "emotional_tone", "call_002.ogg") is False
    assert loses_field(_runs([True, True, True]), [], "emotional_tone", "call_002.ogg") is False


def test_log_run_without_cost_usd_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    with pytest.raises(ValueError, match="every run log must carry its measured cost"):
        log_run("test_exp", 0, {"some": "payload"})


def test_spend_guard_warn_does_not_raise(tmp_path, capsys):
    state = tmp_path / "spend.json"
    g = SpendGuard(state_path=state, cap_usd=10.0, warn_usd=7.0)
    g.add(6.0)
    g.check(1.5)  # 6.0 + 1.5 = 7.5 > 7.0 warns, but < 10.0 does not raise
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "7.50" in captured.out
