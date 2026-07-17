import json

import pytest
from eval.experiments.common import (
    BudgetExceeded,
    SpendGuard,
    field_compare,
    gemini_cost,
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
