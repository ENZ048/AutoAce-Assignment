"""Pure scoring-accumulation logic from eval/bakeoff.py. No models/API calls --
importing eval.bakeoff is safe at module scope (classify_tone's arms all load
their models/clients lazily inside their own classify() functions, never at
import time)."""

from eval.bakeoff import ERROR_SENTINEL, record_outcome
from eval.metrics import macro_f1


def test_record_outcome_appends_successful_prediction():
    y_true, y_pred = [], []
    record_outcome(y_true, y_pred, "upset", "upset")
    assert y_true == ["upset"]
    assert y_pred == ["upset"]


def test_record_outcome_none_prediction_becomes_sentinel_not_excluded():
    """A failed classify_tone call (pred_tone=None) must still land in both
    lists -- review round 1, Important #1: a failed arm scores as a miss, it
    must not shrink its own denominator by being skipped."""
    y_true, y_pred = [], []
    record_outcome(y_true, y_pred, "upset", None)
    assert y_true == ["upset"]
    assert y_pred == [ERROR_SENTINEL]


def test_failed_clip_penalizes_accuracy_and_macro_f1():
    """End-to-end sanity check of the scoring rule: 2 correct + 1 failed clip
    must score as 2/3 accuracy, not 2/2 -- and the sentinel must never
    accidentally equal a real true label (which would let it score a "hit")."""
    y_true, y_pred = [], []
    record_outcome(y_true, y_pred, "upset", "upset")
    record_outcome(y_true, y_pred, "neutral", "neutral")
    record_outcome(y_true, y_pred, "satisfied", None)  # arm raised on this clip

    acc = sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)
    assert acc == 2 / 3
    assert macro_f1(y_true, y_pred) < 1.0
    assert ERROR_SENTINEL not in y_true  # sentinel never masquerades as a true label
