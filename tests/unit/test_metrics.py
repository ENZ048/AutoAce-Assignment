from eval.metrics import confusion, field_report, macro_f1


def test_macro_f1_perfect():
    assert macro_f1(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_macro_f1_weights_minority_class_equally():
    y_true = ["a"] * 9 + ["b"]
    always_a = ["a"] * 10
    assert macro_f1(y_true, always_a) < 0.5  # majority-vote cheat is punished


def test_confusion_counts():
    c = confusion(["a", "a", "b"], ["a", "b", "b"])
    assert c["a"]["a"] == 1 and c["a"]["b"] == 1 and c["b"]["b"] == 1


def test_field_report_handles_partial_truth_rows_without_emotional_tone():
    """Regression: eval/build_validation_set.py's noise_aug/quality_aug rows only
    label the ONE field they were constructed to test (e.g. just
    background_noise_severity), never emotional_tone. field_report must not crash
    building its tone confusion matrix when some labeled clips have no
    emotional_tone truth at all -- discovered running the harness for real against
    the augmented validation set (KeyError: 'emotional_tone')."""
    labels = {
        "call_001.wav": {"emotional_tone": "upset", "audio_quality": "clear"},
        "aug_clip.wav": {"audio_quality": "severely_impaired"},  # no emotional_tone
    }
    preds = {
        "call_001.wav": {"emotional_tone": "upset", "audio_quality": "clear"},
        "aug_clip.wav": {"emotional_tone": "neutral", "audio_quality": "severely_impaired"},
    }
    report = field_report(labels, preds)  # must not raise
    assert "audio_quality" in report
    assert "macro F1: 1.000" in report  # only call_001.wav has tone truth, and it matches


def test_field_report_handles_zero_tone_labeled_clips():
    """Regression: eval/build_validation_set.py's real validation_manifest.csv has
    ZERO clips with emotional_tone truth at all (VAD segments carry empty truth
    pending hand-labeling; every synthetic augmentation labels only its one target
    field) -- macro_f1([], []) divides by zero unless field_report guards the
    tone section when no clip has tone truth."""
    labels = {"aug_clip.wav": {"audio_quality": "severely_impaired"}}
    preds = {"aug_clip.wav": {"emotional_tone": "neutral", "audio_quality": "severely_impaired"}}
    report = field_report(labels, preds)  # must not raise ZeroDivisionError
    assert "audio_quality" in report
