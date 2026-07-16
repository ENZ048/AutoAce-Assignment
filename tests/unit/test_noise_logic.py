import numpy as np
import pytest

from autoace_audio.analyzers.noise import (
    _support_weights,
    _window_starts,
    concise_label,
    severity_from_snr,
    snr_db,
)
from autoace_audio.analyzers.vad import Segment, VadMap
from autoace_audio.schema import Severity


def _vad(speech, gaps, total):
    return VadMap(
        speech=speech,
        gaps=gaps,
        speech_ratio=0.5,
        max_gap_s=0.0,
        long_silence_present=False,
        total_s=total,
    )


def test_snr_loud_speech_quiet_gap_is_high_db():
    sr = 16000
    speech = 0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    gap = 0.005 * np.random.default_rng(0).standard_normal(sr)
    samples = np.concatenate([speech, gap]).astype(np.float32)
    v = _vad([Segment(0.0, 1.0)], [Segment(1.0, 2.0)], 2.0)
    assert snr_db(samples, sr, v) > 20


def test_snr_none_when_no_gaps():
    sr = 16000
    samples = np.ones(sr, dtype=np.float32) * 0.1
    v = _vad([Segment(0.0, 1.0)], [], 1.0)
    assert snr_db(samples, sr, v) is None


def test_severity_mapping_and_presence_invariant():
    assert severity_from_snr(25.0, present=False) == Severity.NONE
    assert severity_from_snr(25.0, present=True) == Severity.LOW  # present => at least low
    assert severity_from_snr(10.0, present=True) == Severity.MEDIUM
    assert severity_from_snr(2.0, present=True) == Severity.HIGH
    assert severity_from_snr(None, present=True) == Severity.LOW


def test_concise_labels():
    assert concise_label("Television") == "TV"
    assert concise_label("Hubbub, speech noise, speech babble") == "office chatter"
    assert concise_label("Vehicle") == "road noise"
    assert concise_label("SomeUnknownClass") == "someunknownclass"


# --- _window_starts / _support_weights: pure windowing + support-crediting logic,
# no model needed. Regression coverage for two reviewer-verified bugs: (1) flat
# hop_s crediting let a tail-anchored window -- which can sit much closer than
# hop_s to its predecessor -- double-count one spike as two independent
# detections; (2) a fixed aed_min_support_s floor made short clips (<5s, and the
# ~5-10s two-window range) physically unable to ever report presence.


@pytest.mark.parametrize(
    ("total_n", "window_n", "hop_n", "expected"),
    [
        (48000, 80000, 40000, [0]),  # 3.0s clip, shorter than the 5.0s window
        (80000, 80000, 40000, [0]),  # exactly one window length (<=, not <)
        (1000, 400, 200, [0, 200, 400, 600]),  # exact hop-multiple: no tail append
    ],
)
def test_window_starts(total_n, window_n, hop_n, expected):
    assert _window_starts(total_n, window_n, hop_n) == expected


def test_window_starts_tail_anchor_matches_residual_gap():
    """172s-shaped clip (window=5.0s, hop=2.5s @16kHz): the tail-anchored window
    sits at 167.0s, only 2.0s after the last regular window at 165.0s (a 60%
    overlap) -- not a full hop_s away."""
    sr = 16000
    starts = _window_starts(total_n=172 * sr, window_n=5 * sr, hop_n=int(2.5 * sr))
    starts_s = [st / sr for st in starts]
    assert starts_s[-2:] == [165.0, 167.0]


def test_window_starts_near_duplicate_tail_for_barely_over_one_window():
    """5.1s clip (window=5.0s): the only two windows are 0.1s apart, a 98%
    overlap -- essentially the same audio scored twice."""
    sr = 16000
    starts = _window_starts(total_n=int(5.1 * sr), window_n=5 * sr, hop_n=int(2.5 * sr))
    starts_s = [round(st / sr, 4) for st in starts]
    assert starts_s == [0.0, 0.1]


@pytest.mark.parametrize(
    ("starts", "total_s", "expected"),
    [
        ([0.0], 3.0, [3.0]),  # single window, clip shorter than window_s
        ([0.0], 5.0, [5.0]),  # single window, clip exactly window_s (<=, not <)
        ([0.0, 2.5, 5.0, 7.5], 12.5, [2.5, 2.5, 2.5, 2.5]),  # exact hop-multiple
    ],
)
def test_support_weights(starts, total_s, expected):
    assert _support_weights(starts, total_s=total_s, window_s=5.0, hop_s=2.5) == expected


def test_support_weights_tail_window_credited_residual_not_hop():
    """The 172s shape's tail window is only 2.0s after its predecessor (165.0 vs
    167.0) -- it must earn 2.0s of credit, NOT the flat hop_s=2.5s the old buggy
    code credited every window, which let a single spike straddling this
    near-duplicate pair masquerade as two independent detections."""
    weights = _support_weights([165.0, 167.0], total_s=172.0, window_s=5.0, hop_s=2.5)
    assert weights == [2.5, 2.0]


def test_support_weights_near_duplicate_windows_cannot_reach_default_floor():
    """5.1s clip: both windows (0.1s apart) activating for the same spike sums to
    only 2.6s of support -- below the 5.0s default aed_min_support_s, which is
    exactly why analyze_noise floor-caps at min(aed_min_support_s, sum(weights))
    instead of demanding support the clip can never physically provide."""
    weights = _support_weights([0.0, 0.1], total_s=5.1, window_s=5.0, hop_s=2.5)
    assert weights == [2.5, 0.1]
    assert sum(weights) == pytest.approx(2.6)
    assert sum(weights) < 5.0
