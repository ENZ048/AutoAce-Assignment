import numpy as np
from eval.experiments.exp1_gap_noise import concat_gaps

from autoace_audio.analyzers.vad import Segment, VadMap

SR = 16000


def _vad(gaps, total_s):
    return VadMap(
        speech=[],
        gaps=[Segment(a, b) for a, b in gaps],
        speech_ratio=0.5,
        max_gap_s=max((b - a for a, b in gaps), default=0.0),
        long_silence_present=False,
        total_s=total_s,
    )


def test_concat_gaps_keeps_only_long_gaps_and_caps():
    total = 30.0
    samples = np.arange(int(total * SR), dtype=np.float32)
    vad = _vad([(0.0, 0.5), (2.0, 4.0), (10.0, 12.5)], total)
    out = concat_gaps(samples, SR, vad, min_gap_s=1.0, cap_s=60.0)
    # 0.5s gap dropped; 2.0s + 2.5s kept = 4.5s
    assert out.size == int(4.5 * SR)
    # content really comes from the gap regions (first kept sample = t=2.0s)
    assert out[0] == samples[int(2.0 * SR)]


def test_concat_gaps_caps_total_seconds():
    total = 200.0
    samples = np.zeros(int(total * SR), dtype=np.float32)
    vad = _vad([(0.0, 50.0), (60.0, 130.0)], total)
    out = concat_gaps(samples, SR, vad, cap_s=60.0)
    assert out.size == int(60.0 * SR)


def test_concat_gaps_empty_when_no_qualifying_gap():
    samples = np.zeros(SR, dtype=np.float32)
    vad = _vad([(0.0, 0.4)], 1.0)
    assert concat_gaps(samples, SR, vad).size == 0
