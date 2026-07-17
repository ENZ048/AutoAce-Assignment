import numpy as np
from eval.experiments.exp2_fewshot import best_window

from autoace_audio.analyzers.vad import Segment, VadMap

SR = 16000


def _vad(speech, total_s):
    return VadMap(
        speech=[Segment(a, b) for a, b in speech],
        gaps=[],
        speech_ratio=0.5,
        max_gap_s=0.0,
        long_silence_present=False,
        total_s=total_s,
    )


def test_best_window_finds_densest_speech():
    # speech: thin at start, dense 40-60s
    vad = _vad([(2.0, 4.0), (40.0, 58.0)], 80.0)
    start, end = best_window(np.zeros(80 * SR, np.float32), SR, vad, win_s=20.0)
    assert 38.0 <= start <= 40.0 and end - start == 20.0


def test_best_window_ties_break_earliest():
    vad = _vad([(0.0, 10.0), (30.0, 40.0)], 60.0)
    start, _ = best_window(np.zeros(60 * SR, np.float32), SR, vad, win_s=20.0)
    assert start == 0.0


def test_best_window_short_clip_returns_whole():
    vad = _vad([(0.0, 5.0)], 12.0)
    assert best_window(np.zeros(12 * SR, np.float32), SR, vad, win_s=20.0) == (0.0, 12.0)
