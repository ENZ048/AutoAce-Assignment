import numpy as np

from autoace_audio.analyzers.noise import concise_label, severity_from_snr, snr_db
from autoace_audio.analyzers.vad import Segment, VadMap
from autoace_audio.schema import Severity


def _vad(speech, gaps, total):
    return VadMap(speech=speech, gaps=gaps, speech_ratio=0.5, max_gap_s=0.0,
                  long_silence_present=False, total_s=total)


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
