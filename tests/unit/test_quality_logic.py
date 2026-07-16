import numpy as np

from autoace_audio.analyzers.quality import _dropout_count, _spectral_rolloff, rate_quality
from autoace_audio.analyzers.vad import Segment, VadMap
from autoace_audio.schema import AudioQuality


def _vad_one_segment(total_s: float) -> VadMap:
    return VadMap(
        speech=[Segment(0.0, total_s)], gaps=[], speech_ratio=1.0,
        max_gap_s=0.0, long_silence_present=False, total_s=total_s,
    )


def _vad_no_speech(total_s: float) -> VadMap:
    return VadMap(
        speech=[], gaps=[Segment(0.0, total_s)], speech_ratio=0.0,
        max_gap_s=total_s, long_silence_present=True, total_s=total_s,
    )


# --- rate_quality: pure decision logic over pre-measured evidence values ---
# Clean baseline used by every scenario below unless the test is specifically
# exercising that evidence value: clipping=0, dropouts=0, rolloff=3000Hz (wideband),
# volume=-20dBFS (loud), pesq=3.5 (good), snr=20dB (clean-ish background).


def test_no_adverse_evidence_is_clear():
    rating, override = rate_quality(
        clipping_ratio=0.0, dropouts_per_min=0.0, rolloff_hz=3000.0,
        speech_rms_dbfs=-20.0, pesq=3.5, snr_db=20.0,
    )
    assert rating == AudioQuality.CLEAR and not override


def test_missing_speech_derived_evidence_is_still_clear():
    """No speech at all -> rolloff_hz/speech_rms_dbfs/pesq/snr_db are all None.
    Absence of a speech-derived measurement must not be treated as adverse
    evidence."""
    rating, override = rate_quality(
        clipping_ratio=0.0, dropouts_per_min=0.0, rolloff_hz=None,
        speech_rms_dbfs=None, pesq=None, snr_db=None,
    )
    assert rating == AudioQuality.CLEAR and not override


def test_clipping_overrides_everything():
    rating, override = rate_quality(
        clipping_ratio=0.08, dropouts_per_min=10.0, rolloff_hz=200.0,
        speech_rms_dbfs=-60.0, pesq=1.0, snr_db=20.0,
    )
    assert rating == AudioQuality.SEVERELY_IMPAIRED and override


def test_dropouts_slightly_impaired_at_2_per_min():
    rating, override = rate_quality(
        clipping_ratio=0.0, dropouts_per_min=2.0, rolloff_hz=3000.0,
        speech_rms_dbfs=-20.0, pesq=3.5, snr_db=20.0,
    )
    assert rating == AudioQuality.SLIGHTLY_IMPAIRED and not override


def test_dropouts_severely_impaired_at_5_per_min():
    rating, _ = rate_quality(
        clipping_ratio=0.0, dropouts_per_min=5.0, rolloff_hz=3000.0,
        speech_rms_dbfs=-20.0, pesq=3.5, snr_db=20.0,
    )
    assert rating == AudioQuality.SEVERELY_IMPAIRED


def test_rolloff_slightly_impaired_at_800hz():
    # rolloff_severe_hz=600.0 / rolloff_slight_hz=900.0 -- see config.py: retuned
    # down from an initial 1200/2200 after real anchor measurement showed all 3
    # labeled-clear calls sit in the 1024-1591Hz range (see task-7-report.md).
    rating, _ = rate_quality(
        clipping_ratio=0.0, dropouts_per_min=0.0, rolloff_hz=800.0,
        speech_rms_dbfs=-20.0, pesq=3.5, snr_db=20.0,
    )
    assert rating == AudioQuality.SLIGHTLY_IMPAIRED


def test_rolloff_severely_impaired_at_400hz():
    rating, _ = rate_quality(
        clipping_ratio=0.0, dropouts_per_min=0.0, rolloff_hz=400.0,
        speech_rms_dbfs=-20.0, pesq=3.5, snr_db=20.0,
    )
    assert rating == AudioQuality.SEVERELY_IMPAIRED


def test_low_volume_slightly_impaired_at_minus_38dbfs():
    rating, _ = rate_quality(
        clipping_ratio=0.0, dropouts_per_min=0.0, rolloff_hz=3000.0,
        speech_rms_dbfs=-38.0, pesq=3.5, snr_db=20.0,
    )
    assert rating == AudioQuality.SLIGHTLY_IMPAIRED


def test_low_volume_severely_impaired_at_minus_50dbfs():
    rating, _ = rate_quality(
        clipping_ratio=0.0, dropouts_per_min=0.0, rolloff_hz=3000.0,
        speech_rms_dbfs=-50.0, pesq=3.5, snr_db=20.0,
    )
    assert rating == AudioQuality.SEVERELY_IMPAIRED


def test_squim_backstop_fires_when_noise_cannot_excuse_bad_pesq():
    rating, override = rate_quality(
        clipping_ratio=0.0, dropouts_per_min=0.0, rolloff_hz=3000.0,
        speech_rms_dbfs=-20.0, pesq=1.1, snr_db=20.0,
    )
    assert rating == AudioQuality.SEVERELY_IMPAIRED and not override


def test_squim_backstop_does_not_fire_when_noise_explains_bad_pesq():
    rating, _ = rate_quality(
        clipping_ratio=0.0, dropouts_per_min=0.0, rolloff_hz=3000.0,
        speech_rms_dbfs=-20.0, pesq=1.1, snr_db=3.0,
    )
    assert rating == AudioQuality.CLEAR


def test_squim_backstop_does_not_fire_when_snr_is_unmeasured():
    """snr_db is None (e.g. VAD found no usable silence to measure noise against) ->
    the backstop cannot prove noise ISN'T the excuse, so it stays off."""
    rating, _ = rate_quality(
        clipping_ratio=0.0, dropouts_per_min=0.0, rolloff_hz=3000.0,
        speech_rms_dbfs=-20.0, pesq=1.1, snr_db=None,
    )
    assert rating == AudioQuality.CLEAR


def test_worst_triggered_level_wins():
    """Dropouts alone would be slightly_impaired; rolloff alone would be
    severely_impaired. The two are independent failure modes -- the worst wins."""
    rating, _ = rate_quality(
        clipping_ratio=0.0, dropouts_per_min=2.0, rolloff_hz=400.0,
        speech_rms_dbfs=-20.0, pesq=3.5, snr_db=20.0,
    )
    assert rating == AudioQuality.SEVERELY_IMPAIRED


# --- synthetic-audio unit tests for the deterministic detectors ---


def test_dropout_detector_counts_hard_zero_runs_in_speech_segment():
    sr = 16000
    t = np.arange(int(2.0 * sr)) / sr
    samples = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def zero_out(start_s: float, dur_s: float) -> None:
        lo, hi = int(start_s * sr), int((start_s + dur_s) * sr)
        samples[lo:hi] = 0.0

    zero_out(0.5, 0.080)  # two 80ms hard-zero gaps, both well inside [0, 2.0]
    zero_out(1.2, 0.080)
    vad = _vad_one_segment(2.0)
    assert _dropout_count(samples, sr, vad) == 2


def test_dropout_detector_ignores_gaps_touching_segment_edges():
    """A near-zero run that touches the segment boundary is a VAD-edge artifact, not
    a mid-speech dropout, and must not be counted."""
    sr = 16000
    t = np.arange(int(2.0 * sr)) / sr
    samples = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    samples[: int(0.1 * sr)] = 0.0  # touches the left edge
    samples[-int(0.1 * sr):] = 0.0  # touches the right edge
    vad = _vad_one_segment(2.0)
    assert _dropout_count(samples, sr, vad) == 0


def test_dropout_detector_ignores_gaps_under_50ms():
    sr = 16000
    t = np.arange(int(1.0 * sr)) / sr
    samples = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    lo = int(0.5 * sr)
    samples[lo: lo + int(0.030 * sr)] = 0.0  # 30ms: below the 50ms floor
    vad = _vad_one_segment(1.0)
    assert _dropout_count(samples, sr, vad) == 0


def test_rolloff_detector_measures_low_rolloff_for_lowpassed_signal():
    sr = 16000
    dur = 2.0
    t = np.arange(int(dur * sr)) / sr
    # Sum of tones all comfortably under 1200Hz -- no energy above it, so a correct
    # rolloff measurement must land below rolloff_severe_hz.
    samples = (
        0.3 * np.sin(2 * np.pi * 300 * t)
        + 0.2 * np.sin(2 * np.pi * 600 * t)
        + 0.1 * np.sin(2 * np.pi * 900 * t)
    ).astype(np.float32)
    vad = _vad_one_segment(dur)
    rolloff = _spectral_rolloff(samples, sr, vad)
    assert rolloff is not None and rolloff < 1200.0


def test_rolloff_detector_measures_high_rolloff_for_wideband_signal():
    sr = 16000
    dur = 2.0
    rng = np.random.default_rng(0)
    samples = (0.3 * rng.standard_normal(int(dur * sr))).astype(np.float32)  # white noise
    vad = _vad_one_segment(dur)
    rolloff = _spectral_rolloff(samples, sr, vad)
    assert rolloff is not None and rolloff > 2200.0


def test_rolloff_detector_none_when_no_speech():
    sr = 16000
    samples = np.zeros(sr, dtype=np.float32)
    vad = _vad_no_speech(1.0)
    assert _spectral_rolloff(samples, sr, vad) is None
