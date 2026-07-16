from autoace_audio.analyzers.noise import NoiseResult
from autoace_audio.analyzers.quality import QualityResult
from autoace_audio.analyzers.tone.base import ToneResult
from autoace_audio.analyzers.vad import VadMap
from autoace_audio.fusion import fuse
from autoace_audio.schema import (
    AudioQuality,
    EmotionalIntensity,
    EmotionalTone,
    Severity,
)


def _vad(long_silence=False):
    return VadMap([], [], 0.8, 3.0, long_silence, 60.0)


def _noise(present=True, type_label=None, support_s=0.0, support_floor_s=0.0):
    # NoiseResult's real dataclass (analyzers/noise.py): present, type_label,
    # severity, snr_db, top_events, then the task-9 support_s/support_floor_s pair
    # (defaults 0.0 -- unused unless a test is specifically exercising the
    # thin-margin rule).
    label = type_label if type_label is not None else ("TV" if present else "")
    return NoiseResult(
        present,
        label,
        Severity.MEDIUM if present else Severity.NONE,
        12.0,
        [("Television", 0.6)],
        support_s,
        support_floor_s,
    )


def _quality():
    # QualityResult's real v2 dataclass (analyzers/quality.py, task 7 rework):
    # rating, pesq, stoi, si_sdr, clipping_ratio, clipping_override,
    # dropouts_per_min, rolloff_hz, speech_rms_dbfs -- the brief's stale 6-arg stub
    # predates this; adapted here to the current 9-field signature.
    return QualityResult(
        rating=AudioQuality.CLEAR,
        pesq=3.4,
        stoi=0.9,
        si_sdr=18.0,
        clipping_ratio=0.0,
        clipping_override=False,
        dropouts_per_min=0.0,
        rolloff_hz=3000.0,
        speech_rms_dbfs=-20.0,
    )


def _tone(noise_type="TV"):
    return ToneResult(
        EmotionalTone.NEUTRAL,
        EmotionalIntensity.MEDIUM,
        0.8,
        overlap_opinion=True,
        noise_opinion={"present": True, "type": noise_type},
    )


def test_happy_path_fields():
    r = fuse(_vad(), _noise(), _quality(), _tone(), None)
    assert r.emotional_tone == EmotionalTone.NEUTRAL
    assert r.background_noise_present and r.background_noise_type == "TV"
    assert r.background_noise_severity == Severity.MEDIUM
    assert r.speaker_overlap_present is True
    assert 0.05 <= r.confidence <= 0.98


def test_noise_absent_forces_empty_type_and_none_severity():
    # Brief staleness fix (controller amendment C): the brief's own version of this
    # test passed _tone() (whose noise_opinion always claims present=True)
    # alongside _noise(present=False) -- under the brief's own Rule A ("AED
    # absent + LLM present" fills in noise), that combination is SUPPOSED to flip
    # present to True, contradicting this test's own assertion. Isolate the pure
    # invariant instead: no tone arm opinion to contradict AED's "absent" read.
    # Rule A's fill-in behavior has its own dedicated test below.
    r = fuse(_vad(), _noise(present=False), _quality(), None, None)
    assert not r.background_noise_present
    assert r.background_noise_type == "" and r.background_noise_severity == Severity.NONE


def test_tone_failure_degrades_to_neutral_low_confidence():
    r = fuse(_vad(), _noise(), _quality(), None, "gemini timeout")
    assert r.emotional_tone == EmotionalTone.NEUTRAL
    assert r.confidence <= 0.40


def test_gemini_noise_opinion_fills_type_when_aed_uncertain():
    n = NoiseResult(False, "", Severity.NONE, 13.0, [("Television", 0.30)])  # below threshold
    t = _tone()  # gemini says TV present
    r = fuse(_vad(), n, _quality(), t, None)
    assert r.background_noise_present and r.background_noise_type == "TV"
    assert r.background_noise_severity == Severity.MEDIUM  # severity re-derived from SNR 13dB


def test_gemini_present_with_no_type_falls_back_to_concise_labeled_aed_top_event():
    """Rule A's last-resort branch: the LLM says noise IS present but gives no type
    of its own (empty string) -- fall back to AED's own best (unsustained) guess,
    but it must go through concise_label() same as analyze_noise's own type_label
    does, not leak a raw AudioSet class name like "Television" past fusion."""
    n = NoiseResult(False, "", Severity.NONE, 13.0, [("Television", 0.30)])
    t = _tone(noise_type="")  # gemini agrees present, but has no type opinion
    r = fuse(_vad(), n, _quality(), t, None)
    assert r.background_noise_present
    assert r.background_noise_type == "TV"  # concise_label("Television"), not raw


# --- Controller amendment B: thin-margin AED support + LLM type disagreement ---
# CNN14 has no static-family AudioSet class; call_003's real "sharp static" comes
# out "radio" from AED (see task-6-report.md / test_noise_sample_calls.py's
# documented xfail). When AED's sustained support barely clears its own effective
# floor (within one hop, aed_hop_s=2.5s) AND the tone arm's noise_opinion agrees
# noise is present but names a different type, the LLM's type should win -- that's
# exactly the low-evidence case CNN14 structurally cannot get right. Severity stays
# AED/SNR-derived either way: the LLM has no calibrated severity opinion.


def test_thin_margin_and_type_disagreement_prefers_llm_type():
    n = _noise(present=True, type_label="radio", support_s=5.0, support_floor_s=5.0)  # diff 0.0s
    t = _tone(noise_type="static")
    r = fuse(_vad(), n, _quality(), t, None)
    assert r.background_noise_present
    assert r.background_noise_type == "static"
    assert r.background_noise_severity == Severity.MEDIUM  # unchanged: AED/SNR-derived


def test_comfortable_margin_keeps_aed_type_despite_disagreement():
    n = _noise(present=True, type_label="radio", support_s=20.0, support_floor_s=5.0)  # diff 15s
    t = _tone(noise_type="static")
    r = fuse(_vad(), n, _quality(), t, None)
    assert r.background_noise_present
    assert r.background_noise_type == "radio"
    assert r.background_noise_severity == Severity.MEDIUM


def test_matching_type_opinion_is_a_no_op():
    """When the LLM agrees with AED on the type string, the thin-margin rule has
    nothing to decide -- output is simply the (matching) AED type, regardless of
    margin width, and neither presence nor severity are disturbed."""
    n = _noise(present=True, type_label="TV", support_s=5.0, support_floor_s=5.0)
    t = _tone(noise_type="TV")
    r = fuse(_vad(), n, _quality(), t, None)
    assert r.background_noise_present is True
    assert r.background_noise_type == "TV"
    assert r.background_noise_severity == Severity.MEDIUM
