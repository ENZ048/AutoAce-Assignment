"""Pipeline orchestration unit tests. All heavy analyzers (vad/noise/quality/tone)
are monkeypatched at the pipeline module's imported names -- no model loads, no
audio decode, no network. Exercises the fallback-arm wiring and the
tone_arm/tone_arm_used diagnostics contract in isolation from real inference."""

import numpy as np

from autoace_audio import pipeline as pipeline_mod
from autoace_audio.analyzers.noise import NoiseResult
from autoace_audio.analyzers.quality import QualityResult
from autoace_audio.analyzers.tone.base import ToneClassifierError, ToneResult
from autoace_audio.analyzers.vad import VadMap
from autoace_audio.audio_io import DecodedAudio
from autoace_audio.schema import AudioQuality, EmotionalIntensity, EmotionalTone, Severity


def _fake_audio() -> DecodedAudio:
    return DecodedAudio(
        samples=np.zeros(16000, dtype=np.float32),
        sr=16000,
        duration_s=1.0,
        src_codec="pcm_s16le",
        src_sample_rate=16000,
        src_channels=1,
    )


def _fake_vad() -> VadMap:
    return VadMap([], [], 0.8, 0.0, False, 1.0)


def _fake_noise() -> NoiseResult:
    return NoiseResult(False, "", Severity.NONE, 20.0, [])


def _fake_quality() -> QualityResult:
    return QualityResult(AudioQuality.CLEAR, 3.4, 0.9, 18.0, 0.0, False, 0.0, 3000.0, -20.0)


def _patch_common(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "load_audio", lambda p: _fake_audio())
    monkeypatch.setattr(pipeline_mod, "analyze_vad", lambda samples, sr: _fake_vad())
    monkeypatch.setattr(pipeline_mod, "analyze_noise", lambda samples, sr, vad: _fake_noise())
    monkeypatch.setattr(
        pipeline_mod, "analyze_quality", lambda samples, sr, vad, snr: _fake_quality()
    )


def test_tone_arm_used_matches_requested_arm_on_happy_path(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    ok_tone = ToneResult(EmotionalTone.SATISFIED, EmotionalIntensity.MEDIUM, 0.9)
    monkeypatch.setattr(
        pipeline_mod, "classify_tone", lambda arm, samples, sr, vad, snr_db: ok_tone
    )

    out = pipeline_mod.analyze(tmp_path / "fake.ogg", tone_arm="gemini")

    assert out.diagnostics["tone_arm"] == "gemini"
    assert out.diagnostics["tone_arm_used"] == "gemini"
    assert out.diagnostics["tone_error"] is None


def test_tone_arm_used_reports_the_fallback_arm_after_primary_failure(monkeypatch, tmp_path):
    """Reviewer finding: diagnostics must identify the arm that ACTUALLY produced
    the tone result, not just the one requested. Primary ('gemini') fails;
    pipeline.analyze falls back to the local 'dimensional' arm, which succeeds."""
    _patch_common(monkeypatch)
    fallback_tone = ToneResult(EmotionalTone.NEUTRAL, EmotionalIntensity.LOW, 0.5)

    def fake_classify_tone(arm, samples, sr, vad, snr_db):
        if arm == "gemini":
            raise ToneClassifierError("simulated gemini failure")
        assert arm == "dimensional"
        return fallback_tone

    monkeypatch.setattr(pipeline_mod, "classify_tone", fake_classify_tone)

    out = pipeline_mod.analyze(tmp_path / "fake.ogg", tone_arm="gemini")

    assert out.diagnostics["tone_arm"] == "gemini"  # requested
    assert out.diagnostics["tone_arm_used"] == "dimensional"  # actually produced it
    assert out.diagnostics["tone_error"] == "simulated gemini failure"
    assert out.result.emotional_tone == EmotionalTone.NEUTRAL


def test_tone_arm_used_is_none_when_every_arm_fails(monkeypatch, tmp_path):
    _patch_common(monkeypatch)

    def always_fail(arm, samples, sr, vad, snr_db):
        raise ToneClassifierError(f"{arm} failed")

    monkeypatch.setattr(pipeline_mod, "classify_tone", always_fail)

    out = pipeline_mod.analyze(tmp_path / "fake.ogg", tone_arm="gemini")

    assert out.diagnostics["tone_arm"] == "gemini"
    assert out.diagnostics["tone_arm_used"] is None
    assert "gemini failed" in out.diagnostics["tone_error"]
    assert "fallback: dimensional failed" in out.diagnostics["tone_error"]
    # No tone arm at all reached fuse() -- degrades to neutral/low per fusion.py.
    assert out.result.emotional_tone == EmotionalTone.NEUTRAL


def test_analyze_survives_a_raw_non_tone_classifier_exception(monkeypatch, tmp_path):
    """Reviewer finding: dimensional model-load failures raise raw OSError/HF errors,
    not ToneClassifierError. analyze() must degrade gracefully (neutral tone +
    tone_error set, already-computed deterministic fields preserved) instead of
    letting the raw exception crash the whole pipeline (design doc §8: total
    failure only on undecodable audio)."""
    _patch_common(monkeypatch)

    def always_raise_raw(arm, samples, sr, vad, snr_db):
        raise RuntimeError(f"{arm} blew up")

    monkeypatch.setattr(pipeline_mod, "classify_tone", always_raise_raw)

    out = pipeline_mod.analyze(tmp_path / "fake.ogg", tone_arm="gemini")

    assert out.diagnostics["tone_arm"] == "gemini"
    assert out.diagnostics["tone_arm_used"] is None
    assert "gemini blew up" in out.diagnostics["tone_error"]
    assert "fallback: dimensional blew up" in out.diagnostics["tone_error"]
    assert out.result.emotional_tone == EmotionalTone.NEUTRAL


def test_preload_models_touches_each_always_used_singleton(monkeypatch):
    """preload_models must warm exactly the models every batch uses (VAD,
    PANNs tagger, SQUIM) — fallback-only models (dimensional tone, whisper)
    stay lazy so a preload never downloads what a batch may never need."""
    from autoace_audio import pipeline
    from autoace_audio.analyzers import noise, quality, vad

    called = []
    monkeypatch.setattr(vad, "_model", lambda: called.append("vad"))
    monkeypatch.setattr(noise, "_tagger", lambda: called.append("panns"))
    monkeypatch.setattr(quality, "_squim", lambda: called.append("squim"))
    pipeline.preload_models()
    assert set(called) == {"vad", "panns", "squim"}
