"""Gemini tone arm parse/contract-failure path. The genai client is fully
monkeypatched -- no network, no real audio encode, seconds to run."""

import numpy as np
import pytest

from autoace_audio.analyzers.tone import gemini_tone
from autoace_audio.analyzers.tone.base import ToneClassifierError
from autoace_audio.analyzers.vad import VadMap
from autoace_audio.config import Settings


class _FakeResp:
    """Mimics a safety-blocked Gemini response: text is None, no usage metadata."""

    text = None
    usage_metadata = None


class _FakeModels:
    def generate_content(self, **kwargs):
        return _FakeResp()


class _FakeClient:
    def __init__(self, api_key=None):
        self.models = _FakeModels()


def _fake_settings() -> Settings:
    return Settings(_env_file=None, gemini_api_key="fake-key-for-test")


def _fake_vad() -> VadMap:
    return VadMap(
        speech=[], gaps=[], speech_ratio=0.5, max_gap_s=0.0, long_silence_present=False, total_s=1.0
    )


def test_none_response_text_raises_tone_classifier_error_not_typeerror(monkeypatch):
    """Reviewer finding: a safety-blocked Gemini response has resp.text=None;
    json.loads(None) raises a raw TypeError. That must surface as
    ToneClassifierError (so pipeline.analyze's fallback wiring can catch it),
    never crash the caller with an uncaught TypeError."""
    monkeypatch.setattr(gemini_tone, "get_settings", _fake_settings)
    monkeypatch.setattr(gemini_tone, "encode_opus_ogg", lambda samples, sr: b"fake-ogg-bytes")
    monkeypatch.setattr("google.genai.Client", _FakeClient)

    samples = np.zeros(16000, dtype=np.float32)

    with pytest.raises(ToneClassifierError):
        gemini_tone.classify(samples, sr=16000, vad=_fake_vad(), snr_db=10.0)
