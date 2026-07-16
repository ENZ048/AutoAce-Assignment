import json

import pytest
from pydantic import ValidationError

from autoace_audio.schema import AnalysisResult


def _valid(**over):
    base = dict(
        emotional_tone="frustrated",
        emotional_intensity="medium",
        background_noise_present=True,
        background_noise_type="office chatter",
        background_noise_severity="low",
        audio_quality="clear",
        speaker_overlap_present=False,
        long_silence_present=False,
        confidence=0.82,
    )
    base.update(over)
    return AnalysisResult(**base)


def test_field_order_matches_brief_example():
    keys = list(json.loads(_valid().to_result_json()).keys())
    assert keys == [
        "emotional_tone", "emotional_intensity", "background_noise_present",
        "background_noise_type", "background_noise_severity", "audio_quality",
        "speaker_overlap_present", "long_silence_present", "confidence",
    ]


def test_rejects_unknown_enum_value():
    with pytest.raises(ValidationError):
        _valid(emotional_tone="angry")


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        _valid(confidence=1.3)


def test_round_trip():
    r = _valid()
    assert AnalysisResult.model_validate_json(r.to_result_json()) == r
