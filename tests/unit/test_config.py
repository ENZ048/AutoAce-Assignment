import pytest

from autoace_audio.config import Settings


def test_defaults_are_calibration_ready(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = Settings(_env_file=None)
    assert s.long_silence_s == 10.0
    assert s.snr_none_db > s.snr_low_db > s.snr_medium_db
    assert s.dropout_high_per_min > s.dropout_low_per_min
    assert s.rolloff_slight_hz > s.rolloff_severe_hz
    assert s.volume_slight_dbfs > s.volume_severe_dbfs
    assert s.tone_arm == "gemini"
    assert s.gemini_model == "gemini-3.1-flash-lite"


def test_env_override(monkeypatch):
    monkeypatch.setenv("LONG_SILENCE_S", "12.5")
    s = Settings(_env_file=None)
    assert s.long_silence_s == 12.5


def test_fusion_confidence_blend_weights_sum_to_one(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = Settings(_env_file=None)
    total = s.conf_w_tone + s.conf_w_noise + s.conf_w_quality
    assert total == pytest.approx(1.0)
