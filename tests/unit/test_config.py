from autoace_audio.config import Settings


def test_defaults_are_calibration_ready(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = Settings(_env_file=None)
    assert s.long_silence_s == 10.0
    assert s.snr_none_db > s.snr_low_db > s.snr_medium_db
    assert s.pesq_clear > s.pesq_slight
    assert s.tone_arm == "gemini"
    assert s.gemini_model == "gemini-3.1-flash-lite"


def test_env_override(monkeypatch):
    monkeypatch.setenv("LONG_SILENCE_S", "12.5")
    s = Settings(_env_file=None)
    assert s.long_silence_s == 12.5
