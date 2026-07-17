from autoace_audio.analyzers.tone.gemini_tone import GEMINI_RESPONSE_SCHEMA, build_prompt


def test_prompt_targets_customer_not_agent():
    p = build_prompt(duration_s=45.0, snr_db=28.0, speech_ratio=0.8)
    assert "customer" in p.lower()
    assert "erica" in p.lower() or "ai agent" in p.lower()
    assert "loud" in p.lower()  # explicit do-not-infer-from-loudness instruction
    assert "28.0 dB" in p


def test_response_schema_enums_match_brief():
    props = GEMINI_RESPONSE_SCHEMA["properties"]
    assert props["emotional_tone"]["enum"] == [
        "neutral",
        "satisfied",
        "frustrated",
        "upset",
        "distressed",
    ]
    assert props["emotional_intensity"]["enum"] == ["low", "medium", "high"]
    assert set(GEMINI_RESPONSE_SCHEMA["required"]) >= {
        "emotional_tone",
        "emotional_intensity",
        "tone_confidence",
        "background_noise_present",
        "background_noise_type",
        "speaker_overlap_present",
    }


def test_client_is_constructed_with_hard_timeout(monkeypatch):
    """A single pathological clip must never wedge a batch: the client carries a
    bounded per-request timeout (stress-batch finding, 2026-07-17)."""
    import google.genai as genai

    from autoace_audio.analyzers.tone.gemini_tone import _make_client
    from autoace_audio.config import Settings

    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(genai, "Client", fake_client)
    s = Settings(_env_file=None, gemini_api_key="test-key", gemini_timeout_s=60.0)
    _make_client(s)
    assert captured["http_options"] == {"timeout": 60000}
