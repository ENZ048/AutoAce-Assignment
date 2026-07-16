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
