import pytest

from autoace_audio.analyzers.tone.base import classify_tone
from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import load_audio

EXPECTED = {  # name -> labeled tone
    "call_001.ogg": "upset",
    "call_002.ogg": "neutral",
    "call_003.ogg": "satisfied",
}


@pytest.mark.network
@pytest.mark.parametrize("name,tone", EXPECTED.items())
def test_gemini_arm_matches_labels(sample_calls_dir, name, tone):
    a = load_audio(sample_calls_dir / name)
    vad = analyze_vad(a.samples, a.sr)
    r = classify_tone("gemini", a.samples, a.sr, vad, snr_db=None)
    assert r.tone.value == tone, f"{name}: got {r.tone.value} raw={r.raw.get('response')}"


@pytest.mark.slow
def test_dimensional_arm_runs_and_orders_sensibly(sample_calls_dir):
    results = {}
    for name in EXPECTED:
        a = load_audio(sample_calls_dir / name)
        vad = analyze_vad(a.samples, a.sr)
        results[name] = classify_tone("dimensional", a.samples, a.sr, vad, snr_db=None)
    # weaker assertion: upset call must not score higher valence than satisfied call
    assert results["call_001.ogg"].raw["valence"] < results["call_003.ogg"].raw["valence"]
