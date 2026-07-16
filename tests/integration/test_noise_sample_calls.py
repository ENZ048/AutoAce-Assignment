import pytest

from autoace_audio.analyzers.noise import analyze_noise
from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import load_audio


@pytest.mark.slow
def test_call_002_has_noise_call_001_does_not(sample_calls_dir):
    a1 = load_audio(sample_calls_dir / "call_001.ogg")
    n1 = analyze_noise(a1.samples, a1.sr, analyze_vad(a1.samples, a1.sr))
    a2 = load_audio(sample_calls_dir / "call_002.ogg")
    n2 = analyze_noise(a2.samples, a2.sr, analyze_vad(a2.samples, a2.sr))
    assert not n1.present, f"001 labeled no-noise; got {n1.top_events[:3]}"
    assert n2.present, f"002 labeled TV/medium; got {n2.top_events[:3]}"
