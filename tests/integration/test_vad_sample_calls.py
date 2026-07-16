import pytest

from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import load_audio


@pytest.mark.slow
@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_labels_say_no_long_silence(sample_calls_dir, name):
    a = load_audio(sample_calls_dir / name)
    m = analyze_vad(a.samples, a.sr)
    assert not m.long_silence_present  # all three labeled false
    assert m.speech_ratio > 0.3
