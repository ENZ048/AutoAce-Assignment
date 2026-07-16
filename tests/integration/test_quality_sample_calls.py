import pytest

from autoace_audio.analyzers.quality import analyze_quality
from autoace_audio.audio_io import load_audio
from autoace_audio.schema import AudioQuality


@pytest.mark.slow
@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_labeled_clear(sample_calls_dir, name):
    a = load_audio(sample_calls_dir / name)
    q = analyze_quality(a.samples, a.sr)
    msg = f"{name}: pesq={q.pesq} stoi={q.stoi} clip={q.clipping_ratio}"
    assert q.rating == AudioQuality.CLEAR, msg
