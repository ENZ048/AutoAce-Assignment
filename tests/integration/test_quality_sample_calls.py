import pytest

from autoace_audio.analyzers.noise import snr_db
from autoace_audio.analyzers.quality import analyze_quality
from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import load_audio
from autoace_audio.schema import AudioQuality


@pytest.mark.slow
@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_labeled_clear(sample_calls_dir, name):
    a = load_audio(sample_calls_dir / name)
    vad = analyze_vad(a.samples, a.sr)
    snr = snr_db(a.samples, a.sr, vad)
    q = analyze_quality(a.samples, a.sr, vad, snr)
    msg = (
        f"{name}: pesq={q.pesq} stoi={q.stoi} snr={snr} clip={q.clipping_ratio} "
        f"dropouts/min={q.dropouts_per_min} rolloff_hz={q.rolloff_hz} "
        f"rms_dbfs={q.speech_rms_dbfs}"
    )
    assert q.rating == AudioQuality.CLEAR, msg
