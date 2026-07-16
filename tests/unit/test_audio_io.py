from pathlib import Path

import numpy as np
import pytest

from autoace_audio.audio_io import DecodeError, encode_opus_ogg, load_audio


def test_decodes_wav_to_16k_mono_float32(make_wav):
    path = make_wav([("tone", 2.0)], sr=48000)
    a = load_audio(path)
    assert a.sr == 16000
    assert a.samples.dtype == np.float32
    assert abs(a.duration_s - 2.0) < 0.1
    assert a.src_sample_rate == 48000


def test_misnamed_extension_decodes_by_content(make_wav, tmp_path):
    """Our production smoke set has .mp3-named files that are PCM WAV inside."""
    wav = make_wav([("tone", 1.0)])
    lying = tmp_path / "actually_wav.mp3"
    lying.write_bytes(Path(wav).read_bytes())
    a = load_audio(lying)
    assert a.src_codec.startswith("pcm")


def test_garbage_raises_decode_error(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(DecodeError):
        load_audio(bad)


def test_opus_reencode_produces_ogg(make_wav):
    a = load_audio(make_wav([("tone", 1.0)]))
    blob = encode_opus_ogg(a.samples, a.sr)
    assert blob[:4] == b"OggS"
    assert len(blob) < 40_000  # 24kbps mono: tiny


def test_zero_byte_file_raises_decode_error(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(DecodeError):
        load_audio(empty)


def test_subprocess_timeout_surfaces_as_decode_error(make_wav, monkeypatch):
    import subprocess as sp

    def fake_run(*args, **kwargs):
        raise sp.TimeoutExpired(cmd="ffprobe", timeout=1)

    monkeypatch.setattr(sp, "run", fake_run)
    with pytest.raises(DecodeError, match="timed out"):
        load_audio(make_wav([("tone", 0.5)]))
