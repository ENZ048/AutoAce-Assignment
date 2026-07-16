from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_calls_dir() -> Path:
    """Real labeled sample calls (gitignored). Tests that need them skip if absent."""
    d = REPO_ROOT / "data"
    if not (d / "call_001.ogg").exists():
        pytest.skip("sample calls not present in data/")
    return d


@pytest.fixture()
def make_wav(tmp_path):
    """Build a synthetic WAV: list of (kind, seconds) blocks, kind in {'tone','silence','noise'}."""

    def _make(blocks, sr: int = 16000, name: str = "fixture.wav", amp: float = 0.3) -> Path:
        rng = np.random.default_rng(42)
        parts = []
        for kind, secs in blocks:
            n = int(secs * sr)
            if kind == "tone":
                t = np.arange(n) / sr
                parts.append(amp * np.sin(2 * np.pi * 220 * t))
            elif kind == "noise":
                parts.append(amp * 0.5 * rng.standard_normal(n))
            else:  # silence
                parts.append(np.zeros(n))
        audio = np.concatenate(parts).astype(np.float32)
        path = tmp_path / name
        sf.write(path, audio, sr)
        return path

    return _make
