# AutoAce Call-Audio Analysis Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python package that analyzes one call recording into the AutoAce 9-field JSON, plus a batch CLI over folder+manifest, an evaluation harness, and a 3-arm tone bake-off.

**Architecture:** Deterministic local layer (ffmpeg ingest → silero-VAD → PANNs AED + SNR → SQUIM + clipdetect) feeds a fusion module; emotional tone comes from a swappable `ToneClassifier` (Gemini audio / audeering dimensional / whisper+text-LLM). `pipeline.analyze()` is the single public entry point; `batch.py` wraps it with per-file failure isolation.

**Tech Stack:** Python 3.12, pydantic v2, torch/torchaudio (CPU), silero-vad, panns-inference, transformers (audeering wav2vec2), torchaudio SQUIM, clipdetect, google-genai (`gemini-3.1-flash-lite`), faster-whisper + openai (Arm C), pytest + ruff, ffmpeg (system).

## Global Constraints

- Cost ceiling: final pipeline ≤ $0.003 per audio-minute (Gemini 3.1 Flash-Lite ≈ $0.0011–0.0016/min; audio billed at 32 tokens/sec).
- `data/` and `.env` are gitignored — production audio and keys must NEVER be committed. Verify `git status` before every commit.
- Audio format detected by content (ffprobe), never by extension.
- Enum values byte-exact per brief: tones `neutral|satisfied|frustrated|upset|distressed`; intensity `low|medium|high`; severity `none|low|medium|high`; quality `clear|slightly_impaired|severely_impaired`.
- Output JSON field order matches the brief's example exactly.
- Noise evidence and quality evidence never cross-contaminate; loudness alone never drives emotion.
- All thresholds live in `config.py` with a calibration comment; no magic numbers inline.
- Every commit: conventional message, tests green (`make test`), Co-Authored-By footer.
- Python: type hints everywhere; `ruff check` clean.
- Tests requiring model downloads → `@pytest.mark.slow`; tests requiring API keys/network → `@pytest.mark.network`. Default `make test` runs neither.

---

### Task 1: Scaffold

**Files:**
- Create: `pyproject.toml`, `Makefile`, `README.md`, `.env.example`, `src/autoace_audio/__init__.py`, `tests/conftest.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`

**Interfaces:**
- Produces: installable package `autoace_audio`; `make setup|test|lint`; pytest markers `slow`, `network`; conftest fixture `sample_calls_dir` and audio-fixture builder `make_wav`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "autoace-audio"
version = "0.1.0"
description = "AutoAce technical trial: emotional tone + background noise analysis for production call audio"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pydantic>=2.7",
    "pydantic-settings>=2.2",
    "torch>=2.2",
    "torchaudio>=2.2",
    "silero-vad>=5.1",
    "panns-inference>=0.1.1",
    "transformers>=4.44",
    "faster-whisper>=1.0",
    "google-genai>=1.0",
    "openai>=1.40",
    "clipdetect>=0.1.4",
    "soundfile>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-timeout>=2.3", "ruff>=0.5"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = [
    "slow: needs local model download/inference",
    "network: needs API keys / internet",
]
addopts = "-ra"

[tool.ruff]
line-length = 100
src = ["src", "tests", "eval"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

- [ ] **Step 2: Write `Makefile`**

```makefile
PY := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest

setup:
	python3.12 -m venv .venv || python3 -m venv .venv
	$(PIP) install -q -U pip
	$(PIP) install -q -e ".[dev]"
	@command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found on PATH — required at runtime"

test:
	$(PYTEST) -q -m "not slow and not network"

test-all:
	$(PYTEST) -q

lint:
	.venv/bin/ruff check src tests eval
	.venv/bin/ruff format --check src tests eval

analyze:
	$(PY) -m autoace_audio analyze $(DIR) --out out/

evaluate:
	$(PY) -m eval.evaluate --pred out/results.json --labels data/labels.csv

bakeoff:
	$(PY) -m eval.bakeoff --data data/ --out out/bakeoff.md
```

- [ ] **Step 3: Write `src/autoace_audio/__init__.py`**

```python
"""AutoAce call-audio analysis engine."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Write `.env.example`**

```bash
# Copy to .env and fill in. NEVER commit .env.
GEMINI_API_KEY=            # paid-tier key (free tier trains on data — not allowed for call audio)
OPENAI_API_KEY=            # optional: tone Arm C bake-off only
DEEPGRAM_API_KEY=          # optional: bake-off STT comparison only
SONIOX_API_KEY=            # optional: bake-off STT comparison only
```

- [ ] **Step 5: Write `tests/conftest.py`**

```python
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
```

- [ ] **Step 6: Create empty `tests/unit/__init__.py`, `tests/integration/__init__.py`, minimal `README.md`**

`README.md` (stub — completed in Task 12):

```markdown
# AutoAce Call-Audio Analysis

Analyzes production call audio into a structured 9-field JSON: emotional tone,
background noise, technical quality, speaker overlap, long silences, confidence.

## Quickstart

    make setup
    cp .env.example .env   # add GEMINI_API_KEY (paid tier)
    make analyze DIR=data/

Full architecture: `docs/superpowers/specs/2026-07-16-autoace-backend-design.md`.
```

- [ ] **Step 7: Install and verify**

Run: `make setup && make test`
Expected: install succeeds; pytest reports "no tests ran" (exit 5 is acceptable at this stage).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml Makefile README.md .env.example src tests
git status   # verify: no data/, no .env
git commit -m "chore: scaffold package, tooling, and test fixtures"
```

---

### Task 2: Output schema

**Files:**
- Create: `src/autoace_audio/schema.py`
- Test: `tests/unit/test_schema.py`

**Interfaces:**
- Produces: `EmotionalTone`, `EmotionalIntensity`, `Severity`, `AudioQuality` (str enums); `AnalysisResult` (pydantic model, 9 fields, brief-ordered); `AnalysisResult.to_result_json() -> str` (compact JSON, exact field order); `FileError(name: str, error: str)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_schema.py
import json

import pytest
from pydantic import ValidationError

from autoace_audio.schema import AnalysisResult


def _valid(**over):
    base = dict(
        emotional_tone="frustrated",
        emotional_intensity="medium",
        background_noise_present=True,
        background_noise_type="office chatter",
        background_noise_severity="low",
        audio_quality="clear",
        speaker_overlap_present=False,
        long_silence_present=False,
        confidence=0.82,
    )
    base.update(over)
    return AnalysisResult(**base)


def test_field_order_matches_brief_example():
    keys = list(json.loads(_valid().to_result_json()).keys())
    assert keys == [
        "emotional_tone", "emotional_intensity", "background_noise_present",
        "background_noise_type", "background_noise_severity", "audio_quality",
        "speaker_overlap_present", "long_silence_present", "confidence",
    ]


def test_rejects_unknown_enum_value():
    with pytest.raises(ValidationError):
        _valid(emotional_tone="angry")


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        _valid(confidence=1.3)


def test_round_trip():
    r = _valid()
    assert AnalysisResult.model_validate_json(r.to_result_json()) == r
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_schema.py -q`
Expected: FAIL — `ModuleNotFoundError: autoace_audio.schema`

- [ ] **Step 3: Write `src/autoace_audio/schema.py`**

```python
"""Output contract for the AutoAce trial. Enum values are byte-exact per the brief."""

from enum import Enum

from pydantic import BaseModel, Field


class EmotionalTone(str, Enum):
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    FRUSTRATED = "frustrated"
    UPSET = "upset"
    DISTRESSED = "distressed"


class EmotionalIntensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AudioQuality(str, Enum):
    CLEAR = "clear"
    SLIGHTLY_IMPAIRED = "slightly_impaired"
    SEVERELY_IMPAIRED = "severely_impaired"


class AnalysisResult(BaseModel):
    """One clip's analysis. Field order mirrors the brief's example output."""

    emotional_tone: EmotionalTone
    emotional_intensity: EmotionalIntensity
    background_noise_present: bool
    background_noise_type: str
    background_noise_severity: Severity
    audio_quality: AudioQuality
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float = Field(ge=0.0, le=1.0)

    def to_result_json(self) -> str:
        return self.model_dump_json()


class FileError(BaseModel):
    """Per-file failure record for batch runs."""

    name: str
    error: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_schema.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/autoace_audio/schema.py tests/unit/test_schema.py
git commit -m "feat(schema): output contract with brief-exact enums and field order"
```

---

### Task 3: Configuration

**Files:**
- Create: `src/autoace_audio/config.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `Settings` (pydantic-settings, reads `.env`) and module accessor `get_settings() -> Settings` (cached). Fields used later: `gemini_api_key`, `gemini_model`, `openai_api_key`, `openai_model`, `long_silence_s`, `snr_none_db`, `snr_low_db`, `snr_medium_db`, `aed_prob_threshold`, `aed_min_support_s`, `pesq_clear`, `pesq_slight`, `stoi_floor`, `clipping_ratio_max`, `va_*` thresholds, `intensity_a_low`, `intensity_a_high`, `tone_arm`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_config.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Write `src/autoace_audio/config.py`**

```python
"""All tunable thresholds, with calibration rationale. Values are initial and
revisited by eval/ against the labeled + augmented validation set."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- API keys / models ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"  # verified available; ~$0.0011-0.0016/audio-min
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"  # Arm C bake-off only
    tone_arm: str = "gemini"  # gemini | dimensional | transcript

    # --- VAD / silence ---
    # AutoAce labeled a 7.4s dead-air stretch long_silence=false -> bar is above that.
    long_silence_s: float = 10.0
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 300

    # --- Noise severity via SNR (speech RMS vs non-speech RMS, dB) ---
    snr_none_db: float = 20.0   # > this: no meaningful interference
    snr_low_db: float = 15.0    # (low..none]: audible, doesn't interfere
    snr_medium_db: float = 5.0  # (medium..low]: occasionally interferes; <= : high

    # --- AED (PANNs CNN14) ---
    aed_prob_threshold: float = 0.35  # per user's converging research doc
    aed_min_support_s: float = 2.0    # sustained evidence, not a blip

    # --- Quality (SQUIM + clipdetect) ---
    # Initial bands; the 3 sample calls are all labeled "clear" -> calibrate against them.
    pesq_clear: float = 3.0
    pesq_slight: float = 2.0
    stoi_floor: float = 0.75          # below this, degrade one level
    clipping_ratio_max: float = 0.02  # >2% clipped frames -> severely_impaired override

    # --- Dimensional tone mapping (audeering A/V/D in [0,1]) — initial, calibrated in eval ---
    va_satisfied_v: float = 0.60
    va_upset_v: float = 0.40
    va_upset_a: float = 0.60
    va_distressed_v: float = 0.30
    va_distressed_a: float = 0.75
    va_frustrated_v: float = 0.45
    va_frustrated_a_min: float = 0.40
    intensity_a_low: float = 0.45
    intensity_a_high: float = 0.65

    # --- Fusion confidence ---
    confidence_floor: float = 0.05
    confidence_ceiling: float = 0.98
    tone_degraded_confidence_cap: float = 0.40


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_config.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/autoace_audio/config.py tests/unit/test_config.py
git commit -m "feat(config): centralized calibratable thresholds with rationale"
```

---

### Task 4: Audio ingest

**Files:**
- Create: `src/autoace_audio/audio_io.py`
- Test: `tests/unit/test_audio_io.py`

**Interfaces:**
- Produces: `DecodedAudio` dataclass — `samples: np.ndarray` (float32 mono 16 kHz), `sr: int` (always 16000), `duration_s: float`, `src_codec: str`, `src_sample_rate: int`, `src_channels: int`; `DecodeError(Exception)`; `load_audio(path: Path) -> DecodedAudio`; `encode_opus_ogg(samples, sr) -> bytes` (for Gemini upload).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_audio_io.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_audio_io.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Write `src/autoace_audio/audio_io.py`**

```python
"""ffmpeg-based ingest. Format is detected from CONTENT (ffprobe), never the extension.
No temp files: decode via subprocess pipes (pattern proven in our production backend)."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_SR = 16000


class DecodeError(Exception):
    pass


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray  # float32 mono @ 16 kHz
    sr: int
    duration_s: float
    src_codec: str
    src_sample_rate: int
    src_channels: int


def _probe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise DecodeError(f"ffprobe failed: {proc.stderr.decode(errors='replace')[:200]}")
    streams = json.loads(proc.stdout or b"{}").get("streams") or []
    if not streams:
        raise DecodeError("no audio stream found")
    return streams[0]


def load_audio(path: Path) -> DecodedAudio:
    path = Path(path)
    if not path.is_file():
        raise DecodeError(f"file not found: {path}")
    info = _probe(path)
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", str(TARGET_SR), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0 or not proc.stdout:
        raise DecodeError(f"ffmpeg decode failed: {proc.stderr.decode(errors='replace')[:200]}")
    samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if samples.size == 0:
        raise DecodeError("decoded zero samples")
    return DecodedAudio(
        samples=samples,
        sr=TARGET_SR,
        duration_s=samples.size / TARGET_SR,
        src_codec=str(info.get("codec_name", "unknown")),
        src_sample_rate=int(info.get("sample_rate", 0) or 0),
        src_channels=int(info.get("channels", 0) or 0),
    )


def encode_opus_ogg(samples: np.ndarray, sr: int, bitrate: str = "24k") -> bytes:
    """Compact upload payload for the Gemini API (billing is per-second, not per-byte)."""
    cmd = [
        "ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
        "-c:a", "libopus", "-b:a", bitrate, "-f", "ogg", "pipe:1",
    ]
    proc = subprocess.run(cmd, input=samples.astype(np.float32).tobytes(),
                          capture_output=True, timeout=300)
    if proc.returncode != 0 or not proc.stdout:
        raise DecodeError(f"opus encode failed: {proc.stderr.decode(errors='replace')[:200]}")
    return proc.stdout
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_audio_io.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/autoace_audio/audio_io.py tests/unit/test_audio_io.py
git commit -m "feat(audio-io): content-sniffing ffmpeg ingest and opus re-encode"
```

---

### Task 5: VAD + long-silence

**Files:**
- Create: `src/autoace_audio/analyzers/__init__.py` (empty), `src/autoace_audio/analyzers/vad.py`
- Test: `tests/unit/test_vad_logic.py`, `tests/integration/test_vad_sample_calls.py`

**Interfaces:**
- Consumes: `DecodedAudio` fields (`samples`, `sr`).
- Produces: `Segment` (NamedTuple `start: float, end: float`); `VadMap` dataclass — `speech: list[Segment]`, `gaps: list[Segment]` (non-speech incl. leading/trailing), `speech_ratio: float`, `max_gap_s: float`, `long_silence_present: bool`, `total_s: float`; pure function `build_vad_map(speech: list[Segment], total_s: float, long_silence_s: float) -> VadMap`; model wrapper `analyze_vad(samples, sr) -> VadMap`.

- [ ] **Step 1: Write the failing unit test (pure logic, no model)**

```python
# tests/unit/test_vad_logic.py
from autoace_audio.analyzers.vad import Segment, build_vad_map


def test_gaps_include_leading_and_trailing():
    m = build_vad_map([Segment(5.0, 10.0)], total_s=20.0, long_silence_s=10.0)
    assert m.gaps == [Segment(0.0, 5.0), Segment(10.0, 20.0)]
    assert m.max_gap_s == 10.0
    assert m.long_silence_present  # trailing 10s gap hits threshold


def test_seven_second_gap_is_not_long_silence():
    """AutoAce labeled a 7.4s mid-call gap as long_silence=false."""
    m = build_vad_map(
        [Segment(0.0, 100.0), Segment(107.4, 170.0)], total_s=170.0, long_silence_s=10.0
    )
    assert not m.long_silence_present
    assert abs(m.max_gap_s - 7.4) < 1e-6


def test_no_speech_at_all_is_one_giant_gap():
    m = build_vad_map([], total_s=30.0, long_silence_s=10.0)
    assert m.speech_ratio == 0.0
    assert m.long_silence_present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_vad_logic.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Write `src/autoace_audio/analyzers/vad.py`**

```python
"""Speech/non-speech timeline via silero-vad; long-silence per calibrated threshold."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from autoace_audio.config import get_settings


class Segment(NamedTuple):
    start: float
    end: float


@dataclass(frozen=True)
class VadMap:
    speech: list[Segment]
    gaps: list[Segment]
    speech_ratio: float
    max_gap_s: float
    long_silence_present: bool
    total_s: float


def build_vad_map(speech: list[Segment], total_s: float, long_silence_s: float) -> VadMap:
    """Pure logic: derive gaps (incl. leading/trailing) and the long-silence flag."""
    speech = sorted(speech)
    gaps: list[Segment] = []
    cursor = 0.0
    for seg in speech:
        if seg.start > cursor:
            gaps.append(Segment(cursor, seg.start))
        cursor = max(cursor, seg.end)
    if total_s > cursor:
        gaps.append(Segment(cursor, total_s))
    max_gap = max((g.end - g.start for g in gaps), default=0.0)
    speech_s = sum(s.end - s.start for s in speech)
    return VadMap(
        speech=speech,
        gaps=gaps,
        speech_ratio=(speech_s / total_s) if total_s > 0 else 0.0,
        max_gap_s=max_gap,
        long_silence_present=max_gap >= long_silence_s,
        total_s=total_s,
    )


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from silero_vad import load_silero_vad

        _MODEL = load_silero_vad()
    return _MODEL


def analyze_vad(samples: np.ndarray, sr: int) -> VadMap:
    import torch
    from silero_vad import get_speech_timestamps

    s = get_settings()
    ts = get_speech_timestamps(
        torch.from_numpy(samples),
        _model(),
        sampling_rate=sr,
        min_speech_duration_ms=s.vad_min_speech_ms,
        min_silence_duration_ms=s.vad_min_silence_ms,
        return_seconds=True,
    )
    speech = [Segment(float(t["start"]), float(t["end"])) for t in ts]
    return build_vad_map(speech, total_s=samples.size / sr, long_silence_s=s.long_silence_s)
```

- [ ] **Step 4: Run unit test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_vad_logic.py -q`
Expected: 3 passed

- [ ] **Step 5: Write the integration test against the labeled calls**

```python
# tests/integration/test_vad_sample_calls.py
import pytest

from autoace_audio.audio_io import load_audio
from autoace_audio.analyzers.vad import analyze_vad


@pytest.mark.slow
@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_labels_say_no_long_silence(sample_calls_dir, name):
    a = load_audio(sample_calls_dir / name)
    m = analyze_vad(a.samples, a.sr)
    assert not m.long_silence_present  # all three labeled false
    assert m.speech_ratio > 0.3
```

- [ ] **Step 6: Run integration test**

Run: `.venv/bin/pytest tests/integration/test_vad_sample_calls.py -q -m slow`
Expected: 3 passed (first run downloads the silero model)

- [ ] **Step 7: Commit**

```bash
git add src/autoace_audio/analyzers tests/unit/test_vad_logic.py tests/integration/test_vad_sample_calls.py
git commit -m "feat(vad): silero speech timeline with calibrated long-silence detection"
```

---

### Task 6: Noise — AED + SNR severity

**Files:**
- Create: `src/autoace_audio/analyzers/noise.py`
- Test: `tests/unit/test_noise_logic.py`, `tests/integration/test_noise_sample_calls.py`

**Interfaces:**
- Consumes: `VadMap`, `Segment` from Task 5.
- Produces: `NoiseResult` dataclass — `present: bool`, `type_label: str`, `severity: Severity`, `snr_db: float | None`, `top_events: list[tuple[str, float]]`; pure functions `snr_db(samples, sr, vad: VadMap) -> float | None`, `severity_from_snr(snr: float | None, present: bool) -> Severity`, `concise_label(audioset_class: str) -> str`; model wrapper `analyze_noise(samples, sr, vad) -> NoiseResult`.

- [ ] **Step 1: Write the failing unit test**

```python
# tests/unit/test_noise_logic.py
import numpy as np

from autoace_audio.analyzers.noise import concise_label, severity_from_snr, snr_db
from autoace_audio.analyzers.vad import Segment, VadMap
from autoace_audio.schema import Severity


def _vad(speech, gaps, total):
    return VadMap(speech=speech, gaps=gaps, speech_ratio=0.5, max_gap_s=0.0,
                  long_silence_present=False, total_s=total)


def test_snr_loud_speech_quiet_gap_is_high_db():
    sr = 16000
    speech = 0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    gap = 0.005 * np.random.default_rng(0).standard_normal(sr)
    samples = np.concatenate([speech, gap]).astype(np.float32)
    v = _vad([Segment(0.0, 1.0)], [Segment(1.0, 2.0)], 2.0)
    assert snr_db(samples, sr, v) > 20


def test_snr_none_when_no_gaps():
    sr = 16000
    samples = np.ones(sr, dtype=np.float32) * 0.1
    v = _vad([Segment(0.0, 1.0)], [], 1.0)
    assert snr_db(samples, sr, v) is None


def test_severity_mapping_and_presence_invariant():
    assert severity_from_snr(25.0, present=False) == Severity.NONE
    assert severity_from_snr(25.0, present=True) == Severity.LOW  # present => at least low
    assert severity_from_snr(10.0, present=True) == Severity.MEDIUM
    assert severity_from_snr(2.0, present=True) == Severity.HIGH
    assert severity_from_snr(None, present=True) == Severity.LOW


def test_concise_labels():
    assert concise_label("Television") == "TV"
    assert concise_label("Hubbub, speech noise, speech babble") == "office chatter"
    assert concise_label("Vehicle") == "road noise"
    assert concise_label("SomeUnknownClass") == "someunknownclass"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_noise_logic.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Write `src/autoace_audio/analyzers/noise.py`**

```python
"""Background noise: WHAT (PANNs CNN14 AED, speech classes masked, on non-speech
segments) and HOW MUCH (SNR of speech vs non-speech RMS). Never inferred from
technical quality — the brief scores those independently."""

from dataclasses import dataclass

import numpy as np

from autoace_audio.analyzers.vad import VadMap
from autoace_audio.config import get_settings
from autoace_audio.schema import Severity

# AudioSet classes that describe the conversation itself — never background noise.
MASKED_CLASSES = {
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue",
    "Speech synthesizer", "Shout", "Yell", "Whispering", "Throat clearing",
    "Breathing", "Sigh", "Gasp", "Cough", "Sneeze", "Silence", "Inside, small room",
    "Inside, large room or hall", "Telephone", "Telephone bell ringing",
    "Telephone dialing, DTMF", "Dial tone",
}

# AudioSet label -> concise human label per the brief's examples.
CONCISE = {
    "Television": "TV",
    "Radio": "radio",
    "Music": "music",
    "Background music": "music",
    "Hubbub, speech noise, speech babble": "office chatter",
    "Chatter": "office chatter",
    "Crowd": "crowd noise",
    "Vehicle": "road noise",
    "Car": "road noise",
    "Traffic noise, roadway noise": "road noise",
    "Motor vehicle (road)": "road noise",
    "Typing": "keyboard typing",
    "Computer keyboard": "keyboard typing",
    "Wind": "wind",
    "Wind noise (microphone)": "wind",
    "Static": "static",
    "White noise": "static",
    "Pink noise": "static",
    "Hum": "electrical hum",
    "Mains hum": "electrical hum",
    "Air conditioning": "air conditioning",
    "Mechanical fan": "fan noise",
    "Engine": "engine noise",
    "Dog": "dog barking",
    "Bark": "dog barking",
    "Baby cry, infant cry": "baby crying",
    "Crying, sobbing": "crying",
    "Siren": "siren",
    "Alarm": "alarm",
}


@dataclass(frozen=True)
class NoiseResult:
    present: bool
    type_label: str
    severity: Severity
    snr_db: float | None
    top_events: list[tuple[str, float]]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _slice(samples: np.ndarray, sr: int, segments) -> np.ndarray:
    parts = [samples[int(s.start * sr): int(s.end * sr)] for s in segments]
    return np.concatenate(parts) if parts else np.empty(0, dtype=samples.dtype)


def snr_db(samples: np.ndarray, sr: int, vad: VadMap) -> float | None:
    speech = _slice(samples, sr, vad.speech)
    gaps = _slice(samples, sr, vad.gaps)
    if speech.size == 0 or gaps.size < int(0.3 * sr):  # need >=300ms of gap evidence
        return None
    p_speech, p_noise = _rms(speech), _rms(gaps)
    if p_noise <= 1e-8:
        return 60.0
    return float(20.0 * np.log10(max(p_speech, 1e-8) / p_noise))


def severity_from_snr(snr: float | None, present: bool) -> Severity:
    if not present:
        return Severity.NONE
    s = get_settings()
    if snr is None:
        return Severity.LOW  # noise detected but unmeasurable -> conservative
    if snr <= s.snr_medium_db:
        return Severity.HIGH
    if snr <= s.snr_low_db:
        return Severity.MEDIUM
    return Severity.LOW  # present => never "none" (brief: none means no meaningful noise)


def concise_label(audioset_class: str) -> str:
    return CONCISE.get(audioset_class, audioset_class.split(",")[0].strip().lower())


_TAGGER = None


def _tagger():
    global _TAGGER
    if _TAGGER is None:
        from panns_inference import AudioTagging

        _TAGGER = AudioTagging(checkpoint_path=None, device="cpu")
    return _TAGGER


def _audioset_labels() -> list[str]:
    from panns_inference import labels

    return list(labels)


def analyze_noise(samples: np.ndarray, sr: int, vad: VadMap) -> NoiseResult:
    import torchaudio.functional as F
    import torch

    s = get_settings()
    gap_audio = _slice(samples, sr, vad.gaps)
    source = gap_audio if gap_audio.size >= int(s.aed_min_support_s * sr) else samples
    audio32 = F.resample(torch.from_numpy(source), sr, 32000).numpy()[None, :]
    clipwise, _ = _tagger().inference(audio32)
    probs = clipwise[0]
    names = _audioset_labels()
    ranked = sorted(
        ((names[i], float(p)) for i, p in enumerate(probs) if names[i] not in MASKED_CLASSES),
        key=lambda t: t[1], reverse=True,
    )
    top = ranked[:5]
    present = bool(top and top[0][1] >= s.aed_prob_threshold)
    snr = snr_db(samples, sr, vad)
    return NoiseResult(
        present=present,
        type_label=concise_label(top[0][0]) if present else "",
        severity=severity_from_snr(snr, present),
        snr_db=snr,
        top_events=top,
    )
```

- [ ] **Step 4: Run unit test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_noise_logic.py -q`
Expected: 4 passed

- [ ] **Step 5: Write integration test (labels: 001 no noise; 002 TV medium; 003 static medium)**

```python
# tests/integration/test_noise_sample_calls.py
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
```

- [ ] **Step 6: Run integration test; record findings**

Run: `.venv/bin/pytest tests/integration/test_noise_sample_calls.py -q -m slow`
Expected: 2 assertions pass (first run downloads CNN14 ~300MB to `~/panns_data/`).
If either fails, print `top_events` and tune `aed_prob_threshold` / extend `MASKED_CLASSES` — record the change and reason in `docs/decisions.md`.

- [ ] **Step 7: Commit**

```bash
git add src/autoace_audio/analyzers/noise.py tests/unit/test_noise_logic.py tests/integration/test_noise_sample_calls.py
git commit -m "feat(noise): PANNs AED with speech masking + VAD-segmented SNR severity"
```

---

### Task 7: Quality — SQUIM + clipdetect

**Files:**
- Create: `src/autoace_audio/analyzers/quality.py`
- Test: `tests/unit/test_quality_logic.py`, `tests/integration/test_quality_sample_calls.py`

**Interfaces:**
- Consumes: `DecodedAudio` fields.
- Produces: `QualityResult` dataclass — `rating: AudioQuality`, `pesq: float | None`, `stoi: float | None`, `si_sdr: float | None`, `clipping_ratio: float`, `clipping_override: bool`; pure function `rate_quality(pesq, stoi, clipping_ratio) -> tuple[AudioQuality, bool]`; wrapper `analyze_quality(samples, sr) -> QualityResult`.

- [ ] **Step 1: Write the failing unit test**

```python
# tests/unit/test_quality_logic.py
from autoace_audio.analyzers.quality import rate_quality
from autoace_audio.schema import AudioQuality


def test_high_pesq_good_stoi_is_clear():
    rating, override = rate_quality(pesq=3.5, stoi=0.9, clipping_ratio=0.0)
    assert rating == AudioQuality.CLEAR and not override


def test_mid_pesq_is_slightly_impaired():
    rating, _ = rate_quality(pesq=2.4, stoi=0.9, clipping_ratio=0.0)
    assert rating == AudioQuality.SLIGHTLY_IMPAIRED


def test_low_stoi_degrades_one_level():
    rating, _ = rate_quality(pesq=3.5, stoi=0.6, clipping_ratio=0.0)
    assert rating == AudioQuality.SLIGHTLY_IMPAIRED


def test_clipping_overrides_everything():
    rating, override = rate_quality(pesq=4.0, stoi=0.95, clipping_ratio=0.08)
    assert rating == AudioQuality.SEVERELY_IMPAIRED and override


def test_missing_scores_default_clear_no_override():
    rating, override = rate_quality(pesq=None, stoi=None, clipping_ratio=0.0)
    assert rating == AudioQuality.CLEAR and not override
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_quality_logic.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Write `src/autoace_audio/analyzers/quality.py`**

```python
"""Technical channel quality ONLY (distortion/clipping/muffling/dropouts) —
independent of background noise by design. SQUIM (reference-free PESQ/STOI/SI-SDR)
+ clipdetect (clipping survives normalization; peak checks don't see it)."""

from dataclasses import dataclass

import numpy as np

from autoace_audio.config import get_settings
from autoace_audio.schema import AudioQuality


@dataclass(frozen=True)
class QualityResult:
    rating: AudioQuality
    pesq: float | None
    stoi: float | None
    si_sdr: float | None
    clipping_ratio: float
    clipping_override: bool


_LEVELS = [AudioQuality.CLEAR, AudioQuality.SLIGHTLY_IMPAIRED, AudioQuality.SEVERELY_IMPAIRED]


def rate_quality(
    pesq: float | None, stoi: float | None, clipping_ratio: float
) -> tuple[AudioQuality, bool]:
    s = get_settings()
    if clipping_ratio > s.clipping_ratio_max:
        return AudioQuality.SEVERELY_IMPAIRED, True
    if pesq is None:
        return AudioQuality.CLEAR, False  # no evidence of impairment
    if pesq >= s.pesq_clear:
        idx = 0
    elif pesq >= s.pesq_slight:
        idx = 1
    else:
        idx = 2
    if stoi is not None and stoi < s.stoi_floor:
        idx = min(idx + 1, 2)
    return _LEVELS[idx], False


def _clipping_ratio(samples: np.ndarray, sr: int) -> float:
    try:
        from clipdetect import detect_clipping

        clipped, total = detect_clipping(samples, sr)
        return float(clipped) / max(int(total), 1)
    except Exception:
        # Fallback: plateau heuristic — fraction of samples within 0.1% of running max.
        peak = float(np.max(np.abs(samples))) or 1.0
        return float(np.mean(np.abs(samples) >= 0.999 * peak))


_SQUIM = None


def _squim():
    global _SQUIM
    if _SQUIM is None:
        from torchaudio.pipelines import SQUIM_OBJECTIVE

        _SQUIM = SQUIM_OBJECTIVE.get_model()
    return _SQUIM


def analyze_quality(samples: np.ndarray, sr: int) -> QualityResult:
    import torch

    assert sr == 16000, "SQUIM expects 16 kHz input"
    # SQUIM is O(n^2)-ish in memory on long clips: score the middle 60s window.
    max_n = 60 * sr
    x = samples if samples.size <= max_n else samples[(samples.size - max_n) // 2:][:max_n]
    pesq = stoi = si_sdr = None
    try:
        with torch.inference_mode():
            stoi_t, pesq_t, si_sdr_t = _squim()(torch.from_numpy(x)[None, :])
        stoi, pesq, si_sdr = float(stoi_t), float(pesq_t), float(si_sdr_t)
    except Exception:
        pass  # rating falls back to clipping-only evidence
    clip_ratio = _clipping_ratio(samples, sr)
    rating, override = rate_quality(pesq, stoi, clip_ratio)
    return QualityResult(
        rating=rating, pesq=pesq, stoi=stoi, si_sdr=si_sdr,
        clipping_ratio=clip_ratio, clipping_override=override,
    )
```

- [ ] **Step 4: Run unit test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_quality_logic.py -q`
Expected: 5 passed

- [ ] **Step 5: Integration test — all three labeled calls are "clear"**

```python
# tests/integration/test_quality_sample_calls.py
import pytest

from autoace_audio.analyzers.quality import analyze_quality
from autoace_audio.audio_io import load_audio
from autoace_audio.schema import AudioQuality


@pytest.mark.slow
@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_labeled_clear(sample_calls_dir, name):
    a = load_audio(sample_calls_dir / name)
    q = analyze_quality(a.samples, a.sr)
    assert q.rating == AudioQuality.CLEAR, f"{name}: pesq={q.pesq} stoi={q.stoi} clip={q.clipping_ratio}"
```

- [ ] **Step 6: Run integration test; calibrate if needed**

Run: `.venv/bin/pytest tests/integration/test_quality_sample_calls.py -q -m slow`
Expected: 3 passed. If SQUIM under-scores these telephony clips (DNS-trained model, known risk), lower `pesq_clear`/`pesq_slight` in config until all three rate clear WITH margin; record measured PESQ values and the chosen thresholds in `docs/decisions.md`.

- [ ] **Step 7: Commit**

```bash
git add src/autoace_audio/analyzers/quality.py tests/unit/test_quality_logic.py tests/integration/test_quality_sample_calls.py
git commit -m "feat(quality): SQUIM scoring with normalization-proof clipping override"
```

---

### Task 8: Tone arms

**Files:**
- Create: `src/autoace_audio/analyzers/tone/__init__.py` (empty), `.../tone/base.py`, `.../tone/dimensional.py`, `.../tone/gemini_tone.py`, `.../tone/transcript_llm.py`
- Test: `tests/unit/test_tone_mapping.py`, `tests/unit/test_gemini_prompt.py`, `tests/integration/test_tone_sample_calls.py`

**Interfaces:**
- Consumes: `VadMap`, `encode_opus_ogg`, `NoiseResult.snr_db`.
- Produces: `ToneResult` dataclass — `tone: EmotionalTone`, `intensity: EmotionalIntensity`, `confidence: float`, `overlap_opinion: bool | None`, `noise_opinion: dict | None` (keys `present: bool`, `type: str`), `raw: dict`; `ToneClassifierError(Exception)`; `classify_tone(arm: str, samples, sr, vad, snr_db) -> ToneResult` dispatcher; pure functions `map_va(arousal: float, valence: float) -> tuple[EmotionalTone, EmotionalIntensity]` (dimensional) and `build_prompt(duration_s, snr_db, speech_ratio) -> str` (gemini).

- [ ] **Step 1: Write the failing unit tests**

```python
# tests/unit/test_tone_mapping.py
from autoace_audio.analyzers.tone.dimensional import map_va
from autoace_audio.schema import EmotionalIntensity, EmotionalTone


def test_high_valence_is_satisfied_any_arousal():
    assert map_va(arousal=0.2, valence=0.8)[0] == EmotionalTone.SATISFIED
    assert map_va(arousal=0.9, valence=0.8)[0] == EmotionalTone.SATISFIED


def test_low_valence_high_arousal_is_upset():
    assert map_va(arousal=0.7, valence=0.35)[0] == EmotionalTone.UPSET


def test_extreme_corner_is_distressed():
    assert map_va(arousal=0.8, valence=0.2)[0] == EmotionalTone.DISTRESSED


def test_mild_negative_is_frustrated():
    assert map_va(arousal=0.5, valence=0.42)[0] == EmotionalTone.FRUSTRATED


def test_middle_is_neutral_and_intensity_bands():
    tone, intensity = map_va(arousal=0.3, valence=0.5)
    assert tone == EmotionalTone.NEUTRAL and intensity == EmotionalIntensity.LOW
    assert map_va(arousal=0.5, valence=0.5)[1] == EmotionalIntensity.MEDIUM
    assert map_va(arousal=0.9, valence=0.5)[1] == EmotionalIntensity.HIGH
```

```python
# tests/unit/test_gemini_prompt.py
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
        "neutral", "satisfied", "frustrated", "upset", "distressed"
    ]
    assert props["emotional_intensity"]["enum"] == ["low", "medium", "high"]
    assert set(GEMINI_RESPONSE_SCHEMA["required"]) >= {
        "emotional_tone", "emotional_intensity", "tone_confidence",
        "background_noise_present", "background_noise_type", "speaker_overlap_present",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_tone_mapping.py tests/unit/test_gemini_prompt.py -q`
Expected: FAIL — modules not found

- [ ] **Step 3: Write `base.py`**

```python
# src/autoace_audio/analyzers/tone/base.py
"""Swappable tone classifiers. Every arm returns the same ToneResult."""

from dataclasses import dataclass, field

import numpy as np

from autoace_audio.analyzers.vad import VadMap
from autoace_audio.schema import EmotionalIntensity, EmotionalTone


class ToneClassifierError(Exception):
    pass


@dataclass(frozen=True)
class ToneResult:
    tone: EmotionalTone
    intensity: EmotionalIntensity
    confidence: float
    overlap_opinion: bool | None = None
    noise_opinion: dict | None = None
    raw: dict = field(default_factory=dict)


def classify_tone(
    arm: str, samples: np.ndarray, sr: int, vad: VadMap, snr_db: float | None
) -> ToneResult:
    if arm == "gemini":
        from autoace_audio.analyzers.tone.gemini_tone import classify

        return classify(samples, sr, vad, snr_db)
    if arm == "dimensional":
        from autoace_audio.analyzers.tone.dimensional import classify

        return classify(samples, sr, vad)
    if arm == "transcript":
        from autoace_audio.analyzers.tone.transcript_llm import classify

        return classify(samples, sr, vad)
    raise ToneClassifierError(f"unknown tone arm: {arm}")
```

- [ ] **Step 4: Write `dimensional.py`**

```python
# src/autoace_audio/analyzers/tone/dimensional.py
"""Arm B: audeering wav2vec2 dimensional SER (arousal/dominance/valence in [0,1])
+ deterministic valence-arousal region mapping. Zero marginal cost, fully local.
Known limits (memo): English-tuned; hears agent+customer mixed."""

from dataclasses import dataclass

import numpy as np

from autoace_audio.analyzers.tone.base import ToneResult
from autoace_audio.analyzers.vad import VadMap
from autoace_audio.config import get_settings
from autoace_audio.schema import EmotionalIntensity, EmotionalTone

MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
CHUNK_S = 20.0


def map_va(arousal: float, valence: float) -> tuple[EmotionalTone, EmotionalIntensity]:
    s = get_settings()
    if valence >= s.va_satisfied_v:
        tone = EmotionalTone.SATISFIED
    elif valence < s.va_distressed_v and arousal >= s.va_distressed_a:
        tone = EmotionalTone.DISTRESSED
    elif valence < s.va_upset_v and arousal >= s.va_upset_a:
        tone = EmotionalTone.UPSET
    elif valence < s.va_frustrated_v and arousal >= s.va_frustrated_a_min:
        tone = EmotionalTone.FRUSTRATED
    else:
        tone = EmotionalTone.NEUTRAL
    if arousal < s.intensity_a_low:
        intensity = EmotionalIntensity.LOW
    elif arousal <= s.intensity_a_high:
        intensity = EmotionalIntensity.MEDIUM
    else:
        intensity = EmotionalIntensity.HIGH
    return tone, intensity


@dataclass
class _Lazy:
    processor: object | None = None
    model: object | None = None


_L = _Lazy()


def _load():
    if _L.model is None:
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

        class RegressionHead(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, features):
                import torch as t

                x = self.dropout(features)
                x = t.tanh(self.dense(x))
                x = self.dropout(x)
                return self.out_proj(x)

        class EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = RegressionHead(config)
                self.init_weights()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                return self.classifier(hidden)

        _L.processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        _L.model = EmotionModel.from_pretrained(MODEL_ID)
        _L.model.eval()
        torch.set_grad_enabled(False)
    return _L.processor, _L.model


def _avd(samples: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Duration-weighted mean (arousal, dominance, valence) over 20s chunks."""
    import torch

    processor, model = _load()
    chunk = int(CHUNK_S * sr)
    outs, weights = [], []
    for i in range(0, samples.size, chunk):
        x = samples[i: i + chunk]
        if x.size < sr:  # skip sub-second tails
            continue
        inputs = processor(x, sampling_rate=sr, return_tensors="pt")
        y = model(inputs.input_values)[0].numpy()
        outs.append(y)
        weights.append(x.size)
    if not outs:
        return 0.5, 0.5, 0.5
    m = np.average(np.stack(outs), axis=0, weights=weights)
    return float(m[0]), float(m[1]), float(m[2])


def classify(samples: np.ndarray, sr: int, vad: VadMap) -> ToneResult:
    from autoace_audio.analyzers.noise import _slice  # speech-only audio

    speech = _slice(samples, sr, vad.speech)
    if speech.size < sr:
        speech = samples
    arousal, dominance, valence = _avd(speech, sr)
    tone, intensity = map_va(arousal, valence)
    # Confidence from distance to nearest mapping boundary (0.35 far -> 0.85 close-to-center).
    s = get_settings()
    boundary_dist = min(
        abs(valence - s.va_satisfied_v), abs(valence - s.va_upset_v),
        abs(valence - s.va_frustrated_v), abs(arousal - s.va_upset_a),
    )
    confidence = float(np.clip(0.45 + 2.0 * boundary_dist, 0.35, 0.85))
    return ToneResult(
        tone=tone, intensity=intensity, confidence=confidence,
        raw={"arousal": arousal, "dominance": dominance, "valence": valence},
    )
```

- [ ] **Step 5: Write `gemini_tone.py`**

```python
# src/autoace_audio/analyzers/tone/gemini_tone.py
"""Arm A (expected primary): gemini-3.1-flash-lite hears the clip once and returns
structured JSON. Audio billed at 32 tok/s => ~$0.0011-0.0016/audio-min all-in.
Label definitions quoted verbatim from the brief; explicitly targets the CUSTOMER."""

import json
import time

import numpy as np

from autoace_audio.analyzers.tone.base import ToneClassifierError, ToneResult
from autoace_audio.analyzers.vad import VadMap
from autoace_audio.audio_io import encode_opus_ogg
from autoace_audio.config import get_settings
from autoace_audio.schema import EmotionalIntensity, EmotionalTone

GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "emotional_tone": {
            "type": "string",
            "enum": ["neutral", "satisfied", "frustrated", "upset", "distressed"],
        },
        "emotional_intensity": {"type": "string", "enum": ["low", "medium", "high"]},
        "tone_confidence": {"type": "number"},
        "background_noise_present": {"type": "boolean"},
        "background_noise_type": {"type": "string"},
        "speaker_overlap_present": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": [
        "emotional_tone", "emotional_intensity", "tone_confidence",
        "background_noise_present", "background_noise_type", "speaker_overlap_present",
    ],
}


def build_prompt(duration_s: float, snr_db: float | None, speech_ratio: float) -> str:
    snr_line = f"{snr_db:.1f} dB" if snr_db is not None else "unmeasurable"
    return f"""You are analyzing ONE recorded phone call ({duration_s:.0f}s) between an AI voice agent (she introduces herself, e.g. "Erica from <dealership>") and a human CUSTOMER.

Classify the CUSTOMER's emotional state only — the AI agent always sounds calm; ignore its tone entirely.

emotional_tone definitions (apply exactly):
- neutral: no clear positive or negative emotion.
- satisfied: pleased, relieved, appreciative, or clearly positive.
- frustrated: annoyed, impatient, or dissatisfied WITHOUT strong anger or distress.
- upset: clearly angry, agitated, or strongly dissatisfied.
- distressed: highly emotional, overwhelmed, panicked, crying, or emotionally escalated.

emotional_intensity: low = subtle/mild; medium = clear and sustained; high = strong, escalated, likely to require attention.

Rules:
- Do NOT infer frustration or distress from loudness or audio volume alone (measured speech-to-background SNR: {snr_line}; speech covers {speech_ratio:.0%} of the call). Judge from words, prosody, and escalation.
- A customer repeatedly saying "hello?" to an unresponsive agent, sighing, or demanding a human indicates frustration or worse — even at normal volume.
- background_noise_present: meaningful NON-SPEECH background sound (TV, music, road noise, chatter, static, typing...). Barely perceptible artifacts do not count. Poor call quality alone is NOT background noise.
- background_noise_type: concise label like "TV", "office chatter", "road noise", "static", "music" — or "" if none.
- speaker_overlap_present: true only if speakers talk over each other enough to affect understanding (brief back-channel "uh-huh" does not count).
- tone_confidence: your 0.0-1.0 confidence in the emotional_tone value.

Return JSON only."""


_TONE = {t.value: t for t in EmotionalTone}
_INT = {i.value: i for i in EmotionalIntensity}


def classify(
    samples: np.ndarray, sr: int, vad: VadMap, snr_db: float | None
) -> ToneResult:
    s = get_settings()
    if not s.gemini_api_key:
        raise ToneClassifierError("GEMINI_API_KEY not configured")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=s.gemini_api_key)
    blob = encode_opus_ogg(samples, sr)
    prompt = build_prompt(samples.size / sr, snr_db, vad.speech_ratio)
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=s.gemini_model,
                contents=[types.Part.from_bytes(data=blob, mime_type="audio/ogg"), prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=GEMINI_RESPONSE_SCHEMA,
                ),
            )
            data = json.loads(resp.text)
            usage = getattr(resp, "usage_metadata", None)
            return ToneResult(
                tone=_TONE[data["emotional_tone"]],
                intensity=_INT[data["emotional_intensity"]],
                confidence=float(np.clip(data.get("tone_confidence", 0.7), 0.0, 1.0)),
                overlap_opinion=bool(data["speaker_overlap_present"]),
                noise_opinion={
                    "present": bool(data["background_noise_present"]),
                    "type": str(data.get("background_noise_type", "")),
                },
                raw={
                    "response": data,
                    "prompt_tokens": getattr(usage, "prompt_token_count", None),
                    "output_tokens": getattr(usage, "candidates_token_count", None),
                },
            )
        except Exception as e:  # noqa: BLE001 — uniform retry, re-raised below
            last_err = e
            time.sleep(2**attempt)
    raise ToneClassifierError(f"gemini failed after 3 attempts: {last_err}")
```

- [ ] **Step 6: Write `transcript_llm.py`**

```python
# src/autoace_audio/analyzers/tone/transcript_llm.py
"""Arm C (bake-off only): faster-whisper multilingual transcript -> OpenAI text model.
Never in the default pipeline unless it wins the bake-off."""

import json

import numpy as np

from autoace_audio.analyzers.tone.base import ToneClassifierError, ToneResult
from autoace_audio.analyzers.vad import VadMap
from autoace_audio.config import get_settings
from autoace_audio.schema import EmotionalIntensity, EmotionalTone

_WHISPER = None


def _whisper():
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel

        _WHISPER = WhisperModel("small", device="cpu", compute_type="int8")
    return _WHISPER


def transcribe(samples: np.ndarray, sr: int) -> str:
    segments, _info = _whisper().transcribe(samples, vad_filter=True)
    return "\n".join(seg.text.strip() for seg in segments)


def classify(samples: np.ndarray, sr: int, vad: VadMap) -> ToneResult:
    s = get_settings()
    if not s.openai_api_key:
        raise ToneClassifierError("OPENAI_API_KEY not configured")
    from openai import OpenAI

    text = transcribe(samples, sr)
    if not text.strip():
        return ToneResult(EmotionalTone.NEUTRAL, EmotionalIntensity.LOW, 0.3,
                          raw={"transcript": ""})
    client = OpenAI(api_key=s.openai_api_key)
    resp = client.chat.completions.create(
        model=s.openai_model,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": (
                "Call transcript between an AI agent (Erica) and a CUSTOMER. Classify the "
                "CUSTOMER's emotion.\nReturn JSON {\"emotional_tone\": one of "
                "[neutral,satisfied,frustrated,upset,distressed], \"emotional_intensity\": "
                "one of [low,medium,high], \"tone_confidence\": 0..1}.\n"
                "frustrated=annoyed/impatient without strong anger; upset=clearly angry; "
                "distressed=overwhelmed/panicked/crying.\n\nTranscript:\n" + text[:8000]
            ),
        }],
    )
    data = json.loads(resp.choices[0].message.content)
    return ToneResult(
        tone=EmotionalTone(data["emotional_tone"]),
        intensity=EmotionalIntensity(data["emotional_intensity"]),
        confidence=float(np.clip(data.get("tone_confidence", 0.6), 0, 1)),
        raw={"transcript": text[:2000], "response": data},
    )
```

- [ ] **Step 7: Run unit tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_tone_mapping.py tests/unit/test_gemini_prompt.py -q`
Expected: 7 passed

- [ ] **Step 8: Write integration test against labels (001=upset/high, 002=neutral, 003=satisfied)**

```python
# tests/integration/test_tone_sample_calls.py
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
```

- [ ] **Step 9: Run integration tests**

Run: `.venv/bin/pytest tests/integration/test_tone_sample_calls.py -q -m "network or slow"`
Expected: gemini test 3/3 (if a label misses, iterate on `build_prompt` wording — record iterations in `docs/decisions.md`); dimensional test passes (first run downloads ~1.2GB audeering model).

- [ ] **Step 10: Commit**

```bash
git add src/autoace_audio/analyzers/tone tests/unit/test_tone_mapping.py tests/unit/test_gemini_prompt.py tests/integration/test_tone_sample_calls.py
git commit -m "feat(tone): three swappable arms — gemini audio, dimensional SER, transcript LLM"
```

---

### Task 9: Fusion + pipeline

**Files:**
- Create: `src/autoace_audio/fusion.py`, `src/autoace_audio/pipeline.py`
- Test: `tests/unit/test_fusion.py`, `tests/integration/test_pipeline_sample_calls.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `fuse(vad: VadMap, noise: NoiseResult, quality: QualityResult, tone: ToneResult | None, tone_error: str | None) -> AnalysisResult`; `PipelineOutput` dataclass — `result: AnalysisResult`, `diagnostics: dict` (keys: `duration_s`, `snr_db`, `pesq`, `tone_arm`, `tone_error`, `gemini_tokens`, `elapsed_s`); `analyze(path: Path, tone_arm: str | None = None) -> PipelineOutput`.

- [ ] **Step 1: Write the failing unit test (stub inputs, no models)**

```python
# tests/unit/test_fusion.py
from autoace_audio.analyzers.noise import NoiseResult
from autoace_audio.analyzers.quality import QualityResult
from autoace_audio.analyzers.tone.base import ToneResult
from autoace_audio.analyzers.vad import VadMap
from autoace_audio.fusion import fuse
from autoace_audio.schema import (
    AudioQuality, EmotionalIntensity, EmotionalTone, Severity,
)


def _vad(long_silence=False):
    return VadMap([], [], 0.8, 3.0, long_silence, 60.0)


def _noise(present=True):
    return NoiseResult(present, "TV" if present else "", Severity.MEDIUM if present else Severity.NONE, 12.0, [("Television", 0.6)])


def _quality():
    return QualityResult(AudioQuality.CLEAR, 3.4, 0.9, 18.0, 0.0, False)


def _tone():
    return ToneResult(EmotionalTone.NEUTRAL, EmotionalIntensity.MEDIUM, 0.8,
                      overlap_opinion=True, noise_opinion={"present": True, "type": "TV"})


def test_happy_path_fields():
    r = fuse(_vad(), _noise(), _quality(), _tone(), None)
    assert r.emotional_tone == EmotionalTone.NEUTRAL
    assert r.background_noise_present and r.background_noise_type == "TV"
    assert r.background_noise_severity == Severity.MEDIUM
    assert r.speaker_overlap_present is True
    assert 0.05 <= r.confidence <= 0.98


def test_noise_absent_forces_empty_type_and_none_severity():
    r = fuse(_vad(), _noise(present=False), _quality(), _tone(), None)
    assert not r.background_noise_present
    assert r.background_noise_type == "" and r.background_noise_severity == Severity.NONE


def test_tone_failure_degrades_to_neutral_low_confidence():
    r = fuse(_vad(), _noise(), _quality(), None, "gemini timeout")
    assert r.emotional_tone == EmotionalTone.NEUTRAL
    assert r.confidence <= 0.40


def test_gemini_noise_opinion_fills_type_when_aed_uncertain():
    n = NoiseResult(False, "", Severity.NONE, 13.0, [("Television", 0.30)])  # below threshold
    t = _tone()  # gemini says TV present
    r = fuse(_vad(), n, _quality(), t, None)
    assert r.background_noise_present and r.background_noise_type == "TV"
    assert r.background_noise_severity == Severity.MEDIUM  # severity re-derived from SNR 13dB
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_fusion.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Write `src/autoace_audio/fusion.py`**

```python
"""Merge analyzer outputs into the final AnalysisResult; enforce invariants;
compute calibrated confidence. All cross-field rules live HERE, nowhere else."""

from autoace_audio.analyzers.noise import NoiseResult, severity_from_snr
from autoace_audio.analyzers.quality import QualityResult
from autoace_audio.analyzers.tone.base import ToneResult
from autoace_audio.analyzers.vad import VadMap
from autoace_audio.config import get_settings
from autoace_audio.schema import AnalysisResult, EmotionalIntensity, EmotionalTone, Severity


def fuse(
    vad: VadMap,
    noise: NoiseResult,
    quality: QualityResult,
    tone: ToneResult | None,
    tone_error: str | None,
) -> AnalysisResult:
    s = get_settings()

    # --- noise: AED is primary; Gemini's opinion breaks borderline cases ---
    present, type_label, severity = noise.present, noise.type_label, noise.severity
    llm_noise = tone.noise_opinion if tone else None
    if not present and llm_noise and llm_noise.get("present"):
        # AED below threshold but the audio-LLM heard something: accept with SNR-derived severity.
        present = True
        type_label = llm_noise.get("type") or (noise.top_events[0][0] if noise.top_events else "background noise")
        severity = severity_from_snr(noise.snr_db, present=True)
    if not present:
        type_label, severity = "", Severity.NONE

    # --- overlap: tone arm's audio judgment; default false without evidence ---
    overlap = bool(tone.overlap_opinion) if tone and tone.overlap_opinion is not None else False

    # --- tone: degrade gracefully if the arm failed ---
    if tone is not None:
        tone_val, intensity, tone_conf = tone.tone, tone.intensity, tone.confidence
    else:
        tone_val, intensity, tone_conf = EmotionalTone.NEUTRAL, EmotionalIntensity.LOW, 0.2

    # --- confidence: weighted blend, clamped; capped when degraded ---
    noise_margin = abs((noise.top_events[0][1] if noise.top_events else 0.0) - s.aed_prob_threshold)
    quality_conf = 0.9 if quality.pesq is not None else 0.5
    confidence = 0.55 * tone_conf + 0.25 * min(1.0, 0.5 + noise_margin) + 0.20 * quality_conf
    if tone_error:
        confidence = min(confidence, s.tone_degraded_confidence_cap)
    confidence = max(s.confidence_floor, min(s.confidence_ceiling, confidence))

    return AnalysisResult(
        emotional_tone=tone_val,
        emotional_intensity=intensity,
        background_noise_present=present,
        background_noise_type=type_label,
        background_noise_severity=severity,
        audio_quality=quality.rating,
        speaker_overlap_present=overlap,
        long_silence_present=vad.long_silence_present,
        confidence=round(confidence, 2),
    )
```

- [ ] **Step 4: Write `src/autoace_audio/pipeline.py`**

```python
"""Single public entry point: analyze one clip. The dashboard and batch CLI wrap this."""

import time
from dataclasses import dataclass
from pathlib import Path

from autoace_audio.analyzers.noise import analyze_noise
from autoace_audio.analyzers.quality import analyze_quality
from autoace_audio.analyzers.tone.base import ToneClassifierError, classify_tone
from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import load_audio
from autoace_audio.config import get_settings
from autoace_audio.fusion import fuse
from autoace_audio.schema import AnalysisResult


@dataclass(frozen=True)
class PipelineOutput:
    result: AnalysisResult
    diagnostics: dict


def analyze(path: Path, tone_arm: str | None = None) -> PipelineOutput:
    """Raises DecodeError on unreadable audio; everything else degrades gracefully."""
    s = get_settings()
    arm = tone_arm or s.tone_arm
    t0 = time.monotonic()
    audio = load_audio(Path(path))
    vad = analyze_vad(audio.samples, audio.sr)
    noise = analyze_noise(audio.samples, audio.sr, vad)
    quality = analyze_quality(audio.samples, audio.sr)
    tone, tone_error = None, None
    try:
        tone = classify_tone(arm, audio.samples, audio.sr, vad, noise.snr_db)
    except ToneClassifierError as e:
        tone_error = str(e)
        if arm != "dimensional":  # local fallback arm
            try:
                tone = classify_tone("dimensional", audio.samples, audio.sr, vad, noise.snr_db)
            except ToneClassifierError as e2:
                tone_error = f"{tone_error}; fallback: {e2}"
    result = fuse(vad, noise, quality, tone, tone_error)
    return PipelineOutput(
        result=result,
        diagnostics={
            "duration_s": round(audio.duration_s, 2),
            "snr_db": noise.snr_db,
            "pesq": quality.pesq,
            "tone_arm": arm,
            "tone_error": tone_error,
            "gemini_tokens": (tone.raw.get("prompt_tokens") if tone else None),
            "elapsed_s": round(time.monotonic() - t0, 2),
        },
    )
```

- [ ] **Step 5: Run unit tests**

Run: `.venv/bin/pytest tests/unit/test_fusion.py -q`
Expected: 4 passed

- [ ] **Step 6: Write end-to-end integration test vs all labels**

```python
# tests/integration/test_pipeline_sample_calls.py
import json

import pytest

from autoace_audio.pipeline import analyze

FIELDS_SCORED = [
    "emotional_tone", "background_noise_present", "audio_quality",
    "speaker_overlap_present", "long_silence_present",
]


@pytest.mark.network
def test_end_to_end_against_labels(sample_calls_dir):
    import csv

    labels = {}
    with open(sample_calls_dir / "labels.csv", newline="") as f:
        for row in csv.DictReader(f):
            labels[row["name"]] = json.loads(row["result_json"])
    hits, total = 0, 0
    for name, expected in labels.items():
        out = analyze(sample_calls_dir / name).result.model_dump(mode="json")
        for field in FIELDS_SCORED:
            total += 1
            hits += int(out[field] == expected[field])
        print(name, out)
    accuracy = hits / total
    assert accuracy >= 0.8, f"sample-call field accuracy {accuracy:.0%} below 80%"
```

- [ ] **Step 7: Run it**

Run: `.venv/bin/pytest tests/integration/test_pipeline_sample_calls.py -q -m network -s`
Expected: PASS with printed per-call outputs. Investigate any field misses now — this is the cheapest debugging moment.

- [ ] **Step 8: Commit**

```bash
git add src/autoace_audio/fusion.py src/autoace_audio/pipeline.py tests/unit/test_fusion.py tests/integration/test_pipeline_sample_calls.py
git commit -m "feat(pipeline): fusion with cross-field invariants and graceful tone fallback"
```

---

### Task 10: Batch runner + CLI

**Files:**
- Create: `src/autoace_audio/batch.py`, `src/autoace_audio/__main__.py`
- Test: `tests/unit/test_batch.py`

**Interfaces:**
- Consumes: `analyze`, `PipelineOutput`, `FileError`, `AnalysisResult`.
- Produces: `validate_batch(input_dir: Path) -> tuple[list[Path], list[str]]` (audio files matched to manifest, warnings); `run_batch(input_dir: Path, out_dir: Path, tone_arm: str | None = None, analyze_fn=analyze) -> BatchReport`; `BatchReport` dataclass — `results: dict[str, AnalysisResult]`, `errors: list[FileError]`, `warnings: list[str]`; writes `out_dir/results.csv` (`name,result_json`), `results.json`, `errors.csv`. CLI: `python -m autoace_audio analyze <dir> [--out out/] [--arm gemini]`, accepts a directory or a `.zip`.

- [ ] **Step 1: Write the failing test (stubbed analyze_fn — no models, no network)**

```python
# tests/unit/test_batch.py
import csv
import json
from pathlib import Path

from autoace_audio.audio_io import DecodeError
from autoace_audio.batch import run_batch, validate_batch
from autoace_audio.pipeline import PipelineOutput
from autoace_audio.schema import AnalysisResult


GOOD = AnalysisResult(
    emotional_tone="neutral", emotional_intensity="low",
    background_noise_present=False, background_noise_type="",
    background_noise_severity="none", audio_quality="clear",
    speaker_overlap_present=False, long_silence_present=False, confidence=0.8,
)


def _fake_analyze(path, tone_arm=None):
    if "corrupt" in Path(path).name:
        raise DecodeError("bad file")
    return PipelineOutput(result=GOOD, diagnostics={})


def _mkbatch(tmp_path, names, manifest_rows):
    d = tmp_path / "batch"
    d.mkdir()
    for n in names:
        (d / n).write_bytes(b"RIFFxxxxWAVE")
    with open(d / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "result_json"])
        w.writerows(manifest_rows)
    return d


def test_validate_reports_mismatches_both_ways(tmp_path):
    d = _mkbatch(tmp_path, ["a.wav"], [["a.wav", ""], ["missing.wav", ""]])
    (d / "extra.wav").write_bytes(b"RIFF")
    files, warnings = validate_batch(d)
    assert {f.name for f in files} == {"a.wav", "extra.wav"}
    joined = " ".join(warnings)
    assert "missing.wav" in joined and "extra.wav" in joined


def test_one_corrupt_file_does_not_kill_batch(tmp_path):
    d = _mkbatch(tmp_path, ["ok.wav", "corrupt.wav"], [["ok.wav", ""], ["corrupt.wav", ""]])
    report = run_batch(d, tmp_path / "out", analyze_fn=_fake_analyze)
    assert set(report.results) == {"ok.wav"}
    assert len(report.errors) == 1 and report.errors[0].name == "corrupt.wav"
    rows = list(csv.DictReader(open(tmp_path / "out" / "results.csv", newline="")))
    assert rows[0]["name"] == "ok.wav"
    assert json.loads(rows[0]["result_json"])["emotional_tone"] == "neutral"


def test_results_json_preserves_filenames(tmp_path):
    d = _mkbatch(tmp_path, ["x.wav"], [["x.wav", ""]])
    run_batch(d, tmp_path / "out", analyze_fn=_fake_analyze)
    data = json.loads((tmp_path / "out" / "results.json").read_text())
    assert list(data.keys()) == ["x.wav"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_batch.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Write `src/autoace_audio/batch.py`**

```python
"""Batch processing with per-file failure isolation: one bad file never kills the run.
Manifest contract per brief: CSV with `name` (exact filename) and `result_json`."""

import csv
import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from autoace_audio.audio_io import DecodeError
from autoace_audio.pipeline import analyze
from autoace_audio.schema import AnalysisResult, FileError

AUDIO_SUFFIXES = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".webm"}


@dataclass
class BatchReport:
    results: dict[str, AnalysisResult] = field(default_factory=dict)
    errors: list[FileError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _find_manifest(input_dir: Path) -> Path | None:
    csvs = sorted(input_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def validate_batch(input_dir: Path) -> tuple[list[Path], list[str]]:
    """Cross-check manifest rows vs files on disk, both directions."""
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
    )
    warnings: list[str] = []
    manifest = _find_manifest(input_dir)
    if manifest is None:
        warnings.append("no CSV manifest found — processing every audio file")
        return files, warnings
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("name")]
    manifest_names = {r["name"].strip() for r in rows}
    disk_names = {p.name for p in files}
    for name in sorted(manifest_names - disk_names):
        warnings.append(f"manifest row has no file on disk: {name}")
    for name in sorted(disk_names - manifest_names):
        warnings.append(f"file not listed in manifest (processed anyway): {name}")
    return files, warnings


def _unzip_if_needed(input_path: Path) -> Path:
    if input_path.suffix.lower() == ".zip":
        target = Path(tempfile.mkdtemp(prefix="autoace_batch_"))
        with zipfile.ZipFile(input_path) as z:
            z.extractall(target)
        inner = [d for d in target.iterdir() if d.is_dir()]
        return inner[0] if len(inner) == 1 and not list(target.glob("*.csv")) else target
    return input_path


def run_batch(
    input_path: Path,
    out_dir: Path,
    tone_arm: str | None = None,
    analyze_fn=analyze,
    progress_cb=None,
) -> BatchReport:
    input_dir = _unzip_if_needed(Path(input_path))
    files, warnings = validate_batch(input_dir)
    report = BatchReport(warnings=warnings)
    for i, path in enumerate(files):
        try:
            out = analyze_fn(path, tone_arm=tone_arm)
            report.results[path.name] = out.result
        except DecodeError as e:
            report.errors.append(FileError(name=path.name, error=f"decode: {e}"))
        except Exception as e:  # noqa: BLE001 — isolation is the contract
            report.errors.append(FileError(name=path.name, error=f"{type(e).__name__}: {e}"))
        if progress_cb:
            progress_cb(i + 1, len(files), path.name)
    _write_outputs(report, Path(out_dir))
    return report


def _write_outputs(report: BatchReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "result_json"])
        for name, result in report.results.items():
            w.writerow([name, result.to_result_json()])
    (out_dir / "results.json").write_text(
        json.dumps({n: json.loads(r.to_result_json()) for n, r in report.results.items()}, indent=2)
    )
    with open(out_dir / "errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "error"])
        for e in report.errors:
            w.writerow([e.name, e.error])
```

- [ ] **Step 4: Write `src/autoace_audio/__main__.py`**

```python
import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="autoace_audio")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyze", help="analyze a folder or ZIP of calls (+ optional CSV manifest)")
    p.add_argument("input", type=Path)
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--arm", default=None, help="tone arm: gemini | dimensional | transcript")
    args = parser.parse_args()

    from autoace_audio.batch import run_batch

    def progress(done: int, total: int, name: str) -> None:
        print(f"[{done}/{total}] {name}", flush=True)

    report = run_batch(args.input, args.out, tone_arm=args.arm, progress_cb=progress)
    for w in report.warnings:
        print(f"WARN: {w}", file=sys.stderr)
    for e in report.errors:
        print(f"ERROR: {e.name}: {e.error}", file=sys.stderr)
    print(f"done: {len(report.results)} ok, {len(report.errors)} failed -> {args.out}/")
    return 0 if report.results or not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/unit/test_batch.py -q`
Expected: 3 passed

- [ ] **Step 6: Smoke the real CLI on the labeled calls**

Run: `.venv/bin/python -m autoace_audio analyze data/ --out out/`
Expected: `[1/3]..[3/3]`, `done: 3 ok, 0 failed`; `out/results.csv` has 3 rows of valid JSON.

- [ ] **Step 7: Commit**

```bash
git add src/autoace_audio/batch.py src/autoace_audio/__main__.py tests/unit/test_batch.py
git commit -m "feat(batch): manifest-validated batch runner with per-file isolation and CLI"
```

---

### Task 11: Evaluation harness + bake-off

**Files:**
- Create: `eval/__init__.py` (empty), `eval/metrics.py`, `eval/build_validation_set.py`, `eval/evaluate.py`, `eval/bakeoff.py`
- Test: `tests/unit/test_metrics.py`

**Interfaces:**
- Consumes: `run_batch`, `analyze`, `classify_tone`, labels CSV format.
- Produces: `eval/metrics.py`: `macro_f1(y_true: list[str], y_pred: list[str]) -> float`, `confusion(y_true, y_pred) -> dict[str, dict[str, int]]`, `field_report(labels: dict[str, dict], preds: dict[str, dict]) -> str` (markdown). Scripts runnable via `make evaluate` / `make bakeoff`.

- [ ] **Step 1: Write the failing metrics test**

```python
# tests/unit/test_metrics.py
from eval.metrics import confusion, macro_f1


def test_macro_f1_perfect():
    assert macro_f1(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_macro_f1_weights_minority_class_equally():
    y_true = ["a"] * 9 + ["b"]
    always_a = ["a"] * 10
    assert macro_f1(y_true, always_a) < 0.5  # majority-vote cheat is punished


def test_confusion_counts():
    c = confusion(["a", "a", "b"], ["a", "b", "b"])
    assert c["a"]["a"] == 1 and c["a"]["b"] == 1 and c["b"]["b"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_metrics.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Write `eval/metrics.py`**

```python
"""Self-contained metrics (no sklearn dependency)."""

from collections import defaultdict


def confusion(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    m: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred, strict=True):
        m[t][p] += 1
    return {t: dict(row) for t, row in m.items()}


def _prf(y_true: list[str], y_pred: list[str], cls: str) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p != cls)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    classes = sorted(set(y_true))
    return sum(_prf(y_true, y_pred, c) for c in classes) / len(classes)


def field_report(labels: dict[str, dict], preds: dict[str, dict]) -> str:
    """Markdown report: per-field accuracy; macro F1 + confusion for emotional_tone."""
    names = [n for n in labels if n in preds]
    lines = [f"# Evaluation report ({len(names)} clips)", ""]
    for f in [
        "emotional_tone", "emotional_intensity", "background_noise_present",
        "background_noise_severity", "audio_quality", "speaker_overlap_present",
        "long_silence_present",
    ]:
        pairs = [(str(labels[n][f]), str(preds[n][f])) for n in names if f in labels[n]]
        if not pairs:
            continue
        acc = sum(t == p for t, p in pairs) / len(pairs)
        lines.append(f"- **{f}**: accuracy {acc:.0%} ({len(pairs)} clips)")
    tones = [(labels[n]["emotional_tone"], preds[n]["emotional_tone"]) for n in names]
    y_t, y_p = [t for t, _ in tones], [p for _, p in tones]
    lines += ["", f"**emotional_tone macro F1: {macro_f1(y_t, y_p):.3f}**", "", "## Tone confusion"]
    conf = confusion(y_t, y_p)
    classes = sorted(set(y_t) | set(y_p))
    lines.append("| true\\pred | " + " | ".join(classes) + " |")
    lines.append("|---" * (len(classes) + 1) + "|")
    for t in classes:
        lines.append(f"| {t} | " + " | ".join(str(conf.get(t, {}).get(p, 0)) for p in classes) + " |")
    return "\n".join(lines)
```

- [ ] **Step 4: Run metrics test**

Run: `.venv/bin/pytest tests/unit/test_metrics.py -q`
Expected: 3 passed

- [ ] **Step 5: Write `eval/build_validation_set.py`**

```python
"""Build a leakage-safe validation set from the 3 labeled calls:
1) VAD-aligned 15-45s segments (hand-label these once, in segments_labels.csv);
2) synthetic augmentations with KNOWN ground truth:
   - noise beds harvested from the real calls (TV from call_002 gaps, static from call_003)
     plus generated white/pink/babble/hum, mixed at controlled SNRs -> severity labels;
   - controlled degradations (clipping, band-limit, dropouts) -> audio_quality labels.
Group key = source call: all derivatives of one call stay in one fold."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import load_audio

RNG = np.random.default_rng(7)
SR = 16000


def _segments(samples, vad, min_s=15.0, max_s=45.0):
    """Contiguous windows aligned to speech activity."""
    total = samples.size / SR
    out, start = [], 0.0
    while start + min_s < total:
        end = min(start + max_s, total)
        out.append((start, end))
        start = end
    return out


def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if noise.size < clean.size:
        noise = np.tile(noise, int(np.ceil(clean.size / noise.size)))
    noise = noise[: clean.size]
    p_c = np.sqrt(np.mean(clean**2)) or 1e-8
    p_n = np.sqrt(np.mean(noise**2)) or 1e-8
    gain = p_c / (p_n * 10 ** (snr_db / 20))
    return np.clip(clean + gain * noise, -1.0, 1.0).astype(np.float32)


def _degrade(clean: np.ndarray, kind: str) -> np.ndarray:
    if kind == "clip":
        return np.clip(clean * 8.0, -0.55, 0.55).astype(np.float32) / 0.55 * 0.9
    if kind == "dropout":
        x = clean.copy()
        for _ in range(int(len(x) / SR / 3)):
            i = RNG.integers(0, max(1, len(x) - SR // 4))
            x[i: i + SR // 4] = 0.0
        return x
    return clean


def main(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    calls = {
        "call_001.ogg": {"noise": None},
        "call_002.ogg": {"noise": "TV"},
        "call_003.ogg": {"noise": "static"},
    }
    beds = {}
    for name, meta in calls.items():
        audio = load_audio(data_dir / name)
        vad = analyze_vad(audio.samples, audio.sr)
        # harvest real noise beds from gaps >= 1s
        gap_audio = np.concatenate(
            [audio.samples[int(g.start * SR): int(g.end * SR)] for g in vad.gaps
             if g.end - g.start >= 1.0] or [np.zeros(0, dtype=np.float32)]
        )
        if meta["noise"] and gap_audio.size > SR:
            beds[meta["noise"]] = gap_audio
        for j, (s0, s1) in enumerate(_segments(audio.samples, vad)):
            seg = audio.samples[int(s0 * SR): int(s1 * SR)]
            seg_name = f"{Path(name).stem}_seg{j}.wav"
            sf.write(out_dir / seg_name, seg, SR)
            rows.append({"name": seg_name, "group": name, "kind": "segment", "truth": ""})
    # synthetic beds
    n = 30 * SR
    beds.setdefault("static", 0.3 * RNG.standard_normal(n).astype(np.float32))
    t = np.arange(n) / SR
    beds["electrical hum"] = (0.5 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    # clean source = call_001 (labeled no-noise)
    clean_audio = load_audio(data_dir / "call_001.ogg").samples
    for bed_name, bed in beds.items():
        for snr, sev in [(18.0, "low"), (10.0, "medium"), (2.0, "high")]:
            mixed = _mix_at_snr(clean_audio, bed, snr)
            fname = f"aug_{bed_name.replace(' ', '_')}_snr{int(snr)}.wav"
            sf.write(out_dir / fname, mixed, SR)
            rows.append({
                "name": fname, "group": "call_001.ogg", "kind": "noise_aug",
                "truth": json.dumps({
                    "background_noise_present": True,
                    "background_noise_severity": sev,
                }),
            })
    for kind, quality in [("clip", "severely_impaired"), ("dropout", "slightly_impaired")]:
        x = _degrade(clean_audio, kind)
        fname = f"aug_{kind}.wav"
        sf.write(out_dir / fname, x, SR)
        rows.append({
            "name": fname, "group": "call_001.ogg", "kind": "quality_aug",
            "truth": json.dumps({"audio_quality": quality}),
        })
    with open(out_dir / "validation_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "group", "kind", "truth"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} validation clips -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/validation"))
    a = ap.parse_args()
    main(a.data, a.out)
```

- [ ] **Step 6: Write `eval/evaluate.py`**

```python
"""Score pipeline predictions against ground truth (labels.csv format or
validation_manifest.csv partial-truth format) and emit the markdown report."""

import argparse
import csv
import json
from pathlib import Path

from eval.metrics import field_report


def load_labels(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            blob = row.get("result_json") or row.get("truth") or ""
            if row.get("name") and blob.strip():
                out[row["name"].strip()] = json.loads(blob)
    return out


def main(pred_path: Path, labels_path: Path, out_path: Path | None) -> None:
    preds = json.loads(Path(pred_path).read_text())
    labels = load_labels(labels_path)
    report = field_report(labels, preds)
    print(report)
    if out_path:
        Path(out_path).write_text(report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True, help="out/results.json from the batch CLI")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    main(a.pred, a.labels, a.out)
```

- [ ] **Step 7: Write `eval/bakeoff.py`**

```python
"""Tone bake-off: run arms A/B/C over labeled clips; report accuracy, macro F1,
measured cost per audio-minute, and latency. Output feeds the technical memo."""

import argparse
import csv
import json
import time
from pathlib import Path

from autoace_audio.analyzers.tone.base import classify_tone
from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import load_audio
from eval.metrics import macro_f1

# Live prices (2026-07-16): gemini-3.1-flash-lite $0.50/1M audio-in tok (32 tok/s), $1.50/1M out.
GEMINI_IN_PER_TOK = 0.50 / 1e6
GEMINI_OUT_PER_TOK = 1.50 / 1e6


def main(data_dir: Path, labels_path: Path, arms: list[str], out_path: Path) -> None:
    labels = {}
    with open(labels_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("result_json", "").strip():
                labels[row["name"]] = json.loads(row["result_json"])
    rows, table = [], ["| arm | tone acc | macro F1 | $ / audio-min | s / clip |", "|---|---|---|---|---|"]
    for arm in arms:
        y_true, y_pred, costs, times = [], [], [], []
        for name, truth in labels.items():
            audio = load_audio(data_dir / name)
            vad = analyze_vad(audio.samples, audio.sr)
            t0 = time.monotonic()
            try:
                r = classify_tone(arm, audio.samples, audio.sr, vad, snr_db=None)
            except Exception as e:  # noqa: BLE001 — a failed arm scores as a miss
                print(f"{arm} failed on {name}: {e}")
                continue
            dt = time.monotonic() - t0
            y_true.append(truth["emotional_tone"])
            y_pred.append(r.tone.value)
            times.append(dt / (audio.duration_s / 60.0))
            if arm == "gemini" and r.raw.get("prompt_tokens"):
                dollars = (r.raw["prompt_tokens"] * GEMINI_IN_PER_TOK
                           + (r.raw.get("output_tokens") or 0) * GEMINI_OUT_PER_TOK)
                costs.append(dollars / (audio.duration_s / 60.0))
            rows.append({"arm": arm, "clip": name, "true": truth["emotional_tone"],
                         "pred": r.tone.value, "elapsed_s": round(dt, 2)})
        if not y_true:
            continue
        acc = sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)
        cost = f"${sum(costs)/len(costs):.5f}" if costs else "$0 (local)"
        table.append(
            f"| {arm} | {acc:.0%} | {macro_f1(y_true, y_pred):.3f} | {cost} | {sum(times)/len(times):.1f}/min |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(table) + "\n\n```json\n" + json.dumps(rows, indent=2) + "\n```\n")
    print("\n".join(table))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    ap.add_argument("--arms", nargs="+", default=["gemini", "dimensional", "transcript"])
    ap.add_argument("--out", type=Path, default=Path("out/bakeoff.md"))
    a = ap.parse_args()
    main(a.data, a.labels, a.arms, a.out)
```

- [ ] **Step 8: Run the harness for real**

Run, in order:
1. `.venv/bin/python eval/build_validation_set.py` → expect ~15–25 clips in `data/validation/`
2. `.venv/bin/python -m autoace_audio analyze data/validation --out out/validation` (uses default gemini arm)
3. `.venv/bin/python -m eval.evaluate --pred out/validation/results.json --labels data/validation/validation_manifest.csv --out out/validation_report.md`
4. `.venv/bin/python -m eval.bakeoff --arms gemini dimensional`
Expected: severity/quality augmentation rows score ≥80%; bake-off table prints. **Act on results:** tune config thresholds where systematically off (grouped by source call — no leakage), record every change in `docs/decisions.md`.

- [ ] **Step 9: Commit**

```bash
git add eval tests/unit/test_metrics.py
git commit -m "feat(eval): augmented validation set, field metrics, and tone bake-off"
```

---

### Task 12: Docs, decisions, README completion

**Files:**
- Create: `docs/decisions.md`
- Modify: `README.md`

- [ ] **Step 1: Write `docs/decisions.md`** — record every calibration/threshold change made during Tasks 6–11 with: what was measured, what changed, why. Seed it with the pre-build decisions:

```markdown
# Decision log

- 2026-07-16 — long_silence threshold 10s: AutoAce labeled a 7.4s dead-air gap `false` (call_003).
- 2026-07-16 — format sniffing via ffprobe, never extension: production smoke set contains `.mp3`-named PCM WAVs.
- 2026-07-16 — tone primary = gemini-3.1-flash-lite (paid tier, disclosed): only promptable audio API under the $0.003/min ceiling (verified 2026-07-16); multilingual (call_002 is Spanish); promptable to target the customer, not the TTS agent.
- 2026-07-16 — no audio few-shot in Gemini prompt: 3 example calls would add ~4 audio-minutes of input tokens per request (~3x cost) — label definitions + DSP hints only.
- 2026-07-16 — overlap = Gemini audio judgment: pyannote pretrained OSD refuted/unavailable (research 0-3); energy heuristics on mono mixes are unreliable. Disclosed as weakest field in memo.
- (append build-time findings here: measured SQUIM PESQ on the 3 clear calls, AED threshold tuning, bake-off table + chosen arm...)
```

- [ ] **Step 2: Complete `README.md`** — architecture diagram (the §4 flow from the spec), 3-command quickstart, tone-arm table with measured bake-off numbers, cost table (Gemini token math: 32 tok/s × $0.50/M input + output ≈ $0.0011–0.0016/min; local layer ≈ $0.0002–0.0005/min amortized; total vs $0.003 ceiling), latency measurements from `out/bakeoff.md`, limitations section (overlap weakest; dimensional arm English-tuned; SQUIM narrowband calibration), and a "reproducing our results" section (`make setup && make test && make analyze DIR=data/`).

- [ ] **Step 3: Final full run + lint**

Run: `make lint && make test && .venv/bin/pytest -q -m "slow or network"`
Expected: all green.

- [ ] **Step 4: Commit and push**

```bash
git add docs/decisions.md README.md
git commit -m "docs: decision log, README with measured cost/latency and bake-off results"
git push
```

---

## Self-review notes

- **Spec coverage:** §3 schema→Task 2; §5.1→Task 4; §5.2→Task 5; §5.3→Task 6; §5.4 overlap→Tasks 8+9 (Gemini opinion + fusion default-false; pyannote timebox recorded in decisions.md); §5.5→Task 7; §5.6→Task 8; §6 fusion/confidence→Task 9; §7 validation/bake-off→Task 11; §8 errors→Tasks 9+10; §9 cost/latency→Tasks 11+12; §10 security→global constraints + Task 1; §11 standards→Tasks 1+12.
- **Deliberate deviation:** spec §5.4 step 1 (pyannote attempt) is folded into the decision log rather than a coded task — no HF token is available; revisit only if one is provided.
- **Type consistency check:** `Segment`/`VadMap` (Task 5) consumed by Tasks 6, 8, 9; `NoiseResult.severity` uses `Severity` enum from Task 2; `ToneResult` shape identical across all three arms; `analyze()` signature `(path, tone_arm=None) -> PipelineOutput` used by batch and eval.
