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


def _run(
    cmd: list[str], *, input_: bytes | None = None, timeout: float
) -> subprocess.CompletedProcess:
    """Wrapper to ensure subprocess timeouts surface as DecodeError."""
    try:
        return subprocess.run(cmd, input=input_, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise DecodeError(f"{cmd[0]} timed out after {timeout:.0f}s") from e


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
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    proc = _run(cmd, timeout=60)
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
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    proc = _run(cmd, timeout=300)
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
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-c:a",
        "libopus",
        "-b:a",
        bitrate,
        "-f",
        "ogg",
        "pipe:1",
    ]
    proc = _run(cmd, input_=samples.astype(np.float32).tobytes(), timeout=300)
    if proc.returncode != 0 or not proc.stdout:
        raise DecodeError(f"opus encode failed: {proc.stderr.decode(errors='replace')[:200]}")
    return proc.stdout
