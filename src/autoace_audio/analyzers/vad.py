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
    # Union coverage, robust to overlapping input: everything that isn't a gap.
    speech_s = max(0.0, total_s - sum(g.end - g.start for g in gaps))
    return VadMap(
        speech=speech,
        gaps=gaps,
        speech_ratio=(speech_s / total_s) if total_s > 0 else 0.0,
        max_gap_s=max_gap,
        long_silence_present=max_gap >= long_silence_s,
        total_s=total_s,
    )


_MODEL = None


def _model() -> object:
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
