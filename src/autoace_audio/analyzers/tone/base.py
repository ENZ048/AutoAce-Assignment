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
