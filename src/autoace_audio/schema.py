"""Output contract for the AutoAce trial. Enum values are byte-exact per the brief."""

from enum import StrEnum

from pydantic import BaseModel, Field


class EmotionalTone(StrEnum):
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    FRUSTRATED = "frustrated"
    UPSET = "upset"
    DISTRESSED = "distressed"


class EmotionalIntensity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Severity(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AudioQuality(StrEnum):
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
