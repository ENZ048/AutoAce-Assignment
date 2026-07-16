"""Arm C (bake-off only): faster-whisper multilingual transcript -> OpenAI text model.
Never in the default pipeline unless it wins the bake-off."""

import json

import numpy as np

from autoace_audio.analyzers.tone.base import ToneClassifierError, ToneResult
from autoace_audio.analyzers.vad import VadMap
from autoace_audio.config import get_settings
from autoace_audio.schema import EmotionalIntensity, EmotionalTone

_WHISPER = None

_TRANSCRIPT_PROMPT_CHARS = 8000  # cap on transcript text sent into the LLM prompt
_TRANSCRIPT_RAW_CHARS = 2000  # cap on transcript text retained in ToneResult.raw


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
        return ToneResult(
            EmotionalTone.NEUTRAL,
            EmotionalIntensity.LOW,
            s.transcript_empty_confidence,
            raw={"transcript": ""},
        )
    client = OpenAI(api_key=s.openai_api_key)
    resp = client.chat.completions.create(
        model=s.openai_model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Call transcript between an AI agent (Erica) and a CUSTOMER. Classify the "
                    'CUSTOMER\'s emotion.\nReturn JSON {"emotional_tone": one of '
                    '[neutral,satisfied,frustrated,upset,distressed], "emotional_intensity": '
                    'one of [low,medium,high], "tone_confidence": 0..1}.\n'
                    "frustrated=annoyed/impatient without strong anger; upset=clearly angry; "
                    "distressed=overwhelmed/panicked/crying.\n\nTranscript:\n"
                    + text[:_TRANSCRIPT_PROMPT_CHARS]
                ),
            }
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    return ToneResult(
        tone=EmotionalTone(data["emotional_tone"]),
        intensity=EmotionalIntensity(data["emotional_intensity"]),
        confidence=float(
            np.clip(data.get("tone_confidence", s.transcript_default_confidence), 0, 1)
        ),
        raw={"transcript": text[:_TRANSCRIPT_RAW_CHARS], "response": data},
    )
