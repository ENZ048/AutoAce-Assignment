"""Single public entry point: analyze one clip. The dashboard and batch CLI wrap this."""

import time
from dataclasses import dataclass
from pathlib import Path

from autoace_audio.analyzers.noise import analyze_noise
from autoace_audio.analyzers.quality import analyze_quality
from autoace_audio.analyzers.tone.base import classify_tone
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
    # Controller amendment A: analyze_quality's real (v2, task 7) signature takes
    # vad and the already-computed SNR as caller-supplied evidence -- it never
    # loads/re-runs either model itself. Ordering here (vad -> noise -> quality)
    # is what makes noise.snr_db available by the time quality needs it.
    quality = analyze_quality(audio.samples, audio.sr, vad, noise.snr_db)
    tone, tone_error, tone_arm_used = None, None, None
    try:
        tone = classify_tone(arm, audio.samples, audio.sr, vad, noise.snr_db)
        tone_arm_used = arm
    except Exception as e:  # noqa: BLE001 -- graceful degradation is the contract:
        # a dimensional model-load failure (raw OSError/HF errors, not just
        # ToneClassifierError) must never crash analyze() and discard the
        # already-computed deterministic fields (design doc §8: total failure
        # only on undecodable audio).
        tone_error = str(e)
        if arm != "dimensional":  # local fallback arm -- no network/API dependency
            try:
                tone = classify_tone("dimensional", audio.samples, audio.sr, vad, noise.snr_db)
                tone_arm_used = "dimensional"
            except Exception as e2:  # noqa: BLE001 -- same rationale as above
                tone_error = f"{tone_error}; fallback: {e2}"
    result = fuse(vad, noise, quality, tone, tone_error)
    return PipelineOutput(
        result=result,
        diagnostics={
            "duration_s": round(audio.duration_s, 2),
            "snr_db": noise.snr_db,
            "pesq": quality.pesq,
            "tone_arm": arm,  # the arm that was REQUESTED
            "tone_arm_used": tone_arm_used,  # the arm whose ToneResult actually
            # reached fuse() -- "dimensional" after a fallback, same as tone_arm on
            # the happy path, None if every arm (primary + fallback) failed.
            "tone_error": tone_error,
            "tone_prompt_tokens": (tone.raw.get("prompt_tokens") if tone else None),
            # Renamed from gemini_tokens: this reports whichever arm actually
            # produced the tone result (gemini or transcript both populate
            # raw["prompt_tokens"]; dimensional has none, so this is None).
            "elapsed_s": round(time.monotonic() - t0, 2),
        },
    )


def preload_models() -> None:
    """Warm the model singletons every batch uses (silero VAD, PANNs tagger,
    SQUIM) in this process, so their load time lands before the first file
    instead of inside it. Fallback-only models (dimensional tone, whisper)
    stay lazy — a preload must never download what a batch may not need."""
    from autoace_audio.analyzers import noise, quality, vad

    vad._model()
    noise._tagger()
    quality._squim()
