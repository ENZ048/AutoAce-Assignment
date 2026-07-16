"""Merge analyzer outputs into the final AnalysisResult; enforce invariants;
compute calibrated confidence. All cross-field rules live HERE, nowhere else."""

from autoace_audio.analyzers.noise import NoiseResult, severity_from_snr
from autoace_audio.analyzers.quality import QualityResult
from autoace_audio.analyzers.tone.base import ToneResult
from autoace_audio.analyzers.vad import VadMap
from autoace_audio.config import get_settings
from autoace_audio.schema import AnalysisResult, EmotionalIntensity, EmotionalTone, Severity

# Probabilities are naturally bounded in [0, 1] -- this is a structural clamp on
# the noise-margin-confidence transform below, not a calibrated guess, so it isn't
# a config knob (see config.py's noise_margin_confidence_base for the actual
# calibration constant).
_MARGIN_CONF_CEILING = 1.0


def fuse(
    vad: VadMap,
    noise: NoiseResult,
    quality: QualityResult,
    tone: ToneResult | None,
    tone_error: str | None,
) -> AnalysisResult:
    s = get_settings()

    # --- noise: AED is primary; the tone arm's audio-LLM opinion breaks
    # borderline cases in two directions ---
    present, type_label, severity = noise.present, noise.type_label, noise.severity
    llm_noise = tone.noise_opinion if tone else None
    llm_present = bool(llm_noise and llm_noise.get("present"))
    llm_type = str(llm_noise.get("type") or "").strip() if llm_noise else ""

    if not present and llm_present:
        # Rule A: AED found nothing sustained, but the audio-LLM heard something.
        # Accept it wholesale (type from the LLM, or the AED's own best unsustained
        # guess as a last resort); severity is re-derived from SNR now that we
        # believe noise is present at all.
        present = True
        type_label = llm_type or (
            noise.top_events[0][0] if noise.top_events else "background noise"
        )
        severity = severity_from_snr(noise.snr_db, present=True)
    elif present and llm_present and llm_type and llm_type.lower() != type_label.strip().lower():
        # Rule B (controller amendment, task 9): AED agrees noise is present, but
        # its evidence is thin -- the winning class's sustained support barely
        # cleared the effective floor (within one hop, aed_hop_s) -- and it
        # disagrees with the LLM on WHAT the noise is. CNN14 has no static-family
        # AudioSet class; call_003's real "sharp static" comes out "radio" from
        # AED (see test_noise_sample_calls.py's documented xfail). In exactly this
        # low-evidence regime, prefer the LLM's type string over AED's. Severity is
        # deliberately left untouched -- it stays AED/SNR-derived either way, since
        # the LLM has no calibrated severity opinion, only a presence/type one.
        thin_margin = (noise.support_s - noise.support_floor_s) <= s.aed_hop_s
        if thin_margin:
            type_label = llm_type

    if not present:
        type_label, severity = "", Severity.NONE

    # --- overlap: tone arm's audio judgment; default false without evidence ---
    overlap = bool(tone.overlap_opinion) if tone and tone.overlap_opinion is not None else False

    # --- tone: degrade gracefully if no arm produced a result at all ---
    if tone is not None:
        tone_val, intensity, tone_conf = tone.tone, tone.intensity, tone.confidence
    else:
        tone_val, intensity, tone_conf = (
            EmotionalTone.NEUTRAL,
            EmotionalIntensity.LOW,
            s.tone_missing_confidence,
        )

    # --- confidence: weighted blend, clamped; capped further when a fallback
    # arm had to stand in for a failed primary tone arm ---
    top_prob = noise.top_events[0][1] if noise.top_events else 0.0
    noise_margin = abs(top_prob - s.aed_prob_threshold)
    quality_conf = (
        s.quality_confidence_measured
        if quality.pesq is not None
        else s.quality_confidence_unmeasured
    )
    confidence = (
        s.conf_w_tone * tone_conf
        + s.conf_w_noise * min(_MARGIN_CONF_CEILING, s.noise_margin_confidence_base + noise_margin)
        + s.conf_w_quality * quality_conf
    )
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
