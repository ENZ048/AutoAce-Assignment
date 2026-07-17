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
        "emotional_tone",
        "emotional_intensity",
        "tone_confidence",
        "background_noise_present",
        "background_noise_type",
        "speaker_overlap_present",
    ],
}


def build_prompt(duration_s: float, snr_db: float | None, speech_ratio: float) -> str:
    snr_line = f"{snr_db:.1f} dB" if snr_db is not None else "unmeasurable"
    return f"""You are analyzing ONE recorded phone call ({duration_s:.0f}s) between an automated agent (an AI voice assistant or business representative — it typically introduces itself by name at the start) and a human CUSTOMER.

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
- satisfied vs neutral, DECISIVELY: a call that reaches a cooperative resolution — the customer's request gets handled, the customer stays engaged and polite, and nothing negative happens — counts as satisfied, even if no one says "thank you" and even through minor scheduling friction. Neutral is the exception, not the default: reserve it for interactions that are barely emotional at all (near-silent, purely transactional single exchanges, or a call that is cut short before any rapport forms). When genuinely unsure between the two, prefer satisfied over neutral.
- If the agent fails to respond usefully after the customer greets it three or more times (e.g. repeated "hello?"/"hello, hello" with no working reply), classify the call as upset even if it then proceeds calmly once the connection works and even if intensity would otherwise look low — a caller stuck repeating themselves to a non-responsive system is having a strongly negative experience by definition, not a neutral one. Weight the WORST moment of the call, not the average across the whole call, when the call opens with this kind of failure.
- Profanity, insults, or crude/slang language are NEVER sufficient evidence of frustration or upset BY THEMSELVES — regardless of how harsh the word's literal dictionary meaning is. Many callers use such words as a casual verbal tic, especially when it happens exactly once, briefly, without shouting and without repetition. Before letting any crude word move your answer away from neutral, you must find INDEPENDENT evidence of real escalation elsewhere in the call (raised volume sustained over multiple turns, repeated complaints, refusal to cooperate, hostility toward the agent). If a call contains one crude aside but is otherwise calm and cooperative, score it based on the REST of the call, ignoring that single word's dictionary meaning.
- background_noise_present: meaningful NON-SPEECH background sound (TV, music, road noise, chatter, static, typing...). Barely perceptible artifacts do not count. Poor call quality alone is NOT background noise.
- background_noise_type: concise label like "TV", "office chatter", "road noise", "static", "music" — or "" if none.
- speaker_overlap_present: true only if speakers talk over each other enough to affect understanding (brief back-channel "uh-huh" does not count).
- tone_confidence: your 0.0-1.0 confidence in the emotional_tone value.

Return JSON only."""


_TONE = {t.value: t for t in EmotionalTone}
_INT = {i.value: i for i in EmotionalIntensity}

_MAX_ATTEMPTS = 3  # retries for transient network/API errors only (never for a
# parse/contract failure -- that would silently re-send the same paid audio).
_BACKOFF_BASE_S = 2  # exponential backoff: attempt 0/1/2 -> sleep 1s/2s/4s.


def _make_client(s):  # noqa: ANN001, ANN202 — Settings; returns genai.Client (deferred import)
    """Client with a hard per-request timeout. Without it, one pathological clip can
    wedge an entire batch indefinitely (found via adversarial stress batch: a 0.5s
    blip hung the CLI for 25+ minutes). A bounded timeout converts a stall into
    ToneClassifierError -> the pipeline's existing local-fallback path."""
    from google import genai

    return genai.Client(
        api_key=s.gemini_api_key,
        http_options={"timeout": int(s.gemini_timeout_s * 1000)},  # SDK takes ms
    )


def classify(samples: np.ndarray, sr: int, vad: VadMap, snr_db: float | None) -> ToneResult:
    s = get_settings()
    if not s.gemini_api_key:
        raise ToneClassifierError("GEMINI_API_KEY not configured")
    from google.genai import types

    client = _make_client(s)
    blob = encode_opus_ogg(samples, sr)
    prompt = build_prompt(samples.size / sr, snr_db, vad.speech_ratio)
    last_err: Exception | None = None
    resp = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = client.models.generate_content(
                model=s.gemini_model,
                contents=[types.Part.from_bytes(data=blob, mime_type="audio/ogg"), prompt],
                config=types.GenerateContentConfig(
                    temperature=s.gemini_temperature,
                    response_mime_type="application/json",
                    response_schema=GEMINI_RESPONSE_SCHEMA,
                ),
            )
            break  # got a response back; parse it outside the retry loop below
        except Exception as e:  # noqa: BLE001 — uniform retry for transient network/API errors
            last_err = e
            resp = None
            time.sleep(_BACKOFF_BASE_S**attempt)
    else:
        raise ToneClassifierError(f"gemini failed after {_MAX_ATTEMPTS} attempts: {last_err}")

    # Parse/contract failures are never transient -- retrying would silently re-send
    # the same paid audio for no benefit. Fail fast with the raw response attached.
    # TypeError is included because a safety-blocked response has resp.text=None,
    # and json.loads(None) raises TypeError, not JSONDecodeError.
    try:
        data = json.loads(resp.text)
        usage = getattr(resp, "usage_metadata", None)
        return ToneResult(
            tone=_TONE[data["emotional_tone"]],
            intensity=_INT[data["emotional_intensity"]],
            confidence=float(
                np.clip(data.get("tone_confidence", s.gemini_default_confidence), 0.0, 1.0)
            ),
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
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        raise ToneClassifierError(
            f"gemini returned an unparseable/invalid response: {e!r}; raw response text: "
            f"{resp.text!r}"
        ) from e
