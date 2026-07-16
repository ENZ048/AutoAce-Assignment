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
    # Call-channel/line artifacts, not caller-environment noise (found via call_002
    # integration diagnostics: "Sidetone" and "Busy signal" outranked the true TV/music
    # signal in the non-speech gaps of a labeled-TV-noise call) — same category as the
    # Telephone/Dial tone entries above, not the client's "background noise" concept.
    "Sidetone", "Busy signal",
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
    import torch
    import torchaudio.functional as F

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
