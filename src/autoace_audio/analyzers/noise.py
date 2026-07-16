"""Background noise: WHAT (PANNs CNN14 AED, sliding windows over the FULL clip,
speech classes masked, sustained-support pooled across windows) and HOW MUCH (SNR of
speech vs non-speech RMS, VAD-segmented, unchanged). Never inferred from technical
quality — the brief scores those independently.

AED source was originally "concatenated non-speech gaps only" (see task-6-report.md
git history). Measured on the labeled calls, that under-detected continuous
background noise that bleeds through WHILE the customer is speaking (e.g. a TV in the
next room): the true signal was often far stronger in speech-concurrent audio than in
the scattered, short non-speech gaps, and no single probability threshold could
separate the no-noise anchor from the noise anchor under that architecture. This
windowed-full-clip + sustained-support design replaces it — see task-6-report.md for
the measured evidence and the controller's decision.
"""

from dataclasses import dataclass

import numpy as np

from autoace_audio.analyzers.vad import VadMap
from autoace_audio.config import get_settings
from autoace_audio.schema import Severity

# AudioSet classes that describe the conversation itself — never background noise.
MASKED_CLASSES = {
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Speech synthesizer",
    "Shout",
    "Yell",
    "Whispering",
    "Throat clearing",
    "Breathing",
    "Sigh",
    "Gasp",
    "Cough",
    "Sneeze",
    "Silence",
    "Inside, small room",
    "Inside, large room or hall",
    "Telephone",
    "Telephone bell ringing",
    "Telephone dialing, DTMF",
    "Dial tone",
    # Call-channel/line artifacts, not caller-environment noise (found via call_002/
    # call_003 diagnostics: "Sidetone" and "Busy signal" outranked the true noise
    # signal) — same category as the Telephone/Dial tone entries above, not the
    # client's "background noise" concept.
    "Sidetone",
    "Busy signal",
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


def slice_segments(samples: np.ndarray, sr: int, segments) -> np.ndarray:
    """Concatenate the sample ranges covered by `segments` (shared by noise.py's
    own SNR calc and the tone arms' speech-only slicing)."""
    parts = [samples[int(s.start * sr) : int(s.end * sr)] for s in segments]
    return np.concatenate(parts) if parts else np.empty(0, dtype=samples.dtype)


def snr_db(samples: np.ndarray, sr: int, vad: VadMap) -> float | None:
    speech = slice_segments(samples, sr, vad.speech)
    gaps = slice_segments(samples, sr, vad.gaps)
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


def _window_starts(total_n: int, window_n: int, hop_n: int) -> list[int]:
    """Sample-domain window start offsets: window_n-length windows every hop_n
    samples, plus a final window anchored on the tail so trailing audio is never
    left unscored. A clip no longer than one window yields a single start=0 — the
    caller slices the whole (shorter-than-window) clip as one window."""
    if total_n <= window_n:
        return [0]
    starts = list(range(0, total_n - window_n + 1, hop_n))
    last_start = total_n - window_n
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _support_weights(
    starts: list[float], total_s: float, window_s: float, hop_s: float
) -> list[float]:
    """Residual-ownership "sustained" credit in seconds, one per window in `starts`
    (seconds, ascending — as produced by `_window_starts`, converted to seconds). A
    single window (clip no longer than one window) owns the whole clip, capped at
    window_s. Otherwise the first window owns hop_s and every later window owns only
    min(hop_s, the gap from the previous window's start) — a tail-anchored window
    can sit much closer than hop_s to its predecessor (e.g. a 172s clip's last two
    windows are only 2.0s apart, a 60% overlap; a 5.1s clip's are 0.1s apart, 98%),
    so this caps what it can claim: one spike straddling two near-duplicate windows
    can't be double-counted as two independent detections. On clips too short to
    offer two independently-spaced windows, sum(weights) can come out below
    aed_min_support_s — see analyze_noise's effective_floor, which accepts the best
    available evidence instead of demanding support the clip physically cannot
    provide."""
    if len(starts) == 1:
        return [min(total_s, window_s)]
    weights = [hop_s]
    for prev, cur in zip(starts, starts[1:], strict=False):  # pairwise: len differs by 1
        weights.append(min(hop_s, cur - prev))
    return weights


def analyze_noise(samples: np.ndarray, sr: int, vad: VadMap) -> NoiseResult:
    """AED runs on sliding windows over the FULL clip. A class is "sustained" (and
    only then counts toward presence) if the windows where it scores >=
    aed_prob_threshold cover >= an effective support floor (see `_support_weights`
    and the effective_floor calc below) of residual-ownership window time. This
    requires multiple, genuinely-spaced activations — not one spike, and not one
    spike seen twice through overlapping windows — which is what filters out
    CNN14's overconfident-but-wrong single-window reads on short/out-of-distribution
    audio. top_events reports the duration-weighted mean probability per class
    across ALL windows (whether sustained or not) so near-misses stay visible for
    diagnostics.
    """
    import torch
    import torchaudio.functional as F

    s = get_settings()
    total_s = samples.size / sr
    window_n = max(1, int(round(s.aed_window_s * sr)))
    hop_n = max(1, int(round(s.aed_hop_s * sr)))
    starts = _window_starts(samples.size, window_n, hop_n)

    if len(starts) == 1:
        clips16k = samples[np.newaxis, :]
    else:
        clips16k = np.stack([samples[st : st + window_n] for st in starts])
    weights = _support_weights([st / sr for st in starts], total_s, s.aed_window_s, s.aed_hop_s)

    clips32k = F.resample(torch.from_numpy(clips16k), sr, 32000).numpy()
    clipwise, _ = _tagger().inference(clips32k)  # (n_windows, n_classes)

    names = _audioset_labels()
    total_weight = sum(weights)
    # Floor-cap: on a clip too short to offer aed_min_support_s of independently-
    # spaced window evidence (a <5s single-window clip, or a ~5-10s two-window clip
    # whose windows are forced close together), the configured floor is physically
    # unreachable — accept the best available evidence instead of the clip never
    # being able to report presence at all.
    effective_floor = min(s.aed_min_support_s, total_weight)
    mean_prob: dict[str, float] = {}
    support_s: dict[str, float] = {}
    for i, name in enumerate(names):
        if name in MASKED_CLASSES:
            continue
        col = clipwise[:, i]
        weighted = list(zip(col, weights, strict=True))
        mean_prob[name] = float(sum(p * w for p, w in weighted) / total_weight)
        support_s[name] = float(sum(w for p, w in weighted if p >= s.aed_prob_threshold))

    top = sorted(mean_prob.items(), key=lambda t: t[1], reverse=True)[:5]
    sustained = [(name, p) for name, p in mean_prob.items() if support_s[name] >= effective_floor]
    present = bool(sustained)
    type_label = concise_label(max(sustained, key=lambda t: t[1])[0]) if present else ""

    snr = snr_db(samples, sr, vad)
    return NoiseResult(
        present=present,
        type_label=type_label,
        severity=severity_from_snr(snr, present),
        snr_db=snr,
        top_events=top,
    )
