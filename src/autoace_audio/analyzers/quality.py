"""Technical channel quality ONLY (distortion/clipping/muffling/dropouts) —
independent of background noise by design.

Architecture (task 7 rework; full rationale + per-call evidence table in
task-7-report.md): SQUIM's PESQ tracked background SNR almost exactly on the 3
labeled-clear anchor calls (measured on the original 60s scoring window — the
noisiest-but-technically-clean call scored the WORST PESQ of the three; the shipped
15s window shuffles that ranking, see task-7-report.md rework section, but the
conclusion stands) — it measures general perceptual quality, which conflates
channel damage with ambient noise, but this field is scored independent of
background noise. So SQUIM PESQ/STOI bands can no longer be primary evidence.
Primary evidence is now 4 deterministic, channel-only signals computed straight
from the waveform + VAD speech timeline: clipping (clipdetect), dropouts (hard
near-zero runs inside speech), bandwidth/muffle (spectral rolloff), and low volume
(speech-segment RMS). The worst level any one of them triggers wins; default CLEAR.
SQUIM is demoted to a noise-conditioned backstop: it may only escalate a call when
the background is clean enough (snr_db above a threshold) that noise cannot explain
an otherwise-catastrophic PESQ — i.e. it now only fires on real channel damage that
happens to coincide with a clean background, never on noisy-but-undistorted audio.
"""

from dataclasses import dataclass

import numpy as np

from autoace_audio.analyzers.vad import Segment, VadMap
from autoace_audio.config import get_settings
from autoace_audio.schema import AudioQuality


@dataclass(frozen=True)
class QualityResult:
    rating: AudioQuality
    pesq: float | None
    stoi: float | None
    si_sdr: float | None
    clipping_ratio: float
    clipping_override: bool
    dropouts_per_min: float
    rolloff_hz: float | None
    speech_rms_dbfs: float | None


_LEVELS = [AudioQuality.CLEAR, AudioQuality.SLIGHTLY_IMPAIRED, AudioQuality.SEVERELY_IMPAIRED]


def rate_quality(
    clipping_ratio: float,
    dropouts_per_min: float,
    rolloff_hz: float | None,
    speech_rms_dbfs: float | None,
    pesq: float | None,
    snr_db: float | None,
) -> tuple[AudioQuality, bool]:
    """Pure decision logic over pre-measured evidence. Deterministic channel
    evidence (clipping/dropouts/rolloff/volume) is primary; each is an independent
    failure mode, so the worst level any one of them triggers wins (not a blended
    score). Clipping is a hard override — SEVERELY_IMPAIRED + override=True
    immediately, regardless of every other value — because a clipped channel is
    unambiguous technical damage. SQUIM's PESQ is a demoted backstop, gated on
    snr_db so it can only escalate a call when background noise CANNOT be the
    explanation for a bad PESQ (see module docstring and task-7-report.md); a
    missing pesq or snr_db simply leaves the backstop unable to fire, never treated
    as adverse evidence on its own."""
    s = get_settings()
    if clipping_ratio > s.clipping_ratio_max:
        return AudioQuality.SEVERELY_IMPAIRED, True

    idx = 0  # CLEAR by default

    if dropouts_per_min > s.dropout_high_per_min:
        idx = max(idx, 2)
    elif dropouts_per_min > s.dropout_low_per_min:
        idx = max(idx, 1)

    if rolloff_hz is not None:
        if rolloff_hz < s.rolloff_severe_hz:
            idx = max(idx, 2)
        elif rolloff_hz < s.rolloff_slight_hz:
            idx = max(idx, 1)

    if speech_rms_dbfs is not None:
        if speech_rms_dbfs < s.volume_severe_dbfs:
            idx = max(idx, 2)
        elif speech_rms_dbfs < s.volume_slight_dbfs:
            idx = max(idx, 1)

    if (
        pesq is not None
        and pesq < s.pesq_severe_backstop
        and snr_db is not None
        and snr_db > s.snr_no_excuse_db
    ):
        idx = max(idx, 2)

    return _LEVELS[idx], False


def _clipping_ratio(samples: np.ndarray, sr: int) -> float:
    try:
        # Real clipdetect API (v0.1.5) differs from a naive (samples, sr) -> (count,
        # total) guess: detect_clipping(samples_array, max_threshold=0.995,
        # min_threshold=0.995) -> (clipping_sections: list[dict], total_clipped:
        # int). No `sr` argument at all — it works purely in sample-index space, and
        # the thresholds are fractions of the array's own min/max (that's *why* it
        # survives normalization: a hard-clipped plateau is still a plateau at
        # whatever the new peak is after gain is applied). The first return value is
        # a list of {"start", "end"} sample-index dicts (for locating the clipped
        # regions), not a count — total_clipped_samples (2nd value) is the count we
        # want. Confirmed by reading clipdetect/__init__.py directly (task 7 brief's
        # authorized adaptation path) and by running it against the 3 real sample
        # calls (see task-7-report.md): fast (<0.2s even on the 172s call_003) and
        # gives a near-zero ratio (~1e-6) on all three, consistent with them all
        # being labeled "clear".
        from clipdetect import detect_clipping

        _sections, clipped = detect_clipping(samples)
        return float(clipped) / max(int(samples.size), 1)
    except Exception:
        # Fallback: plateau heuristic — fraction of samples within 0.1% of running max.
        peak = float(np.max(np.abs(samples))) or 1.0
        return float(np.mean(np.abs(samples) >= 0.999 * peak))


def _dropout_count_in_segment(samples: np.ndarray, sr: int, seg: Segment, min_run: int) -> int:
    """Vectorized run-length scan: count zero-runs of length >= min_run that start
    AND end strictly inside [seg.start, seg.end) — i.e. never touching the
    segment's own edges. A run touching an edge is a VAD-boundary artifact (fade
    in/out, or the VAD cutting a hair into real silence), not a mid-speech dropout."""
    lo, hi = int(seg.start * sr), int(seg.end * sr)
    chunk = samples[lo:hi]
    n = chunk.size
    if n < min_run + 2:  # no room for an interior run with a real sample on each side
        return 0
    is_zero = np.abs(chunk) < 1e-4
    # Sentinel-padded edge detection: diff==1 marks a run start index (in chunk
    # coordinates), diff==-1 marks the exclusive end index of that run.
    padded = np.concatenate(([False], is_zero, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int(np.sum((ends - starts >= min_run) & (starts > 0) & (ends < n)))


def _dropout_count(samples: np.ndarray, sr: int, vad: VadMap) -> int:
    """Hard near-zero runs (|sample| < 1e-4, >= 50ms) across all VAD speech
    segments, each required to start and end strictly inside its segment."""
    min_run = int(round(0.050 * sr))
    return sum(_dropout_count_in_segment(samples, sr, seg, min_run) for seg in vad.speech)


def _spectral_rolloff(samples: np.ndarray, sr: int, vad: VadMap) -> float | None:
    """Energy-weighted mean, across 32ms/16ms-hop Hann-windowed frames drawn only
    from VAD speech segments, of each frame's 95%-energy spectral rolloff point (the
    frequency below which 95% of that frame's spectral energy sits — the standard
    MIR "spectral rolloff" feature). Frame values are weighted by their own energy
    when combined into one call-level number, so near-silent dips inside a speech
    segment (stop consonants, brief pauses) don't skew the estimate. Telephony
    wideband speech should sit well above 2.2kHz; muffled/band-limited audio falls
    below (see rolloff_severe_hz/rolloff_slight_hz in config.py). None if there are
    no full 32ms speech frames to measure (no speech at all, or every speech segment
    is shorter than one frame)."""
    frame_n = int(round(0.032 * sr))
    hop_n = int(round(0.016 * sr))
    window = np.hanning(frame_n)
    freqs = np.fft.rfftfreq(frame_n, d=1.0 / sr)

    rolloffs: list[float] = []
    weights: list[float] = []
    for seg in vad.speech:
        lo, hi = int(seg.start * sr), int(seg.end * sr)
        chunk = samples[lo:hi]
        for start in range(0, chunk.size - frame_n + 1, hop_n):
            frame = chunk[start : start + frame_n] * window
            power = np.abs(np.fft.rfft(frame)) ** 2
            total_energy = float(power.sum())
            if total_energy <= 0.0:
                continue
            idx = int(np.searchsorted(np.cumsum(power), 0.95 * total_energy))
            rolloffs.append(float(freqs[min(idx, len(freqs) - 1)]))
            weights.append(total_energy)

    if not rolloffs:
        return None
    return float(np.average(rolloffs, weights=weights))


def _speech_rms_dbfs(samples: np.ndarray, sr: int, vad: VadMap) -> float | None:
    """Speech-segment RMS in dBFS (0 dBFS == full-scale amplitude 1.0). None if
    there's no speech to measure."""
    parts = [samples[int(seg.start * sr) : int(seg.end * sr)] for seg in vad.speech]
    speech = np.concatenate(parts) if parts else np.empty(0, dtype=samples.dtype)
    if speech.size == 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(speech))))
    return float(20.0 * np.log10(max(rms, 1e-8)))


_SQUIM = None


def _squim():
    global _SQUIM
    if _SQUIM is None:
        from torchaudio.pipelines import SQUIM_OBJECTIVE

        _SQUIM = SQUIM_OBJECTIVE.get_model()
    return _SQUIM


def analyze_quality(
    samples: np.ndarray, sr: int, vad: VadMap, snr_db: float | None
) -> QualityResult:
    """Deterministic channel evidence (clipping/dropouts/rolloff/volume) drives the
    rating; SQUIM runs as a noise-conditioned backstop only (see rate_quality).
    pesq/stoi/si_sdr are still surfaced on QualityResult for diagnostics even though
    only pesq feeds the decision. `vad` and `snr_db` are precomputed by the caller
    (analyze_vad / noise.snr_db) so this module never has to load or re-run either
    model itself."""
    import torch

    assert sr == 16000, "SQUIM expects 16 kHz input"

    clip_ratio = _clipping_ratio(samples, sr)
    dropouts = _dropout_count(samples, sr, vad)
    speech_s = sum(seg.end - seg.start for seg in vad.speech)
    dropouts_per_min = (dropouts / (speech_s / 60.0)) if speech_s > 0 else 0.0
    rolloff_hz = _spectral_rolloff(samples, sr, vad)
    speech_rms_dbfs = _speech_rms_dbfs(samples, sr, vad)

    # SQUIM's memory grows superlinearly with input length (measured peak footprint:
    # 5s=0.44GB, 15s=1.5GB, 30s=4.6GB, 60s=~14GB — swamped a 16GB machine into swap).
    # Score the middle 15s window: representative for the <1.3 backstop gate while
    # keeping peak memory ~1.5GB.
    max_n = 15 * sr
    x = samples if samples.size <= max_n else samples[(samples.size - max_n) // 2 :][:max_n]
    pesq = stoi = si_sdr = None
    try:
        with torch.inference_mode():
            stoi_t, pesq_t, si_sdr_t = _squim()(torch.from_numpy(x)[None, :])
        stoi, pesq, si_sdr = float(stoi_t), float(pesq_t), float(si_sdr_t)
    except Exception:
        pass  # backstop simply can't fire; deterministic evidence still applies

    rating, override = rate_quality(
        clipping_ratio=clip_ratio,
        dropouts_per_min=dropouts_per_min,
        rolloff_hz=rolloff_hz,
        speech_rms_dbfs=speech_rms_dbfs,
        pesq=pesq,
        snr_db=snr_db,
    )
    return QualityResult(
        rating=rating,
        pesq=pesq,
        stoi=stoi,
        si_sdr=si_sdr,
        clipping_ratio=clip_ratio,
        clipping_override=override,
        dropouts_per_min=dropouts_per_min,
        rolloff_hz=rolloff_hz,
        speech_rms_dbfs=speech_rms_dbfs,
    )
