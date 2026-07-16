"""Technical channel quality ONLY (distortion/clipping/muffling/dropouts) —
independent of background noise by design. SQUIM (reference-free PESQ/STOI/SI-SDR)
+ clipdetect (clipping survives normalization; peak checks don't see it)."""

from dataclasses import dataclass

import numpy as np

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


_LEVELS = [AudioQuality.CLEAR, AudioQuality.SLIGHTLY_IMPAIRED, AudioQuality.SEVERELY_IMPAIRED]


def rate_quality(
    pesq: float | None, stoi: float | None, clipping_ratio: float
) -> tuple[AudioQuality, bool]:
    s = get_settings()
    if clipping_ratio > s.clipping_ratio_max:
        return AudioQuality.SEVERELY_IMPAIRED, True
    if pesq is None:
        return AudioQuality.CLEAR, False  # no evidence of impairment
    if pesq >= s.pesq_clear:
        idx = 0
    elif pesq >= s.pesq_slight:
        idx = 1
    else:
        idx = 2
    if stoi is not None and stoi < s.stoi_floor:
        idx = min(idx + 1, 2)
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


_SQUIM = None


def _squim():
    global _SQUIM
    if _SQUIM is None:
        from torchaudio.pipelines import SQUIM_OBJECTIVE

        _SQUIM = SQUIM_OBJECTIVE.get_model()
    return _SQUIM


def analyze_quality(samples: np.ndarray, sr: int) -> QualityResult:
    import torch

    assert sr == 16000, "SQUIM expects 16 kHz input"
    # SQUIM is O(n^2)-ish in memory on long clips: score the middle 60s window.
    max_n = 60 * sr
    x = samples if samples.size <= max_n else samples[(samples.size - max_n) // 2:][:max_n]
    pesq = stoi = si_sdr = None
    try:
        with torch.inference_mode():
            stoi_t, pesq_t, si_sdr_t = _squim()(torch.from_numpy(x)[None, :])
        stoi, pesq, si_sdr = float(stoi_t), float(pesq_t), float(si_sdr_t)
    except Exception:
        pass  # rating falls back to clipping-only evidence
    clip_ratio = _clipping_ratio(samples, sr)
    rating, override = rate_quality(pesq, stoi, clip_ratio)
    return QualityResult(
        rating=rating, pesq=pesq, stoi=stoi, si_sdr=si_sdr,
        clipping_ratio=clip_ratio, clipping_override=override,
    )
