"""Build a leakage-safe validation set from the 3 labeled calls:
1) VAD-aligned 15-45s segments (hand-label these once, in segments_labels.csv);
2) synthetic augmentations with KNOWN ground truth:
   - noise beds harvested from the real calls (TV from call_002 gaps, static from call_003)
     plus generated white/pink/babble/hum, mixed at controlled SNRs -> severity labels;
   - controlled degradations (clipping, dropouts) -> audio_quality labels.
Group key = source call: all derivatives of one call stay in one fold.

Controller amendment A (task-11-brief's dropout variant contradicted its own truth
label -- a single ~20/min variant labeled "slightly_impaired" when config's calibrated
bands are dropout_low=1.0/min (slight) / dropout_high=4.0/min (severe)): generates TWO
dropout variants at explicit target rates (~2/min slight, ~6/min severe), placing every
dropout strictly INSIDE a VAD speech segment (with margin from the segment edges) since
quality.py's `_dropout_count_in_segment` only counts near-zero runs that start and end
strictly inside a speech segment -- a dropout dropped at a random sample offset (the
brief's original approach) can land in a VAD gap or straddle a segment edge and simply
not count at all.

Note: call_001 (the clean source) has only ~13.9s of VAD speech in its 30.9s clip. At
that duration, round(2/min * 13.9s/60) and round(6/min * 13.9s/60) BOTH round to 1
dropout -- the two target rates become indistinguishable in practice, not because the
insertion logic is wrong but because the source clip is too short to host enough
independent dropout counts. `_loop_for_dropouts` tiles the clean audio (and rebuilds a
matching VadMap with time-shifted speech segments) so there is enough speech mass for
the 2/min and 6/min targets to resolve to different, clearly-in-band integer counts.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from autoace_audio.analyzers.vad import Segment, VadMap, analyze_vad, build_vad_map
from autoace_audio.audio_io import load_audio
from autoace_audio.config import get_settings

RNG = np.random.default_rng(7)
SR = 16000
DROPOUT_LOOP_REPEATS = 6  # -> ~83s of speech; 2/min and 6/min targets resolve to
# 3 and 8 dropouts respectively (achieved ~2.16/min, ~5.76/min) -- comfortably inside
# the slight (1.0, 4.0] and severe (>4.0) bands with real margin either side.


def _loop_for_dropouts(
    samples: np.ndarray, vad: VadMap, sr: int, repeats: int
) -> tuple[np.ndarray, VadMap]:
    """Tile `samples` `repeats` times and rebuild a matching VadMap (speech segments
    time-shifted per repeat) so there's enough total speech duration to make the
    2/min vs 6/min dropout targets resolve to clearly different integer counts."""
    looped = np.tile(samples, repeats)
    clip_len_s = samples.size / sr
    speech = [
        Segment(seg.start + r * clip_len_s, seg.end + r * clip_len_s)
        for r in range(repeats)
        for seg in vad.speech
    ]
    looped_vad = build_vad_map(
        speech, total_s=repeats * clip_len_s, long_silence_s=get_settings().long_silence_s
    )
    return looped, looped_vad


def _segments(samples, min_s=15.0, max_s=45.0):
    """Fixed-stride chopping: tiles the clip into contiguous [min_s, max_s)-second
    windows from the start. Purely duration-based -- does NOT align to VAD speech
    activity (the `vad` parameter this used to take was never referenced in the
    body and has been removed)."""
    total = samples.size / SR
    out, start = [], 0.0
    while start + min_s < total:
        end = min(start + max_s, total)
        out.append((start, end))
        start = end
    return out


def _mix_at_snr(
    clean: np.ndarray, noise: np.ndarray, snr_db: float, speech_rms: float
) -> np.ndarray:
    """Mix `noise` into `clean` so the resulting speech-vs-gap SNR (as measured by
    `analyze_noise.snr_db`, the SAME function the pipeline uses) lands near `snr_db`.

    Fixes a real bug found while running this harness for real (not a brief
    pseudocode nit): the brief's original version referenced the WHOLE clip's RMS
    (`np.sqrt(np.mean(clean**2))`) as the signal power. call_001 is only 45% speech
    by duration (`speech_ratio`) with near-silent gaps, so its whole-clip RMS
    (0.166) sits well below its speech-only RMS (0.246) -- using the diluted
    whole-clip figure as the SNR reference under-gains the injected noise relative
    to what `analyze_noise.snr_db` (speech RMS vs gap RMS) actually measures.
    Measured evidence: at nominal targets (18, 10, 2) dB the OLD code produced
    measured SNR of (21.3, 16.5, 10.6) dB for the TV bed alone -- 3-9dB of
    systematic offset, enough to silently shift "high" severity truth into "low"/
    "medium" territory. `speech_rms` must be precomputed from the clean signal's
    OWN VAD speech segments (not the mixed result) and passed in explicitly."""
    if noise.size < clean.size:
        noise = np.tile(noise, int(np.ceil(clean.size / noise.size)))
    noise = noise[: clean.size]
    p_c = speech_rms or 1e-8
    p_n = np.sqrt(np.mean(noise**2)) or 1e-8
    gain = p_c / (p_n * 10 ** (snr_db / 20))
    return np.clip(clean + gain * noise, -1.0, 1.0).astype(np.float32)


def _degrade_clip(clean: np.ndarray) -> np.ndarray:
    return np.clip(clean * 8.0, -0.55, 0.55).astype(np.float32) / 0.55 * 0.9


def _insert_dropouts(
    clean: np.ndarray,
    sr: int,
    vad: VadMap,
    rate_per_min: float,
    dropout_s: float = 0.25,
    margin_s: float = 0.10,
    rng: np.random.Generator = RNG,
) -> tuple[np.ndarray, float]:
    """Zero out `dropout_s`-long runs at ~`rate_per_min` occurrences per minute of
    SPEECH (matching quality.py's `dropouts_per_min` denominator), placed strictly
    inside VAD speech segments with `margin_s` clearance from each segment's edges
    so the run never touches -- and therefore is never discarded by --
    `_dropout_count_in_segment`'s edge check. Returns (degraded_samples,
    achieved_rate_per_min) so the caller can verify the realized density rather
    than trust the nominal target blindly.
    """
    x = clean.copy()
    dropout_n = int(round(dropout_s * sr))
    margin_n = int(round(margin_s * sr))
    speech_s = sum(seg.end - seg.start for seg in vad.speech)
    if speech_s <= 0:
        raise ValueError("no speech segments to place dropouts in")
    n_target = max(1, round(rate_per_min * speech_s / 60.0))

    eligible = [
        seg for seg in vad.speech if int((seg.end - seg.start) * sr) > dropout_n + 2 * margin_n
    ]
    if not eligible:
        raise ValueError("no speech segment large enough to host a dropout with margin")

    occupied: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(eligible))}
    placed = 0
    max_attempts = n_target * 200 + 500
    attempts = 0
    while placed < n_target and attempts < max_attempts:
        attempts += 1
        seg_i = int(rng.integers(0, len(eligible)))
        seg = eligible[seg_i]
        lo, hi = int(seg.start * sr), int(seg.end * sr)
        lo_i, hi_i = lo + margin_n, hi - margin_n - dropout_n
        if hi_i <= lo_i:
            continue
        start = int(rng.integers(lo_i, hi_i + 1))
        end = start + dropout_n
        if any(not (end <= s or start >= e) for s, e in occupied[seg_i]):
            continue
        occupied[seg_i].append((start, end))
        x[start:end] = 0.0
        placed += 1
    if placed < n_target:
        raise ValueError(
            f"only placed {placed}/{n_target} dropouts (rate={rate_per_min}/min) -- "
            "source clip's speech segments are too short/few for this density"
        )
    achieved_rate = placed / (speech_s / 60.0)
    return x, achieved_rate


def main(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    calls = {
        "call_001.ogg": {"noise": None},
        "call_002.ogg": {"noise": "TV"},
        "call_003.ogg": {"noise": "static"},
    }
    beds = {}
    clean_vad = None
    for name, meta in calls.items():
        audio = load_audio(data_dir / name)
        vad = analyze_vad(audio.samples, audio.sr)
        if name == "call_001.ogg":
            clean_vad = vad
        # harvest real noise beds from gaps >= 1s
        gap_audio = np.concatenate(
            [
                audio.samples[int(g.start * SR) : int(g.end * SR)]
                for g in vad.gaps
                if g.end - g.start >= 1.0
            ]
            or [np.zeros(0, dtype=np.float32)]
        )
        if meta["noise"] and gap_audio.size > SR:
            beds[meta["noise"]] = gap_audio
        for j, (s0, s1) in enumerate(_segments(audio.samples)):
            seg = audio.samples[int(s0 * SR) : int(s1 * SR)]
            seg_name = f"{Path(name).stem}_seg{j}.wav"
            sf.write(out_dir / seg_name, seg, SR)
            rows.append({"name": seg_name, "group": name, "kind": "segment", "truth": ""})
    # synthetic beds
    n = 30 * SR
    beds.setdefault("static", 0.3 * RNG.standard_normal(n).astype(np.float32))
    t = np.arange(n) / SR
    beds["electrical hum"] = (
        0.5 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 100 * t)
    ).astype(np.float32)
    # clean source = call_001 (labeled no-noise)
    clean_audio = load_audio(data_dir / "call_001.ogg").samples
    from autoace_audio.analyzers.noise import slice_segments

    clean_speech = slice_segments(clean_audio, SR, clean_vad.speech)
    clean_speech_rms = float(np.sqrt(np.mean(clean_speech**2))) if clean_speech.size else 1e-8
    for bed_name, bed in beds.items():
        for snr, sev in [(18.0, "low"), (10.0, "medium"), (2.0, "high")]:
            mixed = _mix_at_snr(clean_audio, bed, snr, clean_speech_rms)
            fname = f"aug_{bed_name.replace(' ', '_')}_snr{int(snr)}.wav"
            sf.write(out_dir / fname, mixed, SR)
            rows.append(
                {
                    "name": fname,
                    "group": "call_001.ogg",
                    "kind": "noise_aug",
                    "truth": json.dumps(
                        {
                            "background_noise_present": True,
                            "background_noise_severity": sev,
                        }
                    ),
                }
            )

    # --- quality augmentations ---
    clip_x = _degrade_clip(clean_audio)
    sf.write(out_dir / "aug_clip.wav", clip_x, SR)
    rows.append(
        {
            "name": "aug_clip.wav",
            "group": "call_001.ogg",
            "kind": "quality_aug",
            "truth": json.dumps({"audio_quality": "severely_impaired"}),
        }
    )

    # Amendment A: two dropout variants, density strictly matched to the config's
    # calibrated bands (dropout_low_per_min=1.0 -> slight, dropout_high_per_min=4.0
    # -> severe), dropouts placed strictly inside VAD speech segments so the
    # detector actually counts them. Looped source (see _loop_for_dropouts) so the
    # two target rates resolve to clearly different, in-band integer counts.
    looped_audio, looped_vad = _loop_for_dropouts(clean_audio, clean_vad, SR, DROPOUT_LOOP_REPEATS)
    for target_rate, quality_label, fname in [
        (2.0, "slightly_impaired", "aug_dropout_slight.wav"),
        (6.0, "severely_impaired", "aug_dropout_severe.wav"),
    ]:
        x, achieved_rate = _insert_dropouts(looped_audio, SR, looped_vad, target_rate)
        sf.write(out_dir / fname, x, SR)
        rows.append(
            {
                "name": fname,
                "group": "call_001.ogg",
                "kind": "quality_aug",
                "truth": json.dumps({"audio_quality": quality_label}),
            }
        )
        print(f"{fname}: target {target_rate}/min speech -> achieved {achieved_rate:.2f}/min")

    with open(out_dir / "validation_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "group", "kind", "truth"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} validation clips -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/validation"))
    a = ap.parse_args()
    main(a.data, a.out)
