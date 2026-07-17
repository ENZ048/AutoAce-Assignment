"""exp0: shipping-config baseline, run 3x over the anchors. Every later
lever's delta is measured against THIS distribution, not a single run."""

import argparse

from autoace_audio.analyzers.noise import analyze_noise
from autoace_audio.analyzers.tone.base import classify_tone
from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import load_audio
from eval.experiments.common import (
    ANCHORS,
    DATA_DIR,
    SpendGuard,
    field_compare,
    gemini_cost,
    load_truth,
    log_run,
)

FIELDS = ["emotional_tone", "emotional_intensity", "speaker_overlap_present"]
EST_COST_PER_RUN = 0.01  # 3 anchors, ~8k in + ~300 out tokens


def run_once(run_idx: int) -> dict:
    truth = load_truth()
    guard = SpendGuard()
    guard.check(EST_COST_PER_RUN)
    per_clip, cost = {}, 0.0
    for name in ANCHORS:
        audio = load_audio(DATA_DIR / name)
        vad = analyze_vad(audio.samples, audio.sr)
        noise = analyze_noise(audio.samples, audio.sr, vad)
        r = classify_tone("gemini", audio.samples, audio.sr, vad, noise.snr_db)
        pred = {
            "emotional_tone": r.tone.value,
            "emotional_intensity": r.intensity.value,
            "speaker_overlap_present": bool(r.overlap_opinion),
            "noise_opinion": r.noise_opinion,
        }
        c = gemini_cost(r.raw.get("prompt_tokens"), r.raw.get("output_tokens"))
        cost += c
        per_clip[name] = {
            "pred": pred,
            "correct": field_compare(pred, truth[name], FIELDS),
            "tokens": {"in": r.raw.get("prompt_tokens"), "out": r.raw.get("output_tokens")},
            "cost_usd": c,
        }
    guard.add(cost)
    payload = {"exp": "exp0_baseline", "run": run_idx, "cost_usd": cost, "per_clip": per_clip}
    log_run("exp0_baseline", run_idx, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    for i in range(1, args.runs + 1):
        p = run_once(i)
        print(f"run {i}: ${p['cost_usd']:.4f}", {k: v["correct"] for k, v in p["per_clip"].items()})


if __name__ == "__main__":
    main()
