"""E1: ask Gemini a FOCUSED noise question on speech-free gap audio only.
Hypothesis: full-call prompting has never once confirmed noise (0 live
triggers across all testing); with no conversation to attend to, the model
may finally hear the bed. A negative result is publishable as-is."""

import argparse
import json
from pathlib import Path

import numpy as np

from autoace_audio.analyzers.vad import VadMap, analyze_vad
from autoace_audio.audio_io import encode_opus_ogg, load_audio
from autoace_audio.config import get_settings
from eval.experiments.common import (
    ANCHORS,
    DATA_DIR,
    SpendGuard,
    gemini_cost,
    load_truth,
    log_run,
)

VALIDATION_DIR = DATA_DIR / "validation"
EST_COST_PER_RUN = 0.02  # 12 clips, short gap audio

GAP_SCHEMA = {
    "type": "object",
    "properties": {
        "background_noise_present": {"type": "boolean"},
        "background_noise_type": {"type": "string"},
        "character": {"type": "string", "enum": ["constant", "intermittent", "none"]},
    },
    "required": ["background_noise_present", "background_noise_type", "character"],
}

PROMPT = (
    "You are hearing ONLY the between-speech moments (silences between turns) "
    "of one phone call, joined together. There is no conversation to analyze. "
    "Is meaningful background sound present (TV, music, static, hum, traffic, "
    "chatter, machinery...)? Faint line hiss alone does not count. Give a "
    "concise type label like 'TV', 'static', 'electrical hum', or '' if none. "
    "Return JSON only."
)


def concat_gaps(
    samples: np.ndarray, sr: int, vad: VadMap, min_gap_s: float = 1.0, cap_s: float = 60.0
) -> np.ndarray:
    parts, kept = [], 0.0
    for g in vad.gaps:
        dur = g.end - g.start
        if dur < min_gap_s:
            continue
        take = min(dur, cap_s - kept)
        if take <= 0:
            break
        parts.append(samples[int(g.start * sr) : int((g.start + take) * sr)])
        kept += take
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def _noise_clips() -> dict[str, dict]:
    """9 synthetic noise clips + their truth, from the validation manifest."""
    import csv

    out = {}
    with open(VALIDATION_DIR / "validation_manifest.csv", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["kind"] == "noise_aug" and row["truth"].strip():
                out[row["name"]] = json.loads(row["truth"])
    return out


def ask_gemini_gaps(blob: bytes) -> tuple[dict, float]:
    from google import genai
    from google.genai import types

    s = get_settings()
    client = genai.Client(api_key=s.gemini_api_key)
    resp = client.models.generate_content(
        model=s.gemini_model,
        contents=[types.Part.from_bytes(data=blob, mime_type="audio/ogg"), PROMPT],
        config=types.GenerateContentConfig(
            temperature=s.gemini_temperature,
            response_mime_type="application/json",
            response_schema=GAP_SCHEMA,
        ),
    )
    data = json.loads(resp.text)
    usage = getattr(resp, "usage_metadata", None)
    cost = gemini_cost(
        getattr(usage, "prompt_token_count", None), getattr(usage, "candidates_token_count", None)
    )
    return data, cost


def run_once(run_idx: int) -> dict:
    guard = SpendGuard()
    guard.check(EST_COST_PER_RUN)
    truth_anchors = load_truth()
    targets: list[tuple[Path, dict]] = [
        (
            DATA_DIR / n,
            {
                "background_noise_present": truth_anchors[n]["background_noise_present"],
                "background_noise_type": truth_anchors[n]["background_noise_type"],
            },
        )
        for n in ANCHORS
    ] + [(VALIDATION_DIR / n, t) for n, t in sorted(_noise_clips().items())]
    per_clip, cost = {}, 0.0
    for path, truth in targets:
        audio = load_audio(path)
        vad = analyze_vad(audio.samples, audio.sr)
        gaps = concat_gaps(audio.samples, audio.sr, vad)
        if gaps.size < 2 * audio.sr:  # <2s of gap audio: not applicable
            per_clip[path.name] = {
                "skipped": True,
                "gap_seconds": gaps.size / audio.sr,
                "truth": truth,
            }
            continue
        pred, c = ask_gemini_gaps(encode_opus_ogg(gaps, audio.sr))
        cost += c
        per_clip[path.name] = {
            "skipped": False,
            "gap_seconds": round(gaps.size / audio.sr, 1),
            "pred": pred,
            "truth": truth,
            "present_correct": bool(pred["background_noise_present"])
            == bool(truth["background_noise_present"]),
        }
    guard.add(cost)
    payload = {"exp": "exp1_gap_noise", "run": run_idx, "cost_usd": cost, "per_clip": per_clip}
    log_run("exp1_gap_noise", run_idx, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    for i in range(1, args.runs + 1):
        p = run_once(i)
        hits = sum(
            1 for v in p["per_clip"].values() if not v.get("skipped") and v["present_correct"]
        )
        n = sum(1 for v in p["per_clip"].values() if not v.get("skipped"))
        print(f"run {i}: presence {hits}/{n}, ${p['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
