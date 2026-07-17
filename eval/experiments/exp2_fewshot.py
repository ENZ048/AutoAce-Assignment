"""E2: two ~20s labeled audio exemplars prepended to the shipping prompt.
LEAVE-ONE-OUT: scoring call k uses excerpts from the other two anchors only.
Known caveat (report it): call_001's exemplar set is medium+medium -- the
anchor pool has no second 'high' call to teach from."""

import argparse
import json

import numpy as np

from autoace_audio.analyzers.tone.gemini_tone import GEMINI_RESPONSE_SCHEMA, build_prompt
from autoace_audio.analyzers.vad import VadMap, analyze_vad
from autoace_audio.audio_io import encode_opus_ogg, load_audio
from autoace_audio.config import get_settings
from eval.experiments.common import (
    ANCHORS,
    DATA_DIR,
    SpendGuard,
    field_compare,
    gemini_cost,
    load_truth,
    log_run,
)

EST_COST_PER_RUN = 0.03  # 3 targets, each with ~40s exemplar audio on top
FIELDS = ["emotional_tone", "emotional_intensity"]


def best_window(
    samples: np.ndarray, sr: int, vad: VadMap, win_s: float = 20.0
) -> tuple[float, float]:
    total = samples.size / sr
    if total <= win_s:
        return 0.0, total
    best_start, best_speech = 0.0, -1.0
    step = 1.0
    t = 0.0
    while t + win_s <= total:
        speech = sum(max(0.0, min(seg.end, t + win_s) - max(seg.start, t)) for seg in vad.speech)
        if speech > best_speech:  # strict > keeps earliest on ties
            best_start, best_speech = t, speech
        t += step
    return best_start, best_start + win_s


def _exemplar(name: str) -> tuple[bytes, str]:
    audio = load_audio(DATA_DIR / name)
    vad = analyze_vad(audio.samples, audio.sr)
    s0, s1 = best_window(audio.samples, audio.sr, vad)
    blob = encode_opus_ogg(audio.samples[int(s0 * audio.sr) : int(s1 * audio.sr)], audio.sr)
    return blob, load_truth()[name]["emotional_intensity"]


def classify_with_exemplars(target: str) -> tuple[dict, float]:
    from google import genai
    from google.genai import types

    s = get_settings()
    client = genai.Client(api_key=s.gemini_api_key)
    audio = load_audio(DATA_DIR / target)
    vad = analyze_vad(audio.samples, audio.sr)
    others = [n for n in ANCHORS if n != target]
    parts = []
    for i, name in enumerate(others):
        blob, intensity = _exemplar(name)
        parts.append(types.Part.from_bytes(data=blob, mime_type="audio/ogg"))
        parts.append(
            f"The excerpt above is EXAMPLE {chr(65 + i)}: a different call whose "
            f"correct emotional_intensity is '{intensity}'. It is calibration "
            f"only -- do not classify it."
        )
    parts.append(
        types.Part.from_bytes(data=encode_opus_ogg(audio.samples, audio.sr), mime_type="audio/ogg")
    )
    parts.append(
        "Now classify ONLY this final full call.\n\n"
        + build_prompt(audio.samples.size / audio.sr, None, vad.speech_ratio)
    )
    resp = client.models.generate_content(
        model=s.gemini_model,
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=s.gemini_temperature,
            response_mime_type="application/json",
            response_schema=GEMINI_RESPONSE_SCHEMA,
        ),
    )
    data = json.loads(resp.text)
    usage = getattr(resp, "usage_metadata", None)
    return data, gemini_cost(
        getattr(usage, "prompt_token_count", None), getattr(usage, "candidates_token_count", None)
    )


def run_once(run_idx: int) -> dict:
    guard = SpendGuard()
    guard.check(EST_COST_PER_RUN)
    truth = load_truth()
    per_clip, cost = {}, 0.0
    for name in ANCHORS:
        data, c = classify_with_exemplars(name)
        cost += c
        pred = {
            "emotional_tone": data["emotional_tone"],
            "emotional_intensity": data["emotional_intensity"],
        }
        per_clip[name] = {
            "pred": pred,
            "correct": field_compare(pred, truth[name], FIELDS),
            "cost_usd": c,
        }
    guard.add(cost)
    payload = {"exp": "exp2_fewshot", "run": run_idx, "cost_usd": cost, "per_clip": per_clip}
    log_run("exp2_fewshot", run_idx, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    for i in range(1, args.runs + 1):
        p = run_once(i)
        print(
            f"run {i}: ${p['cost_usd']:.4f}",
            {k: v["pred"]["emotional_intensity"] for k, v in p["per_clip"].items()},
        )


if __name__ == "__main__":
    main()
