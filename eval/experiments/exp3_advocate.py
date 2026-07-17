"""E3: two-pass tone. Pass 1 = shipping classify. Pass 2 re-sends the audio
with pass 1's verdict+rationale and instructs the model to argue the
STRONGEST case for a different reading before giving a final verdict.
Both the wins AND the regressions (correct answers flipped wrong) are the
result -- report them with equal prominence (spec §7)."""

import argparse
import json

from autoace_audio.analyzers.noise import analyze_noise
from autoace_audio.analyzers.tone.base import classify_tone
from autoace_audio.analyzers.tone.gemini_tone import GEMINI_RESPONSE_SCHEMA
from autoace_audio.analyzers.vad import analyze_vad
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

EST_COST_PER_RUN = 0.02  # pass 2 re-sends the audio once per clip
FIELDS = ["emotional_tone", "emotional_intensity"]

ADVOCATE_PROMPT = """A first analysis of this exact call concluded:
emotional_tone={tone}, emotional_intensity={intensity}.
Its reasoning: {rationale}

Your job now:
1. Argue the STRONGEST honest case that the customer's emotional_tone is
   actually DIFFERENT from that verdict, grounded in what you hear.
2. Then weigh both cases and give your FINAL verdict -- keep the original
   only if it genuinely survives the counter-argument.
Definitions (apply exactly): neutral=no clear positive or negative emotion;
satisfied=pleased, relieved, appreciative, or clearly positive;
frustrated=annoyed, impatient, or dissatisfied WITHOUT strong anger;
upset=clearly angry, agitated, or strongly dissatisfied;
distressed=highly emotional, overwhelmed, panicked, or crying.
Do NOT infer emotion from loudness alone. A single crude or slang phrase,
by itself, is not sufficient evidence of frustration -- judge the whole call.
Return JSON only."""


def advocate_pass(
    name: str, first_tone: str, first_intensity: str, rationale: str
) -> tuple[dict, float, dict]:
    from google import genai
    from google.genai import types

    s = get_settings()
    client = genai.Client(api_key=s.gemini_api_key)
    audio = load_audio(DATA_DIR / name)
    blob = encode_opus_ogg(audio.samples, audio.sr)
    prompt = ADVOCATE_PROMPT.format(
        tone=first_tone, intensity=first_intensity, rationale=rationale or "(none recorded)"
    )
    resp = client.models.generate_content(
        model=s.gemini_model,
        contents=[types.Part.from_bytes(data=blob, mime_type="audio/ogg"), prompt],
        config=types.GenerateContentConfig(
            temperature=s.gemini_temperature,
            response_mime_type="application/json",
            response_schema=GEMINI_RESPONSE_SCHEMA,
        ),
    )
    data = json.loads(resp.text)
    usage = getattr(resp, "usage_metadata", None)
    in_tok = getattr(usage, "prompt_token_count", None)
    out_tok = getattr(usage, "candidates_token_count", None)
    return data, gemini_cost(in_tok, out_tok), {"in": in_tok, "out": out_tok}


def run_once(run_idx: int) -> dict:
    guard = SpendGuard()
    guard.check(2 * EST_COST_PER_RUN)
    truth = load_truth()
    per_clip, cost = {}, 0.0
    for name in ANCHORS:
        audio = load_audio(DATA_DIR / name)
        vad = analyze_vad(audio.samples, audio.sr)
        noise = analyze_noise(audio.samples, audio.sr, vad)
        first = classify_tone("gemini", audio.samples, audio.sr, vad, noise.snr_db)
        first_tokens = {"in": first.raw.get("prompt_tokens"), "out": first.raw.get("output_tokens")}
        cost += gemini_cost(first_tokens["in"], first_tokens["out"])
        rationale = str(first.raw.get("response", {}).get("rationale", ""))
        final, c2, final_tokens = advocate_pass(
            name, first.tone.value, first.intensity.value, rationale
        )
        cost += c2
        pred = {
            "emotional_tone": final["emotional_tone"],
            "emotional_intensity": final["emotional_intensity"],
        }
        first_pred = {
            "emotional_tone": first.tone.value,
            "emotional_intensity": first.intensity.value,
        }
        per_clip[name] = {
            "first": first_pred,
            "final": pred,
            "flipped": pred["emotional_tone"] != first_pred["emotional_tone"],
            "correct": field_compare(pred, truth[name], FIELDS),
            "first_correct": field_compare(first_pred, truth[name], FIELDS),
            "tokens": {"first": first_tokens, "final": final_tokens},
        }
    guard.add(cost)
    payload = {"exp": "exp3_advocate", "run": run_idx, "cost_usd": cost, "per_clip": per_clip}
    log_run("exp3_advocate", run_idx, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    for i in range(1, args.runs + 1):
        p = run_once(i)
        flips = {k: v["flipped"] for k, v in p["per_clip"].items()}
        print(f"run {i}: ${p['cost_usd']:.4f} flips={flips}")


if __name__ == "__main__":
    main()
