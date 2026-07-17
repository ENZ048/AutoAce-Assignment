"""E4: identical shipping prompt/schema, bigger model (gemini-3.5-flash).
The run log records the exact model id + the audio-token pricing used for
its cost column, with the price source, so the study table is auditable.

Model substitution note: the brief's original id, `gemini-3.1-flash`, does
not exist -- confirmed both by a hard 404 from Google's GetModel endpoint
and by its absence from the pricing page (see study-task-6-report.md's
pre-flight section). Per that report's escalation, a controller decision
(recorded in .superpowers/sdd/progress.md) substitutes `gemini-3.5-flash`
(GA, no preview tag) as the E4 lever. The alternate candidate,
`gemini-3-flash-preview` ($1.00/1M in, $3.00/1M out -- under the client's
$0.003/audio-min ceiling on rate alone, but preview status and a
minor-version step backward from the shipping 3.1-lite model), is recorded
as a possible E4b follow-up in the report; it is not implemented here."""

import argparse
import json

from autoace_audio.analyzers.noise import analyze_noise
from autoace_audio.analyzers.tone.gemini_tone import GEMINI_RESPONSE_SCHEMA, build_prompt
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

FLASH_MODEL = "gemini-3.5-flash"
# Controller-verified rates (NOT the brief's placeholder $1.00/$3.00, which
# coincidentally matches the rejected gemini-3-flash-preview candidate --
# see study-task-6-report.md). Source: https://ai.google.dev/gemini-api/docs/pricing,
# checked 2026-07-17. Do NOT reuse Flash-Lite pricing (common.GEMINI_LITE_IN/OUT,
# $0.50/$1.50) or the preview candidate's pricing ($1.00/$3.00) here.
FLASH_IN_PER_1M = 1.50
FLASH_OUT_PER_1M = 9.00
PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing (checked 2026-07-17)"

# ~4 audio-min total across the 3 anchors/run at $1.50/M in (audio billed
# ~32 tok/s per gemini_tone.py) + modest JSON output at $9/M out.
EST_COST_PER_RUN = 0.02
FIELDS = ["emotional_tone", "emotional_intensity", "speaker_overlap_present"]


def classify_flash(name: str) -> tuple[dict, float, dict]:
    from google import genai
    from google.genai import types

    s = get_settings()
    client = genai.Client(api_key=s.gemini_api_key)
    audio = load_audio(DATA_DIR / name)
    vad = analyze_vad(audio.samples, audio.sr)
    noise = analyze_noise(audio.samples, audio.sr, vad)
    resp = client.models.generate_content(
        model=FLASH_MODEL,
        contents=[
            types.Part.from_bytes(
                data=encode_opus_ogg(audio.samples, audio.sr), mime_type="audio/ogg"
            ),
            build_prompt(audio.samples.size / audio.sr, noise.snr_db, vad.speech_ratio),
        ],
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
    cost = gemini_cost(in_tok, out_tok, in_rate=FLASH_IN_PER_1M, out_rate=FLASH_OUT_PER_1M)
    return data, cost, {"in": in_tok, "out": out_tok}


def run_once(run_idx: int) -> dict:
    guard = SpendGuard()
    guard.check(EST_COST_PER_RUN)
    truth = load_truth()
    per_clip, cost = {}, 0.0
    for name in ANCHORS:
        data, c, tokens = classify_flash(name)
        cost += c
        pred = {
            "emotional_tone": data["emotional_tone"],
            "emotional_intensity": data["emotional_intensity"],
            "speaker_overlap_present": bool(data["speaker_overlap_present"]),
        }
        per_clip[name] = {
            "pred": pred,
            "correct": field_compare(pred, truth[name], FIELDS),
            "tokens": tokens,
            "cost_usd": c,
        }
    guard.add(cost)
    payload = {
        "exp": "exp4_flash",
        "run": run_idx,
        "cost_usd": cost,
        "model_id": FLASH_MODEL,  # brief's original mandatory interface field
        "model": FLASH_MODEL,  # standing amendment: new field, same value
        "pricing": {
            "in_per_1m": FLASH_IN_PER_1M,
            "out_per_1m": FLASH_OUT_PER_1M,
            "source": PRICING_SOURCE,
        },
        "per_clip": per_clip,
    }
    log_run("exp4_flash", run_idx, payload)
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
