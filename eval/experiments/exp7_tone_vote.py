"""E7: self-consistency majority-vote harness for the shipping tone/
intensity classification (autoace_audio.analyzers.tone.gemini_tone). Casts
VOTES_PER_CLIP=3 independent Gemini calls per clip per run at
VOTE_TEMPERATURE=0.7 (diversity sampling -- the experiment itself) and
majority-votes emotional_tone and emotional_intensity independently.

Baseline tone at temp 0.1 is run-deterministic: out/experiments/
exp0_baseline_run{1,2,3}.json agree byte-for-byte on every clip's
emotional_tone/emotional_intensity across all 3 live sessions, so voting at
0.1 would trivially null (3 identical votes every time). The experiment is
therefore voting at temperature 0.7 (pinned per the brief -- "it is the
point of the test") vs the temp-0.1 single-call baseline. Everything else is
byte-identical, import-not-copy, to the shipping arm: build_prompt,
GEMINI_RESPONSE_SCHEMA, and the model id (read from settings, same as
shipping -- never hardcoded to a different literal). ONLY the temperature
and vote count differ from exp0_baseline.

Majority rule (brief's pinned rule, verbatim; applied INDEPENDENTLY to
emotional_tone and emotional_intensity -- one field's tie shape never
affects the other field's verdict):
  winner = the value shared by >=2 of the 3 votes ("2-1" or "3-same" shape).
  3-way split (all 3 votes disagree) -> the vote with the highest
  tone_confidence; further tie (equal top confidence) -> first vote
  (original call order).
Dispersion ("note per-clip vote dispersion (3-same / 2-1 / 3-way) --
dispersion itself is a finding, it measures how unstable the model is at
0.7 on these calls") is classified per field per clip per run purely from
the AGREEMENT SHAPE of the 3 votes -- confidence never enters into the
dispersion label, only into breaking a 3-way tie's winner -- and logged
alongside the majority verdict.

Cost model: unlike E6 (which ADDS a voted question on top of the shipping
pipeline), E7's voting REPLACES the shipping arm's single tone call with 3.
Tone is the only billed step in the shipping pipeline (VAD/AED/quality are
local models with zero API cost -- out/bakeoff.md's $0.00146/audio-min
headline is essentially the tone call's own cost), so this module's OWN
measured mean run cost, divided by its own measured audio-minutes, IS the
"voted-tone configuration" $/audio-min directly -- no additive marginal
formula needed (contrast exp6_gap_vote.operating_point, which adds a
marginal on top of a baseline that stays separately billed)."""

import argparse
import json
from collections import Counter

from autoace_audio.analyzers.noise import analyze_noise
from autoace_audio.analyzers.tone.gemini_tone import GEMINI_RESPONSE_SCHEMA, build_prompt
from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import encode_opus_ogg, load_audio
from autoace_audio.config import get_settings
from eval.experiments.common import (
    ANCHORS,
    DATA_DIR,
    GEMINI_LITE_IN,
    GEMINI_LITE_OUT,
    SpendGuard,
    field_compare,
    gemini_cost,
    load_truth,
    log_run,
)

VOTES_PER_CLIP = 3

# Diversity sampling temperature -- the experiment itself (brief: "pin this;
# it is the point of the test"). Hardcoded, deliberately NOT read from
# settings.gemini_temperature (shipping default 0.1, asserted unchanged by
# test_baseline_gemini_temperature_setting_is_unchanged_at_0_1) -- voting at
# 0.1 would trivially null since the shipping arm is already
# run-deterministic there (exp0_baseline_run{1,2,3}.json agree exactly).
VOTE_TEMPERATURE = 0.7

FIELDS = ["emotional_tone", "emotional_intensity"]

# 3 anchors x 3 votes = 9 tone calls/run. exp0's OWN single-call anchors run
# measured $0.0045-0.0046/run (out/experiments/exp0_baseline_run*.json) for
# 3 calls; 3x that (9 calls, same audio, same token profile modulo sampling
# variance at the higher temperature) is a generous ceiling.
EST_COST_PER_RUN = 0.02

# Bake-off headline shipping cost (out/bakeoff.md, README.md) -- see module
# docstring: the shipping pipeline's only billed step is this same tone
# call, so this is directly comparable to E7's own measured $/audio-min.
# Same constant exp6_gap_vote.py uses for its baseline half.
BASELINE_PER_MIN = 0.00146


def _vote_confidence(vote: dict) -> float:
    return float(vote["tone_confidence"])


def _dispersion_shape(values: list[str]) -> str:
    """Classifies a 3-vote value list's agreement shape -- "3-same" (all
    equal), "2-1" (clean majority + one dissenter), "3-way" (all three
    disagree, the tie-break shape). Pure function of the AGREEMENT shape
    only -- confidence never enters here, only into breaking a 3-way tie's
    winner (see _majority_field)."""
    top = max(Counter(values).values())
    if top == 3:
        return "3-same"
    if top == 2:
        return "2-1"
    return "3-way"


def _majority_field(votes: list[dict], field: str) -> tuple[str, str]:
    """One field's majority verdict + dispersion shape, voted INDEPENDENTLY
    of every other field (brief: "Same rule for emotional_intensity voted
    independently"). Brief's pinned rule: winner = value shared by >=2 of 3;
    3-way split (all different) -> highest tone_confidence, further tie
    (equal top confidence) -> first vote in original order."""
    values = [v[field] for v in votes]
    shape = _dispersion_shape(values)
    if shape != "3-way":
        counts = Counter(values)
        winner = max(counts, key=lambda val: counts[val])
        return winner, shape
    confidences = [_vote_confidence(v) for v in votes]
    best_confidence = max(confidences)
    winner = next(v for v, c in zip(values, confidences, strict=True) if c == best_confidence)
    return winner, shape


def majority_vote(votes: list[dict]) -> dict:
    """3 raw vote dicts (each carrying at least emotional_tone,
    emotional_intensity, tone_confidence) -> majority verdict for both
    fields, voted independently, plus each field's dispersion shape.
    Returns {"emotional_tone": ..., "emotional_intensity": ..., "dispersion":
    {"emotional_tone": ..., "emotional_intensity": ...}}."""
    tone, tone_dispersion = _majority_field(votes, "emotional_tone")
    intensity, intensity_dispersion = _majority_field(votes, "emotional_intensity")
    return {
        "emotional_tone": tone,
        "emotional_intensity": intensity,
        "dispersion": {
            "emotional_tone": tone_dispersion,
            "emotional_intensity": intensity_dispersion,
        },
    }


def _sum_tokens(token_dicts: list[dict]) -> dict:
    return {
        "in": sum((t.get("in") or 0) for t in token_dicts),
        "out": sum((t.get("out") or 0) for t in token_dicts),
    }


def _ask_gemini_tone_vote(blob: bytes, prompt: str) -> tuple[dict, float, dict]:
    """One vote: a single VOTE_TEMPERATURE=0.7 Gemini call against the given
    pre-encoded audio blob + prebuilt prompt. Model id (from settings, same
    as shipping), response schema, and prompt text are byte-identical,
    import-not-copy, to the shipping arm -- ONLY the temperature differs."""
    from google import genai
    from google.genai import types

    s = get_settings()
    client = genai.Client(api_key=s.gemini_api_key)
    resp = client.models.generate_content(
        model=s.gemini_model,
        contents=[types.Part.from_bytes(data=blob, mime_type="audio/ogg"), prompt],
        config=types.GenerateContentConfig(
            temperature=VOTE_TEMPERATURE,
            response_mime_type="application/json",
            response_schema=GEMINI_RESPONSE_SCHEMA,
        ),
    )
    data = json.loads(resp.text)
    usage = getattr(resp, "usage_metadata", None)
    in_tok = getattr(usage, "prompt_token_count", None)
    out_tok = getattr(usage, "candidates_token_count", None)
    cost = gemini_cost(in_tok, out_tok)
    return data, cost, {"in": in_tok, "out": out_tok}


def _vote_tone(name: str) -> dict:
    """One clip's full VOTES_PER_CLIP-vote round. Audio load/VAD/noise/
    encode and prompt construction happen ONCE (deterministic -- would add
    cost, not signal, if repeated per vote, same convention as
    exp6_gap_vote._vote_gap_listening); only the Gemini call itself repeats
    VOTES_PER_CLIP times against that SAME blob + SAME prompt -- that
    repeat, at VOTE_TEMPERATURE, is the variance under test."""
    audio = load_audio(DATA_DIR / name)
    audio_s = audio.samples.size / audio.sr
    vad = analyze_vad(audio.samples, audio.sr)
    noise = analyze_noise(audio.samples, audio.sr, vad)
    blob = encode_opus_ogg(audio.samples, audio.sr)
    prompt = build_prompt(audio_s, noise.snr_db, vad.speech_ratio)

    votes = []
    for _ in range(VOTES_PER_CLIP):
        data, cost, tokens = _ask_gemini_tone_vote(blob, prompt)
        votes.append(
            {
                "emotional_tone": data["emotional_tone"],
                "emotional_intensity": data["emotional_intensity"],
                "tone_confidence": data["tone_confidence"],
                "tokens": tokens,
                "cost_usd": cost,
            }
        )

    return {
        "audio_s": round(audio_s, 3),
        "votes": votes,
        "majority": majority_vote(votes),
        "tokens": _sum_tokens([v["tokens"] for v in votes]),
        "cost_usd": sum(v["cost_usd"] for v in votes),
    }


def run_once(run_idx: int) -> dict:
    guard = SpendGuard()
    guard.check(EST_COST_PER_RUN)
    truth = load_truth()
    per_clip: dict = {}
    cost = 0.0
    for name in ANCHORS:
        result = _vote_tone(name)
        cost += result["cost_usd"]
        pred = {
            "emotional_tone": result["majority"]["emotional_tone"],
            "emotional_intensity": result["majority"]["emotional_intensity"],
        }
        per_clip[name] = {
            **result,
            "pred": pred,
            "correct": field_compare(pred, truth[name], FIELDS),
            "dispersion": result["majority"]["dispersion"],
        }
    guard.add(cost)
    audio_minutes = round(sum(v["audio_s"] for v in per_clip.values()) / 60.0, 6)
    payload = {
        "exp": "exp7_tone_vote",
        "run": run_idx,
        "cost_usd": cost,
        "model": get_settings().gemini_model,
        "vote_temperature": VOTE_TEMPERATURE,
        "votes_per_clip": VOTES_PER_CLIP,
        "audio_minutes": audio_minutes,
        "pricing": {
            "in_per_1m": GEMINI_LITE_IN,
            "out_per_1m": GEMINI_LITE_OUT,
            "source": "shipping default rate (eval/experiments/common.py); the tone-vote "
            "calls use the shipping model at its default audio pricing, same convention as "
            "exp6_gap_vote.py.",
        },
        "per_clip": per_clip,
    }
    log_run("exp7_tone_vote", run_idx, payload)
    return payload


def cost_per_min(runs: list[dict]) -> dict:
    """This module's own measured $/audio-min for the full 3-vote tone
    configuration -- see module docstring: voting REPLACES the shipping
    arm's only billed step, so the measured mean run cost IS the "voted
    tone" $/audio-min, directly comparable to the bake-off headline
    (single-call baseline)."""
    if not runs:
        raise ValueError("no runs to compute a cost-per-minute from")
    audio_minutes = [r["audio_minutes"] for r in runs]
    if max(audio_minutes) - min(audio_minutes) > 1e-6:
        raise ValueError(
            f"runs measured different audio_minutes {audio_minutes} -- not directly comparable"
        )
    mean_run_cost = sum(r["cost_usd"] for r in runs) / len(runs)
    voted_per_min = mean_run_cost / audio_minutes[0]
    return {
        "mean_run_cost_usd": mean_run_cost,
        "audio_minutes": audio_minutes[0],
        "voted_tone_per_min": voted_per_min,
        "baseline_per_min": BASELINE_PER_MIN,
        "multiple_of_baseline": voted_per_min / BASELINE_PER_MIN,
    }


def dispersion_summary(runs: list[dict]) -> dict[str, Counter]:
    """Tally of per-clip dispersion shapes ("3-same"/"2-1"/"3-way") across
    all clips in all given runs, independently per field -- brief: "note
    per-clip vote dispersion... dispersion itself is a finding". Feeds the
    report's dispersion table."""
    counts: dict[str, Counter] = {f: Counter() for f in FIELDS}
    for run in runs:
        for clip in run["per_clip"].values():
            for f in FIELDS:
                counts[f][clip["dispersion"][f]] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    runs = []
    for i in range(1, args.runs + 1):
        p = run_once(i)
        runs.append(p)
        hits = {f: sum(1 for v in p["per_clip"].values() if v["correct"][f]) for f in FIELDS}
        print(
            f"run {i}: tone {hits['emotional_tone']}/3, intensity {hits['emotional_intensity']}/3,"
            f" ${p['cost_usd']:.4f}"
        )
    cpm = cost_per_min(runs)
    print(
        f"voted-tone $/audio-min: ${cpm['voted_tone_per_min']:.6f} "
        f"({cpm['multiple_of_baseline']:.2f}x baseline ${cpm['baseline_per_min']:.5f})"
    )
    disp = dispersion_summary(runs)
    for f in FIELDS:
        print(f"{f} dispersion: {dict(disp[f])}")


if __name__ == "__main__":
    main()
