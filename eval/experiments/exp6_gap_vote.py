"""E6: majority-vote harness for E1's gap-listening noise question. Asks the
SAME question (imported, not copied, from exp1_gap_noise) 3 independent
times per clip per run and reduces to a majority verdict, to test whether
voting stabilizes a win that has now been measured at two different
reliabilities in two different live sessions:
  - exp1_gap_noise's OWN session (Study Task 3, out/experiments/
    exp1_gap_noise_run{1,2,3}.json): call_003 static confirmed 3/3 runs.
  - combined.py's session (Study Task 8, out/experiments/combined_run{1,2,3}
    .json, per-clip "gap_listening" sub-object): the SAME question on the
    SAME clip confirmed only 1/3 runs (run 3 only; runs 1-2 came back blind,
    same as baseline's own present=false/type="").
Same clip, same prompt, same model, same temperature -- the only thing that
changed was the session. This module asks whether >=2-of-3 within-session
voting recovers session-to-session stability, without touching the shipping
question or its extraction logic (identity-imported below, never copied).

Scope: the 3 real anchors ONLY (call_001/002/003.ogg), not E1's own 9
synthetic noise-augmented clips. This matches the brief's own framing --
every named clip and expectation in the dispatch ("Anchors and expectations
to report against") is an anchor, none are synthetic -- and mirrors
combined.py's own established precedent for lever experiments ("the
combined stack's live run scores only the 3 real anchors ... even though the
original study design listed [the synthetic set] as in scope"). It also
keeps the operating-point cost math on the same audio-min basis as every
other row in the study's cost table (the 3 anchors' real combined duration),
not diluted by 9 short synthetic clips that were never part of that basis.

Majority rule (brief's pinned rule, verbatim):
  present = >=2 of 3 votes true.
  type = modal normalized (lowercase/strip) string among the present-true
  votes; if no modal winner (a tie -- either a 2-true-vote 1-1 split, or a
  3-true-vote 1-1-1 split), take the vote with the higher self-reported
  noise confidence IF the response carries one, else the first true vote
  (original call order). All ties toward absent.
Forward-compatibility note: GAP_SCHEMA (imported unchanged via
ask_gemini_gaps, per the byte-identical requirement above) has no
confidence field today, so the confidence branch is inert against live
votes -- implemented and unit-tested anyway, directly, against fabricated
vote dicts, because the brief pins it as part of the rule. It looks for a
"noise_confidence" key on the raw vote dict; anything else falls straight
through to "first true vote".

Operating point: the brief's formula, computed and reported explicitly --
  operating_point/audio-min = baseline ($0.00146, out/bakeoff.md headline)
                               + 3 x (measured single-gap-vote cost/audio-min)
using this module's OWN live measurement (mean per-run cost of the anchors'
gap-listening votes, divided by the anchors' real combined duration in
audio-minutes, measured fresh here via load_audio -- the same methodology
as the study doc's established E1-marginal footnote,
docs/experiments/2026-07-17-budget-accuracy-study.md footnote 3, which used
a hand-recorded 3.964018 audio-min constant from a one-time ffprobe
measurement; this module recomputes the equivalent number live every run
instead of hard-coding it, and should closely reproduce it)."""

import argparse
from collections import Counter

from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import encode_opus_ogg, load_audio
from autoace_audio.config import get_settings
from eval.experiments.common import (
    ANCHORS,
    DATA_DIR,
    GEMINI_LITE_IN,
    GEMINI_LITE_OUT,
    SpendGuard,
    load_truth,
    log_run,
)
from eval.experiments.exp1_gap_noise import ask_gemini_gaps, concat_gaps

VOTES_PER_CLIP = 3

# ~3 anchors x 3 votes on SHORT gap audio only (not full-call audio). E1's
# own 12-clip single-vote run measured $0.0037/run (Study Task 3); 3 anchors
# alone are a fraction of that per vote, and this module casts 3 votes each
# -- generous margin over the ~$0.001-0.002/run this module actually
# measures on just the anchors.
EST_COST_PER_RUN = 0.02

# Bake-off headline shipping cost (out/bakeoff.md, README.md) -- the
# "shipping" half of the brief's "shipping + voted gap question" operating
# -point formula.
BASELINE_PER_MIN = 0.00146


def _normalize_type(type_: str) -> str:
    return (type_ or "").strip().lower()


def _vote_confidence(vote: dict) -> float | None:
    """Self-reported noise confidence IF the response carries one -- see
    module docstring: GAP_SCHEMA has no such field today, so this is always
    None against real votes. Kept generic/forward-compatible per the
    brief's pinned tiebreak rule and exercised directly by unit tests."""
    value = vote.get("noise_confidence")
    return float(value) if isinstance(value, (int, float)) else None


def majority_vote(votes: list[dict]) -> dict:
    """Brief's pinned majority rule (full text in the module docstring).
    `votes` are the 3 raw ask_gemini_gaps() response dicts (GAP_SCHEMA
    shape) for one clip. Returns {"background_noise_present": bool,
    "background_noise_type": str} -- the majority verdict only; per-vote
    diagnostics (character, tokens, cost) live alongside this in the
    caller's per-clip log, not inside the verdict itself."""
    present_votes = [bool(v["background_noise_present"]) for v in votes]
    if sum(present_votes) < 2:  # 0 or 1 of 3 true -- all ties toward absent
        return {"background_noise_present": False, "background_noise_type": ""}

    true_votes = [v for v in votes if bool(v["background_noise_present"])]
    normalized = [_normalize_type(v.get("background_noise_type", "")) for v in true_votes]
    counts = Counter(normalized)
    top_count = max(counts.values())
    modal_types = [t for t, n in counts.items() if n == top_count]

    if len(modal_types) == 1:
        chosen = modal_types[0]
    else:
        confidences = [_vote_confidence(v) for v in true_votes]
        if any(c is not None for c in confidences):
            best_idx = max(
                range(len(true_votes)),
                key=lambda i: confidences[i] if confidences[i] is not None else -1.0,
            )
            chosen = normalized[best_idx]
        else:
            chosen = normalized[0]  # first true vote, original call order

    return {"background_noise_present": True, "background_noise_type": chosen}


def _sum_tokens(token_dicts: list[dict]) -> dict:
    return {
        "in": sum((t.get("in") or 0) for t in token_dicts),
        "out": sum((t.get("out") or 0) for t in token_dicts),
    }


def _vote_gap_listening(name: str) -> dict:
    """One clip's full majority-vote round: extract E1's gap audio
    (identity-imported concat_gaps) ONCE, then ask E1's identity-imported
    ask_gemini_gaps VOTES_PER_CLIP independent times against that SAME
    encoded blob -- only the LLM call repeats (that is the variance under
    test, per the brief: "call-to-call, not parameter-induced"); VAD/gap-
    extraction/encoding are deterministic and would add cost, not signal,
    if repeated. Mirrors exp1_gap_noise.run_once's <2s skip floor (same
    collaborators, same threshold) so a too-short clip costs nothing rather
    than raising."""
    audio = load_audio(DATA_DIR / name)
    audio_s = audio.samples.size / audio.sr
    vad = analyze_vad(audio.samples, audio.sr)
    gaps = concat_gaps(audio.samples, audio.sr, vad)
    gap_seconds = gaps.size / audio.sr

    if gaps.size < 2 * audio.sr:
        return {
            "skipped": True,
            "gap_seconds": gap_seconds,
            "audio_s": round(audio_s, 3),
            "votes": [],
            "majority": None,
            "tokens": {"in": 0, "out": 0},
            "cost_usd": 0.0,
        }

    blob = encode_opus_ogg(gaps, audio.sr)
    votes = []
    for _ in range(VOTES_PER_CLIP):
        data, cost, tokens = ask_gemini_gaps(blob)
        votes.append({"data": data, "tokens": tokens, "cost_usd": cost})

    return {
        "skipped": False,
        "gap_seconds": round(gap_seconds, 1),
        "audio_s": round(audio_s, 3),
        "votes": votes,
        "majority": majority_vote([v["data"] for v in votes]),
        "tokens": _sum_tokens([v["tokens"] for v in votes]),
        "cost_usd": sum(v["cost_usd"] for v in votes),
    }


def run_once(run_idx: int) -> dict:
    guard = SpendGuard()
    guard.check(EST_COST_PER_RUN)
    truth_anchors = load_truth()
    per_clip: dict = {}
    cost = 0.0
    for name in ANCHORS:
        result = _vote_gap_listening(name)
        cost += result["cost_usd"]
        truth = {
            "background_noise_present": truth_anchors[name]["background_noise_present"],
            "background_noise_type": truth_anchors[name]["background_noise_type"],
        }
        entry = {**result, "truth": truth}
        if not result["skipped"]:
            majority_present = bool(result["majority"]["background_noise_present"])
            entry["present_correct"] = majority_present == bool(truth["background_noise_present"])
        per_clip[name] = entry
    guard.add(cost)
    audio_minutes = round(sum(v["audio_s"] for v in per_clip.values()) / 60.0, 6)
    payload = {
        "exp": "exp6_gap_vote",
        "run": run_idx,
        "cost_usd": cost,
        "model": get_settings().gemini_model,
        "votes_per_clip": VOTES_PER_CLIP,
        "audio_minutes": audio_minutes,
        "pricing": {
            "in_per_1m": GEMINI_LITE_IN,
            "out_per_1m": GEMINI_LITE_OUT,
            "source": "shipping default rate (eval/experiments/common.py); the gap-listening "
            "vote calls use the shipping model at its default audio pricing, same convention "
            "as exp1_gap_noise.py / combined.py.",
        },
        "per_clip": per_clip,
    }
    log_run("exp6_gap_vote", run_idx, payload)
    return payload


def operating_point(runs: list[dict]) -> dict:
    """The brief's operating-point formula, computed from this module's own
    live run logs: baseline ($0.00146/audio-min, bake-off headline) + 3x the
    measured single-gap-vote marginal cost per audio-min. See module
    docstring for the full derivation and its relationship to the study
    doc's established E1-marginal footnote methodology."""
    if not runs:
        raise ValueError("no runs to compute an operating point from")
    audio_minutes = [r["audio_minutes"] for r in runs]
    if max(audio_minutes) - min(audio_minutes) > 1e-6:
        raise ValueError(
            f"runs measured different audio_minutes {audio_minutes} -- not directly comparable"
        )
    mean_run_cost = sum(r["cost_usd"] for r in runs) / len(runs)
    voting_marginal_per_min = mean_run_cost / audio_minutes[0]
    single_vote_marginal_per_min = voting_marginal_per_min / VOTES_PER_CLIP
    return {
        "mean_run_cost_usd": mean_run_cost,
        "audio_minutes": audio_minutes[0],
        "single_vote_marginal_per_min": single_vote_marginal_per_min,
        "voting_marginal_per_min": voting_marginal_per_min,
        "baseline_per_min": BASELINE_PER_MIN,
        "operating_point_per_min": BASELINE_PER_MIN + 3 * single_vote_marginal_per_min,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    runs = []
    for i in range(1, args.runs + 1):
        p = run_once(i)
        runs.append(p)
        hits = sum(1 for v in p["per_clip"].values() if v.get("present_correct"))
        n = sum(1 for v in p["per_clip"].values() if not v.get("skipped"))
        print(f"run {i}: presence {hits}/{n}, ${p['cost_usd']:.4f}")
    op = operating_point(runs)
    print(
        f"operating point: ${op['operating_point_per_min']:.6f}/audio-min "
        f"(baseline ${op['baseline_per_min']:.5f} + 3x ${op['single_vote_marginal_per_min']:.6f})"
    )


if __name__ == "__main__":
    main()
