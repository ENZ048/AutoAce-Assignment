"""Combined 'best stack': every lever that nets a real win against the
shipping baseline (2-of-3-vs-baseline rule, netted against its own
regressions -- standing amendment: wins AND regressions get equal
prominence, so a single mechanical wins_field hit is not enough on its own)
runs together. This module REPORTS the stack decision in its log so the
study table can show exactly which levers the headline number contains.

CONTROLLER BINDING DETERMINATION (study-task-8-brief.md dispatch, from the
Study Task 2-7 ledger): only E1 (gap-listening noise confirmation) nets
positive. E2 is a clean null (0 wins). E3 and E4 each clear a single
mechanical wins_field hit on their own target field but are outnumbered by
regressions elsewhere (1 win vs 2 regressions, same shape for both -- E4
also runs 2.72x the shipping Lite arm's cost, over the $0.003/audio-min
ceiling). E5 has 0 wins and 1 regression. decide_stack() below is fully
data-driven (reads the on-disk run logs from Study Tasks 2-7 and applies
that same net-wins rule); verify_stack_matches_determination() is a safety
net that fails loudly if a fresh computation and the recorded determination
ever disagree, rather than silently shipping a different combined arm than
the one the study reports.

Two of the brief's original draft functions had bugs, fixed here (both
verified against the real run logs -- see study-task-8-report.md):
  - decide_stack()'s flash/fewshot/advocate flags used a bare
    `any(wins_field(...))` check, which is satisfied by E3's and E4's lone
    mechanical win and would have wrongly included both. Replaced by
    _lever_nets_positive(), which also counts regressions via the new
    common.loses_field and requires a strictly positive net.
  - the draft's _gap_noise_won() required BOTH noisy anchors correct in the
    same run (an `all()` over anchors E1 never clears, since it never
    confirms call_002's TV) -- deterministically False regardless of E1's
    real call_003 win. Replaced by _gap_noise_included(), which reshapes
    exp0/exp1's differently-shaped logs onto the standard
    {"correct": {field: bool}} shape and reuses the same generic
    wins_field/loses_field helpers, scored per noisy anchor (matches how
    _overlap_included and the flash/fewshot/advocate keys already score
    per clip)."""

import argparse

from autoace_audio.analyzers.noise import analyze_noise
from autoace_audio.analyzers.tone.base import classify_tone
from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import encode_opus_ogg, load_audio
from autoace_audio.config import get_settings
from eval.experiments import exp1_gap_noise, exp2_fewshot, exp3_advocate, exp4_flash

# Note: exp5_overlap is read via read_runs("exp5_overlap") (its one deterministic
# run, never re-run live -- see module docstring), not called directly, so it is
# deliberately NOT imported as a module here (ruff F401 would flag it unused; the
# brief's own draft imported it but never referenced it either).
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
    loses_field,
    read_runs,
    wins_field,
)

# ~3 anchors x (1 shipping tone call + 1 short gap-listening call); measured
# baseline ~$0.0046/run (Study Task 2) + E1 gap-listening on 3 short clips
# (a fraction of E1's own $0.0037/run for 12 clips, Study Task 3) -- expect
# ~$0.01/run. Generous 2-3x margin.
EST_COST_PER_RUN = 0.03

INCLUDED_LEVERS = ["E1"]
EXCLUSIONS = {
    "E2": "few-shot intensity: clean null, 0/6 wins vs baseline "
    "(predictions byte-identical to baseline on every clip/run).",
    "E3": "advocate pass: 1 win vs 2 regressions on its own target field (emotional_tone).",
    "E4": "flash tone source: same shape as E3 (1 win vs 2 regressions across "
    "tone/intensity/overlap); also measured 2.72x the shipping Lite arm's cost, "
    "over the $0.003/audio-min ceiling. Tone/intensity source of truth stays "
    "the shipping lite arm.",
    "E5": "diarization overlap: 0 wins, 1 regression (call_003 flipped a solid "
    "baseline correct -> wrong; call_001/002 diarize to a single speaker, making "
    "speaker_overlap_present=True structurally unreachable there).",
    "E5_bonus": "customer-only dimensional intensity bonus: not a planned stack "
    "component, and its call_001 'win' is confounded (diarization fell back to "
    "full audio there, not genuine customer-only audio) -- excluded from the "
    "stack; recorded as a promising-but-unproven lead for Task 9's doc.",
}

_LEVER_KEYS = {
    "gap_noise": "E1",
    "fewshot": "E2",
    "advocate": "E3",
    "flash": "E4",
    "deepgram_overlap": "E5",
}


def _lever_nets_positive(
    base_runs: list[dict], lever_runs: list[dict], fields: list[str], clips: list[str]
) -> bool:
    """Standing amendment: wins AND regressions get equal prominence, so a
    lever is included only if it beats baseline on NET -- a real win
    somewhere it counts, not offset by regressions elsewhere. A lone
    mechanical wins_field hit is NOT enough by itself: E3 and E4 both clear
    wins_field on their target field but net negative once their
    regressions (common.loses_field, the exact mirror threshold) are
    counted the same way, and are correctly excluded."""
    if not base_runs or not lever_runs:
        return False
    wins = sum(wins_field(base_runs, lever_runs, f, c) for f in fields for c in clips)
    losses = sum(loses_field(base_runs, lever_runs, f, c) for f in fields for c in clips)
    return wins > 0 and wins > losses


def _gap_noise_included() -> bool:
    """E1's field (background_noise_present) isn't in exp0's scored FIELDS,
    and exp1's log shape (present_correct, no "correct" dict) differs from
    the standard shape wins_field/loses_field expect -- reshape both sides
    onto the SAME truth before reusing the generic helpers. Scored on the
    anchors truth actually marks noisy (the anchors E1 targets), matching
    the study design's E1 scope."""
    truth = load_truth()
    field = "background_noise_present"
    noisy = [c for c in ANCHORS if truth.get(c, {}).get(field)]

    base_runs = []
    for r in read_runs("exp0_baseline"):
        per_clip = {}
        for clip, v in r.get("per_clip", {}).items():
            if clip not in truth:
                continue
            opinion = (v.get("pred") or {}).get("noise_opinion") or {}
            per_clip[clip] = {
                "correct": {field: bool(opinion.get("present")) == bool(truth[clip][field])}
            }
        base_runs.append({"per_clip": per_clip})

    lever_runs = []
    for r in read_runs("exp1_gap_noise"):
        per_clip = {}
        for clip, v in r.get("per_clip", {}).items():
            if v.get("skipped") or clip not in truth:
                continue
            per_clip[clip] = {"correct": {field: bool(v["present_correct"])}}
        lever_runs.append({"per_clip": per_clip})

    return _lever_nets_positive(base_runs, lever_runs, [field], noisy)


def _overlap_included() -> bool:
    """E5 already logs speaker_overlap_present in the standard
    {"correct": {field: bool}} shape (both exp0 and exp5), so the generic
    helpers apply directly -- single deterministic API run, no repeats."""
    return _lever_nets_positive(
        read_runs("exp0_baseline"), read_runs("exp5_overlap"), ["speaker_overlap_present"], ANCHORS
    )


def decide_stack() -> dict:
    """Data-driven: reads every lever's on-disk run logs and applies the
    2-of-3-wins-vs-baseline rule, netted against regressions."""
    base = read_runs("exp0_baseline")
    return {
        "flash": _lever_nets_positive(
            base,
            read_runs("exp4_flash"),
            ["emotional_tone", "emotional_intensity", "speaker_overlap_present"],
            ANCHORS,
        ),
        "fewshot": _lever_nets_positive(
            base, read_runs("exp2_fewshot"), ["emotional_intensity"], ANCHORS
        ),
        "advocate": _lever_nets_positive(
            base, read_runs("exp3_advocate"), ["emotional_tone"], ANCHORS
        ),
        # E1/E5 are judged on their own evidence (different eval bases / log shapes):
        "gap_noise": _gap_noise_included(),
        "deepgram_overlap": _overlap_included(),
    }


def _stack_to_levers(stack: dict) -> list[str]:
    return sorted(lever for key, lever in _LEVER_KEYS.items() if stack.get(key))


def verify_stack_matches_determination(stack: dict) -> None:
    """Safety net: decide_stack() is data-driven, but the controller has
    already applied the net-wins rule by hand and recorded a binding
    determination (study-task-8-brief.md dispatch). If a fresh computation
    ever disagrees -- stale/edited logs, a math regression -- fail loudly
    before spending live money on the wrong stack."""
    computed = _stack_to_levers(stack)
    if computed != INCLUDED_LEVERS:
        raise RuntimeError(
            f"decide_stack() computed included levers {computed} from the on-disk "
            f"run logs, but the controller's binding determination is "
            f"{INCLUDED_LEVERS} (study-task-8-brief.md). Investigate before "
            "running combined live."
        )


def _shipping_classify(name: str) -> tuple[dict, float, dict]:
    """Same code path as exp0_baseline.run_once's per-clip classification --
    the shipping arm, no lever applied. Returns the FULL raw Gemini response
    dict (not just the 3 scored fields) so callers can read the noise
    fields / rationale too."""
    audio = load_audio(DATA_DIR / name)
    vad = analyze_vad(audio.samples, audio.sr)
    noise = analyze_noise(audio.samples, audio.sr, vad)
    r = classify_tone("gemini", audio.samples, audio.sr, vad, noise.snr_db)
    data = dict(r.raw.get("response", {}))
    data.setdefault("emotional_tone", r.tone.value)
    data.setdefault("emotional_intensity", r.intensity.value)
    data.setdefault("speaker_overlap_present", bool(r.overlap_opinion))
    tokens = {"in": r.raw.get("prompt_tokens"), "out": r.raw.get("output_tokens")}
    cost = gemini_cost(tokens["in"], tokens["out"])
    return data, cost, tokens


def _gap_listen(name: str) -> dict:
    """E1's noise-confirmation lever applied to one anchor: concatenate its
    VAD gaps and ask Gemini the focused noise question, exactly like
    exp1_gap_noise.run_once's per-clip logic (same collaborators, reused not
    copied). Never raises; a too-short gap is recorded as skipped (same rule
    as E1) with a zero cost and no prediction to fall back on."""
    audio = load_audio(DATA_DIR / name)
    vad = analyze_vad(audio.samples, audio.sr)
    gaps = exp1_gap_noise.concat_gaps(audio.samples, audio.sr, vad)
    gap_seconds = gaps.size / audio.sr
    if gaps.size < 2 * audio.sr:
        return {
            "skipped": True,
            "gap_seconds": gap_seconds,
            "pred": None,
            "tokens": {"in": None, "out": None},
            "cost_usd": 0.0,
        }
    pred, cost, tokens = exp1_gap_noise.ask_gemini_gaps(encode_opus_ogg(gaps, audio.sr))
    return {
        "skipped": False,
        "gap_seconds": round(gap_seconds, 1),
        "pred": pred,
        "tokens": tokens,
        "cost_usd": cost,
    }


def _sum_tokens(*dicts: dict) -> dict:
    return {
        "in": sum((d.get("in") or 0) for d in dicts if d),
        "out": sum((d.get("out") or 0) for d in dicts if d),
    }


def run_once(run_idx: int, stack: dict) -> dict:
    truth = load_truth()
    guard = SpendGuard()
    guard.check(EST_COST_PER_RUN)
    per_clip, run_cost = {}, 0.0
    for name in ANCHORS:
        clip_cost = 0.0
        if stack["fewshot"]:
            tone_data, c = exp2_fewshot.classify_with_exemplars(name)
            tone_tokens = {"in": None, "out": None}  # exp2 predates the tokens amendment
        elif stack["flash"]:
            tone_data, c, tone_tokens = exp4_flash.classify_flash(name)
        else:
            tone_data, c, tone_tokens = _shipping_classify(name)
        clip_cost += c

        advocate_tokens = {"in": 0, "out": 0}
        if stack["advocate"]:
            tone_data, c2, advocate_tokens = exp3_advocate.advocate_pass(
                name,
                tone_data["emotional_tone"],
                tone_data["emotional_intensity"],
                str(tone_data.get("rationale", "")),
            )
            clip_cost += c2

        pred = {
            "emotional_tone": tone_data["emotional_tone"],
            "emotional_intensity": tone_data["emotional_intensity"],
            "speaker_overlap_present": bool(tone_data["speaker_overlap_present"]),
            "background_noise_present": bool(tone_data.get("background_noise_present", False)),
            "background_noise_type": str(tone_data.get("background_noise_type", "")),
        }

        if stack["deepgram_overlap"]:
            dg_runs = read_runs("exp5_overlap")
            if dg_runs:
                pred["speaker_overlap_present"] = bool(
                    dg_runs[0]["per_clip"][name]["pred"]["speaker_overlap_present"]
                )

        gap = None
        if stack["gap_noise"]:
            gap = _gap_listen(name)
            clip_cost += gap["cost_usd"]
            if not gap["skipped"] and gap["pred"] is not None:
                pred["background_noise_present"] = bool(gap["pred"]["background_noise_present"])
                pred["background_noise_type"] = str(gap["pred"].get("background_noise_type", ""))

        run_cost += clip_cost
        per_clip[name] = {
            "pred": pred,
            "correct": field_compare(
                pred,
                truth[name],
                [
                    "emotional_tone",
                    "emotional_intensity",
                    "speaker_overlap_present",
                    "background_noise_present",
                ],
            ),
            "background_noise_type_truth": truth[name].get("background_noise_type", ""),
            "tokens": _sum_tokens(tone_tokens, advocate_tokens, gap["tokens"] if gap else {}),
            "cost_usd": clip_cost,
            "gap_listening": gap,
        }
    guard.add(run_cost)
    payload = {
        "exp": "combined",
        "run": run_idx,
        "cost_usd": run_cost,
        "model": get_settings().gemini_model,
        "pricing": {
            "in_per_1m": GEMINI_LITE_IN,
            "out_per_1m": GEMINI_LITE_OUT,
            "source": "shipping default rate (eval/experiments/common.py); no override "
            "-- both the shipping tone call and E1's gap-listening call use the "
            "shipping model at its default audio pricing.",
        },
        "stack": stack,
        "included_levers": INCLUDED_LEVERS,
        "exclusions": EXCLUSIONS,
        "per_clip": per_clip,
    }
    log_run("combined", run_idx, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    stack = decide_stack()
    print("stack decision:", stack)
    verify_stack_matches_determination(stack)
    for i in range(1, args.runs + 1):
        p = run_once(i, stack)
        print(
            f"run {i}: ${p['cost_usd']:.4f}",
            {k: v["correct"] for k, v in p["per_clip"].items()},
        )


if __name__ == "__main__":
    main()
