"""E5: measured overlap from Deepgram diarization (user-authorized recipient,
2026-07-17) instead of Gemini's judgment. Overlap = any cross-speaker turn
intersection >= 0.5s that isn't a bare back-channel (<= 1.0s own duration AND
<= 2 words). Thresholds are first-pass choices from the client's own
definition ("brief back-channel does not count"), NOT tuned on the eval.
Bonus (free, local): dimensional arm re-scored on customer-only audio.

Pricing verification (2026-07-17, see PRICING_SOURCE): the brief's flat
$0.0043/min rate is the last-documented Nova-2 pay-as-you-go prerecorded
price, corroborated by several independent 2026 pricing summaries, but
Nova-2 itself is no longer on the LIVE https://deepgram.com/pricing table
(current tiers are Flux and Nova-3 only; the page's FAQ says "older models
(Nova-2, Enhanced, Base) are still available" without listing a rate for
them). That same live page separately lists a Speaker Diarization add-on of
$0.0020/min for pay-as-you-go prerecorded audio -- not obviously folded into
the brief's single flat figure, and diarize=true is this experiment's whole
premise, so omitting it would understate exactly the cost this study exists
to measure. Both components are named/sourced separately and summed for the
actual per-clip cost log; DG_RATE_PER_MIN alone still matches the brief."""

import argparse
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from autoace_audio.analyzers.tone.base import classify_tone
from autoace_audio.analyzers.tone.dimensional import MODEL_ID as DIMENSIONAL_MODEL_ID
from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import load_audio
from eval.experiments.common import (
    ANCHORS,
    DATA_DIR,
    SpendGuard,
    field_compare,
    load_truth,
    log_run,
)

DG_MODEL_ID = "nova-2"
DG_URL = (
    f"https://api.deepgram.com/v1/listen"
    f"?model={DG_MODEL_ID}&diarize=true&punctuate=false&smart_format=false"
)
PRICING_SOURCE = "https://deepgram.com/pricing (checked 2026-07-17)"
# Nova-2 pay-as-you-go prerecorded base rate -- brief's own figure; see
# module docstring for why it's kept despite Nova-2 no longer being on the
# live pricing table.
DG_RATE_PER_MIN = 0.0043
# NEW finding at verification time, not in the brief: the live page lists
# Speaker Diarization as a separate pay-as-you-go prerecorded add-on.
DG_DIARIZE_ADDON_PER_MIN = 0.0020
EST_COST = 0.03  # ~4 audio-minutes total; safe upper bound at either rate


def _dg_key() -> str:
    from dotenv import dotenv_values

    key = dotenv_values(".env").get("DEEPGRAM_API_KEY") or os.environ.get("DEEPGRAM_API_KEY", "")
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY missing from .env")
    return key


def deepgram_words(path: Path) -> list[dict]:
    req = urllib.request.Request(
        DG_URL,
        data=path.read_bytes(),
        headers={
            "Authorization": f"Token {_dg_key()}",
            "Content-Type": "application/octet-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read())
    words = body["results"]["channels"][0]["alternatives"][0]["words"]
    return [
        {"word": w["word"], "start": w["start"], "end": w["end"], "speaker": w.get("speaker", 0)}
        for w in words
    ]


def turns_from_words(words: list[dict], max_intra_gap_s: float = 0.5) -> list[dict]:
    turns: list[dict] = []
    for w in words:
        if (
            turns
            and turns[-1]["speaker"] == w["speaker"]
            and w["start"] - turns[-1]["end"] <= max_intra_gap_s
        ):
            turns[-1]["end"] = w["end"]
            turns[-1]["words"] += 1
        else:
            turns.append(
                {"speaker": w["speaker"], "start": w["start"], "end": w["end"], "words": 1}
            )
    return turns


def overlap_from_turns(
    turns: list[dict],
    min_overlap_s: float = 0.5,
    backchannel_max_s: float = 1.0,
    backchannel_max_words: int = 2,
) -> bool:
    for i, a in enumerate(turns):
        for b in turns[i + 1 :]:
            if b["start"] >= a["end"]:
                break
            if a["speaker"] == b["speaker"]:
                continue
            inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
            if inter < min_overlap_s:
                continue
            for t in (a, b):
                dur = t["end"] - t["start"]
                if dur <= backchannel_max_s and t["words"] <= backchannel_max_words:
                    break
            else:
                return True
    return False


def overlap_spans(
    turns: list[dict],
    min_overlap_s: float = 0.5,
    backchannel_max_s: float = 1.0,
    backchannel_max_words: int = 2,
) -> list[dict]:
    """Diagnostic-only twin of overlap_from_turns: same rule, but returns the
    qualifying cross-speaker windows instead of short-circuiting to a bool.
    Kept as an independent implementation (not a refactor of
    overlap_from_turns) so that brief-pinned, unit-tested function is never
    touched by this addition; cross-checked for agreement in the test
    suite."""
    spans: list[dict] = []
    for i, a in enumerate(turns):
        for b in turns[i + 1 :]:
            if b["start"] >= a["end"]:
                break
            if a["speaker"] == b["speaker"]:
                continue
            inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
            if inter < min_overlap_s:
                continue
            if any(
                (t["end"] - t["start"]) <= backchannel_max_s and t["words"] <= backchannel_max_words
                for t in (a, b)
            ):
                continue
            spans.append(
                {
                    "speakers": [a["speaker"], b["speaker"]],
                    "start": max(a["start"], b["start"]),
                    "end": min(a["end"], b["end"]),
                    "intersection_s": round(inter, 3),
                }
            )
    return spans


def customer_only_audio(
    samples: np.ndarray, sr: int, turns: list[dict]
) -> tuple[np.ndarray, int, str]:
    """Agent = speaker of the first turn (Erica opens every sample call).
    Returns (customer samples, customer speaker id, attribution note)."""
    if not turns:
        return samples, -1, "no turns; used full audio"
    agent = turns[0]["speaker"]
    speakers = {t["speaker"] for t in turns}
    if len(speakers) < 2:
        return samples, -1, "single speaker diarized; used full audio (ambiguous)"
    customer = next(s for s in sorted(speakers) if s != agent)
    parts = [
        samples[int(t["start"] * sr) : int(t["end"] * sr)]
        for t in turns
        if t["speaker"] == customer
    ]
    return np.concatenate(parts) if parts else samples, customer, "first-turn=agent rule"


def run_once(run_idx: int = 1) -> dict:
    guard = SpendGuard()
    guard.check(EST_COST)
    truth = load_truth()
    per_clip: dict = {}
    effective_rate = DG_RATE_PER_MIN + DG_DIARIZE_ADDON_PER_MIN
    for name in ANCHORS:
        path = DATA_DIR / name
        audio = load_audio(path)
        duration_s = audio.samples.size / audio.sr
        words = deepgram_words(path)
        turns = turns_from_words(words)
        overlap = overlap_from_turns(turns)
        cust, cust_id, note = customer_only_audio(audio.samples, audio.sr, turns)
        vad = analyze_vad(cust, audio.sr)
        dim = classify_tone("dimensional", cust, audio.sr, vad, None)
        pred = {"speaker_overlap_present": overlap}
        clip_cost = (duration_s / 60.0) * effective_rate
        per_clip[name] = {
            "pred": pred,
            "correct": field_compare(pred, truth[name], ["speaker_overlap_present"]),
            "n_words": len(words),
            "n_turns": len(turns),
            "attribution": note,
            "customer_speaker_id": cust_id,
            "overlap_spans": overlap_spans(turns),
            "audio_s": round(duration_s, 3),
            "cost_usd": clip_cost,
            "dimensional_customer_only": {
                "tone": dim.tone.value,
                "intensity": dim.intensity.value,
                "valence": dim.raw.get("valence"),
                "arousal": dim.raw.get("arousal"),
            },
        }
    total_cost = sum(c["cost_usd"] for c in per_clip.values())
    guard.add(total_cost)
    payload = {
        "exp": "exp5_overlap",
        "run": run_idx,
        "cost_usd": total_cost,
        "audio_minutes": round(sum(c["audio_s"] for c in per_clip.values()) / 60.0, 6),
        "rate_per_min": DG_RATE_PER_MIN,
        "diarize_addon_per_min": DG_DIARIZE_ADDON_PER_MIN,
        "pricing_source": PRICING_SOURCE,
        "model": DG_MODEL_ID,  # standing amendment: Deepgram model id used
        "dimensional_model": DIMENSIONAL_MODEL_ID,  # standing amendment: bonus arm's model id
        "per_clip": per_clip,
    }
    log_run("exp5_overlap", run_idx, payload)
    return payload


def main() -> None:
    argparse.ArgumentParser().parse_args()
    p = run_once(1)
    print(f"${p['cost_usd']:.4f}", {k: v["pred"] for k, v in p["per_clip"].items()})


if __name__ == "__main__":
    main()
