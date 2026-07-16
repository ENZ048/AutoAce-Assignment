"""Tone bake-off: run arms A/B/C over labeled clips; report accuracy, macro F1,
measured cost per audio-minute, and latency. Output feeds the technical memo.

Controller amendment C: try all three arms (`--arms gemini dimensional transcript`).
The transcript arm needs OPENAI_API_KEY (.env) for its OpenAI text call -- if it's
absent, the arm is skipped entirely upfront (no clips attempted), printed +
recorded in the output file rather than silently omitted. If instead a clip
raises DURING an attempted arm (a transient API/model failure), that clip is
scored as a miss via `record_outcome`'s ERROR_SENTINEL, not excluded from the
arm's own accuracy/macro-F1 denominator -- a failed arm must never look better
than it is by shrinking its own sample size. Gemini's audio-token cost is billed
by the audio API; the transcript arm's cost is whisper (local compute, $0
metered) + OpenAI TEXT tokens -- reported using the real
prompt_tokens/completion_tokens transcript_llm.py now exposes in ToneResult.raw
(see that module's amendment-C note), priced at the transcript arm's gpt-5-mini
rate looked up 2026-07-16: $0.25/1M input, $2.00/1M output.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from autoace_audio.analyzers.tone.base import classify_tone
from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import load_audio
from autoace_audio.config import get_settings
from eval.metrics import macro_f1

# Live prices (2026-07-16): gemini-3.1-flash-lite $0.50/1M audio-in tok (32 tok/s), $1.50/1M out.
GEMINI_IN_PER_TOK = 0.50 / 1e6
GEMINI_OUT_PER_TOK = 1.50 / 1e6
# gpt-5-mini text pricing (2026-07-16, OpenAI API pricing page): $0.25/1M input, $2.00/1M output.
# Whisper transcription is local compute -- $0 metered, not included in this figure.
OPENAI_IN_PER_TOK = 0.25 / 1e6
OPENAI_OUT_PER_TOK = 2.00 / 1e6

# Never a real EmotionalTone value -- a failed classify_tone call is recorded as
# this sentinel prediction rather than excluded from y_true/y_pred, so a failed
# arm SCORES AS A MISS (both accuracy and macro F1 are penalized) instead of
# quietly shrinking its own denominator (review round 1, Important #1).
ERROR_SENTINEL = "error"


def record_outcome(
    y_true: list[str], y_pred: list[str], true_tone: str, pred_tone: str | None
) -> None:
    """Append one clip's scored outcome in place. `pred_tone=None` means the arm
    raised an exception on this clip -- recorded as ERROR_SENTINEL, which can
    never equal a real EmotionalTone value, so it always counts as a miss.
    Pulled out as a pure function (no I/O, no model calls) so this scoring rule
    is unit-testable without touching any of bakeoff's live model/API arms."""
    y_true.append(true_tone)
    y_pred.append(pred_tone if pred_tone is not None else ERROR_SENTINEL)


def main(data_dir: Path, labels_path: Path, arms: list[str], out_path: Path) -> None:
    labels = {}
    with open(labels_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("result_json", "").strip():
                labels[row["name"]] = json.loads(row["result_json"])
    notes: list[str] = []
    if "transcript" in arms and not get_settings().openai_api_key:
        notes.append(
            "transcript arm requested but OPENAI_API_KEY is not configured -- "
            "skipped entirely (no clips attempted)."
        )
        arms = [a for a in arms if a != "transcript"]
    rows, table = (
        [],
        [
            "| arm | tone acc | macro F1 | $ / audio-min | proc s / audio-min |",
            "|---|---|---|---|---|",
        ],
    )
    for arm in arms:
        y_true, y_pred, costs, times = [], [], [], []
        n_failed = 0
        for name, truth in labels.items():
            audio = load_audio(data_dir / name)
            vad = analyze_vad(audio.samples, audio.sr)
            t0 = time.monotonic()
            try:
                r = classify_tone(arm, audio.samples, audio.sr, vad, snr_db=None)
            except Exception as e:  # noqa: BLE001 — a failed arm scores as a miss
                print(f"{arm} failed on {name}: {e}")
                n_failed += 1
                record_outcome(y_true, y_pred, truth["emotional_tone"], None)
                rows.append(
                    {
                        "arm": arm,
                        "clip": name,
                        "true": truth["emotional_tone"],
                        "pred": ERROR_SENTINEL,
                        "elapsed_s": None,
                    }
                )
                continue
            dt = time.monotonic() - t0
            record_outcome(y_true, y_pred, truth["emotional_tone"], r.tone.value)
            times.append(dt / (audio.duration_s / 60.0))
            if arm == "gemini" and r.raw.get("prompt_tokens"):
                dollars = (
                    r.raw["prompt_tokens"] * GEMINI_IN_PER_TOK
                    + (r.raw.get("output_tokens") or 0) * GEMINI_OUT_PER_TOK
                )
                costs.append(dollars / (audio.duration_s / 60.0))
            elif arm == "transcript" and r.raw.get("prompt_tokens"):
                dollars = (
                    r.raw["prompt_tokens"] * OPENAI_IN_PER_TOK
                    + (r.raw.get("completion_tokens") or 0) * OPENAI_OUT_PER_TOK
                )
                costs.append(dollars / (audio.duration_s / 60.0))
            rows.append(
                {
                    "arm": arm,
                    "clip": name,
                    "true": truth["emotional_tone"],
                    "pred": r.tone.value,
                    "elapsed_s": round(dt, 2),
                }
            )
        if n_failed:
            notes.append(
                f"{arm}: {n_failed}/{len(labels)} clip(s) failed and were scored as a miss "
                f"(sentinel prediction {ERROR_SENTINEL!r}), not excluded from the denominator."
            )
        acc = sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)
        if arm == "transcript" and not costs:
            cost = "$0 whisper (local) + OpenAI text cost N/A (no usage on response)"
        elif costs:
            cost = f"${sum(costs) / len(costs):.5f}"
            if arm == "transcript":
                cost += " (whisper local $0 + OpenAI text, metered)"
        else:
            cost = "$0 (local)"
        f1 = macro_f1(y_true, y_pred)
        # Mean of (processing seconds / audio-minute) across successfully-timed
        # clips -- a real-time factor (how many seconds of compute per minute of
        # audio), NOT a per-clip wall-clock duration; failed clips have no timing
        # and are excluded from this average only (they still count in acc/F1).
        s_per_audio_min = sum(times) / len(times) if times else float("nan")
        table.append(f"| {arm} | {acc:.0%} | {f1:.3f} | {cost} | {s_per_audio_min:.1f}/min |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(table)
    if notes:
        text += "\n\n**Notes:**\n" + "\n".join(f"- {n}" for n in notes)
    text += "\n\n```json\n" + json.dumps(rows, indent=2) + "\n```\n"
    out_path.write_text(text)
    print("\n".join(table))
    for n in notes:
        print(f"NOTE: {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    ap.add_argument("--arms", nargs="+", default=["gemini", "dimensional", "transcript"])
    ap.add_argument("--out", type=Path, default=Path("out/bakeoff.md"))
    a = ap.parse_args()
    main(a.data, a.labels, a.arms, a.out)
