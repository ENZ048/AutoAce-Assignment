# Budget–Accuracy Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, with live repeated runs and logged costs, how much accuracy each of five spending levers buys over the shipping backend, and package the results as a client-presentable study + private pitch page.

**Architecture:** A self-contained `eval/experiments/` package. `common.py` owns run logging, cost accounting, and a file-backed $10 spend guard. Each experiment module (`exp0`–`exp5`, `combined.py`) imports shipping code (`classify_tone`, analyzers, `encode_opus_ogg`) and passes overrides as arguments — shipping config and fusion are never mutated. Results land as JSON run logs under `out/experiments/` (untracked); the final task synthesizes them into `docs/experiments/2026-07-17-budget-accuracy-study.md` (committed) and `Test/BUDGET-PITCH.md` (outside repo).

**Tech Stack:** Python 3.12, existing autoace_audio package, google-genai (already installed), Deepgram prerecorded REST API via stdlib `urllib.request` (no new dependency), pytest fast tests for all pure logic.

## Global Constraints

- Hard spend cap **$10.00** total across all experiments; guard warns at $7.00 and refuses runs projecting past $10.00. All costs computed from real usage metadata / posted per-minute rates, never estimated after the fact.
- Audio may be sent ONLY to Google's Gemini paid tier and Deepgram (user-authorized 2026-07-17). Nothing else. Transcripts-only rule for OpenAI is irrelevant here (unused).
- `data/`, `out/`, `.env` never staged. Study doc quotes NO verbatim customer speech (paraphrase, e.g. "a single Spanish profanity phrase").
- Shipping behavior untouched: no edits to `config.py` defaults, `fusion.py`, `pipeline.py`, or any analyzer. Experiments pass parameters.
- No experiment module imports models or reads `.env` at module scope; heavy work happens inside functions.
- Every accuracy claim in deliverables carries the n=3-anchors caveat inline. Improvement rule: a lever "wins" a field only if correct in ≥2 of 3 runs where baseline was wrong in ≥2 of 3.
- 16 GB machine: strictly sequential; never run two experiments concurrently.
- `make test` stays green (experiment unit tests are fast/unmarked); `ruff format && ruff check src tests eval` clean before every commit; conventional commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Anchor ground truth (from `data/labels.csv`): call_001 tone=upset intensity=high overlap=false; call_002 tone=neutral intensity=medium overlap=true noise=TV/medium; call_003 tone=satisfied intensity=medium overlap=true noise="sharp static"/medium.
- Baseline measured misses (context for every experiment): tone wrong on call_002 (frustrated); intensity low-by-one on call_001+call_003; overlap wrong on call_002 (false); noise type "radio" on both noisy calls.

---

### Task 1: Experiments common harness

**Files:**
- Create: `eval/experiments/__init__.py` (empty), `eval/experiments/common.py`
- Test: `tests/unit/test_experiments_common.py`

**Interfaces:**
- Consumes: `autoace_audio.audio_io.load_audio`, `autoace_audio.analyzers.vad.analyze_vad`, `data/labels.csv`.
- Produces (used by every later task):
  - `ANCHORS: list[str]`, `DATA_DIR: Path`, `OUT_DIR: Path` (= `Path("out/experiments")`)
  - `load_truth() -> dict[str, dict]` — labels.csv parsed
  - `log_run(exp: str, run_idx: int, payload: dict) -> Path` — writes `out/experiments/{exp}_run{run_idx}.json` (creates dirs; `payload` must include `"cost_usd"`)
  - `read_runs(exp: str) -> list[dict]` — all run logs for an experiment, sorted by run index
  - `SpendGuard` with `add(cost_usd: float)`, `total() -> float`, `check(projected_usd: float)` raising `BudgetExceeded` past $10.00 and printing a warning past $7.00; state file `out/experiments/spend.json`
  - `gemini_cost(prompt_tokens: int | None, output_tokens: int | None, in_rate: float = 0.50, out_rate: float = 1.50) -> float` — dollars from tokens at $/1M rates; `None` tokens count as 0
  - `field_compare(pred: dict, truth: dict, fields: list[str]) -> dict[str, bool]`
  - `wins_field(baseline_runs: list[dict], lever_runs: list[dict], field: str, clip: str) -> bool` — the ≥2-of-3 rule: lever correct ≥2/3 where baseline correct ≤1/3 (each run dict holds `{"per_clip": {clip: {"pred": {...}, "correct": {field: bool}}}}`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_experiments_common.py
import json

import pytest

from eval.experiments.common import (
    BudgetExceeded,
    SpendGuard,
    field_compare,
    gemini_cost,
    wins_field,
)


def test_gemini_cost_math():
    # 1M in + 1M out at default rates = $2.00; None counts as zero
    assert gemini_cost(1_000_000, 1_000_000) == pytest.approx(2.00)
    assert gemini_cost(1509, 102) == pytest.approx(1509 * 0.50 / 1e6 + 102 * 1.50 / 1e6)
    assert gemini_cost(None, None) == 0.0


def test_spend_guard_caps_at_10_and_persists(tmp_path):
    state = tmp_path / "spend.json"
    g = SpendGuard(state_path=state, cap_usd=10.0, warn_usd=7.0)
    g.add(6.0)
    g.check(1.0)  # 7.0 projected: fine (warning only past warn threshold)
    g.add(3.5)
    with pytest.raises(BudgetExceeded):
        g.check(1.0)  # 9.5 + 1.0 > 10.0
    # persisted: a fresh instance sees the same total
    g2 = SpendGuard(state_path=state, cap_usd=10.0, warn_usd=7.0)
    assert g2.total() == pytest.approx(9.5)
    assert json.loads(state.read_text())["total_usd"] == pytest.approx(9.5)


def test_field_compare():
    pred = {"a": 1, "b": "x", "c": True}
    truth = {"a": 1, "b": "y", "c": True}
    assert field_compare(pred, truth, ["a", "b", "c"]) == {"a": True, "b": False, "c": True}


def _runs(correct_flags: list[bool], clip="call_002.ogg", field="emotional_tone"):
    return [
        {"per_clip": {clip: {"pred": {}, "correct": {field: flag}}}}
        for flag in correct_flags
    ]


def test_wins_field_requires_2of3_flip():
    base = _runs([False, False, False])
    good = _runs([True, True, False])
    bad = _runs([True, False, False])
    assert wins_field(base, good, "emotional_tone", "call_002.ogg") is True
    assert wins_field(base, bad, "emotional_tone", "call_002.ogg") is False
    # baseline already mostly right -> no win even if lever is right
    base_ok = _runs([True, True, False])
    assert wins_field(base_ok, good, "emotional_tone", "call_002.ogg") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_experiments_common.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.experiments'`

- [ ] **Step 3: Write `eval/experiments/common.py`** (and empty `__init__.py`)

```python
"""Shared harness for the budget-accuracy study (docs/superpowers/specs/
2026-07-17-budget-accuracy-study-design.md). Run logging, cost accounting,
the $10 spend guard, and the 2-of-3 improvement rule. Experiments import
shipping code and pass overrides as arguments -- nothing here mutates
shipping config or fusion."""

import csv
import json
from pathlib import Path

ANCHORS = ["call_001.ogg", "call_002.ogg", "call_003.ogg"]
DATA_DIR = Path("data")
OUT_DIR = Path("out/experiments")

# Live Gemini flash-lite audio rates, $ per 1M tokens (verified 2026-07-16).
GEMINI_LITE_IN = 0.50
GEMINI_LITE_OUT = 1.50


class BudgetExceeded(RuntimeError):
    pass


class SpendGuard:
    """File-backed cumulative spend tracker. check() BEFORE a run with the
    projected cost; add() AFTER with the measured cost."""

    def __init__(self, state_path: Path = OUT_DIR / "spend.json",
                 cap_usd: float = 10.0, warn_usd: float = 7.0) -> None:
        self.state_path = state_path
        self.cap_usd = cap_usd
        self.warn_usd = warn_usd

    def total(self) -> float:
        if self.state_path.exists():
            return float(json.loads(self.state_path.read_text())["total_usd"])
        return 0.0

    def add(self, cost_usd: float) -> None:
        total = self.total() + float(cost_usd)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({"total_usd": total}))

    def check(self, projected_usd: float) -> None:
        projected_total = self.total() + float(projected_usd)
        if projected_total > self.cap_usd:
            raise BudgetExceeded(
                f"projected total ${projected_total:.2f} exceeds cap ${self.cap_usd:.2f}"
            )
        if projected_total > self.warn_usd:
            print(f"WARNING: projected study spend ${projected_total:.2f} "
                  f"(cap ${self.cap_usd:.2f})")


def gemini_cost(prompt_tokens: int | None, output_tokens: int | None,
                in_rate: float = GEMINI_LITE_IN, out_rate: float = GEMINI_LITE_OUT) -> float:
    return ((prompt_tokens or 0) * in_rate + (output_tokens or 0) * out_rate) / 1e6


def load_truth(labels_path: Path = DATA_DIR / "labels.csv") -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(labels_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("name") and row.get("result_json", "").strip():
                out[row["name"].strip()] = json.loads(row["result_json"])
    return out


def log_run(exp: str, run_idx: int, payload: dict) -> Path:
    assert "cost_usd" in payload, "every run log must carry its measured cost"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{exp}_run{run_idx}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def read_runs(exp: str) -> list[dict]:
    return [json.loads(p.read_text())
            for p in sorted(OUT_DIR.glob(f"{exp}_run*.json"))]


def field_compare(pred: dict, truth: dict, fields: list[str]) -> dict[str, bool]:
    return {f: pred.get(f) == truth.get(f) for f in fields}


def _correct_count(runs: list[dict], field: str, clip: str) -> tuple[int, int]:
    flags = [r["per_clip"][clip]["correct"][field]
             for r in runs if clip in r.get("per_clip", {})
             and field in r["per_clip"][clip].get("correct", {})]
    return sum(flags), len(flags)


def wins_field(baseline_runs: list[dict], lever_runs: list[dict],
               field: str, clip: str) -> bool:
    """Spec 2-of-3 rule: lever right >=2/3 where baseline right <=1/3."""
    base_ok, base_n = _correct_count(baseline_runs, field, clip)
    lever_ok, lever_n = _correct_count(lever_runs, field, clip)
    if base_n == 0 or lever_n == 0:
        return False
    return base_ok <= base_n // 3 and lever_ok >= (2 * lever_n + 2) // 3
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_experiments_common.py -q`
Expected: `5 passed`

- [ ] **Step 5: Run full fast suite, ruff, commit**

Run: `make test && .venv/bin/ruff format src tests eval && .venv/bin/ruff check src tests eval`
Expected: all green.

```bash
git add eval/experiments tests/unit/test_experiments_common.py
git commit -m "feat(experiments): common harness — run logs, cost accounting, \$10 spend guard"
```

---

### Task 2: Baseline (exp0) — shipping config, 3 runs

**Files:**
- Create: `eval/experiments/exp0_baseline.py`

**Interfaces:**
- Consumes: `classify_tone` (shipping gemini arm), `analyze_vad`, `analyze_noise`, `load_audio`, common harness.
- Produces: run logs `exp0_baseline_run{1..3}.json`, each `{"exp": "exp0_baseline", "run": N, "cost_usd": float, "per_clip": {name: {"pred": {tone, intensity, overlap, noise_present, noise_type}, "correct": {...}, "tokens": {...}}}}`. Later tasks call `read_runs("exp0_baseline")`.

- [ ] **Step 1: Write `eval/experiments/exp0_baseline.py`**

```python
"""exp0: shipping-config baseline, run 3x over the anchors. Every later
lever's delta is measured against THIS distribution, not a single run."""

import argparse

from autoace_audio.analyzers.noise import analyze_noise
from autoace_audio.analyzers.tone.base import classify_tone
from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import load_audio
from eval.experiments.common import (
    ANCHORS, DATA_DIR, SpendGuard, field_compare, gemini_cost, load_truth, log_run,
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
        print(f"run {i}: ${p['cost_usd']:.4f}",
              {k: v["correct"] for k, v in p["per_clip"].items()})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it live**

Run: `.venv/bin/python -m eval.experiments.exp0_baseline --runs 3`
Expected: 3 lines, each ~$0.005–0.01; call_001 tone correct / intensity varies; call_002 tone typically wrong (the known miss). Three JSON files in `out/experiments/`.

- [ ] **Step 3: Fast suite, ruff, commit**

Run: `make test && .venv/bin/ruff format src tests eval && .venv/bin/ruff check src tests eval`

```bash
git add eval/experiments/exp0_baseline.py
git commit -m "feat(experiments): exp0 baseline runner (shipping config x3)"
```

---

### Task 3: E1 — gap-listening noise question

**Files:**
- Create: `eval/experiments/exp1_gap_noise.py`
- Test: `tests/unit/test_exp1_gaps.py`

**Interfaces:**
- Consumes: `analyze_vad`, `load_audio`, `encode_opus_ogg`, `google.genai` (same client pattern as `gemini_tone.py`), `data/validation/validation_manifest.csv` (rows with kind == "noise_aug" carry truth JSON), common harness.
- Produces: `concat_gaps(samples: np.ndarray, sr: int, vad: VadMap, min_gap_s: float = 1.0, cap_s: float = 60.0) -> np.ndarray` (pure, unit-tested); run logs `exp1_gap_noise_run{1..3}.json` with per-clip `{"pred": {present, type, character}, "truth": {...}, "gap_seconds": float, "skipped": bool}`.

- [ ] **Step 1: Write the failing test for the pure gap logic**

```python
# tests/unit/test_exp1_gaps.py
import numpy as np

from autoace_audio.analyzers.vad import Segment, VadMap
from eval.experiments.exp1_gap_noise import concat_gaps

SR = 16000


def _vad(gaps, total_s):
    return VadMap(speech=[], gaps=[Segment(a, b) for a, b in gaps],
                  speech_ratio=0.5, max_gap_s=max((b - a for a, b in gaps), default=0.0),
                  long_silence_present=False, total_s=total_s)


def test_concat_gaps_keeps_only_long_gaps_and_caps():
    total = 30.0
    samples = np.arange(int(total * SR), dtype=np.float32)
    vad = _vad([(0.0, 0.5), (2.0, 4.0), (10.0, 12.5)], total)
    out = concat_gaps(samples, SR, vad, min_gap_s=1.0, cap_s=60.0)
    # 0.5s gap dropped; 2.0s + 2.5s kept = 4.5s
    assert out.size == int(4.5 * SR)
    # content really comes from the gap regions (first kept sample = t=2.0s)
    assert out[0] == samples[int(2.0 * SR)]


def test_concat_gaps_caps_total_seconds():
    total = 200.0
    samples = np.zeros(int(total * SR), dtype=np.float32)
    vad = _vad([(0.0, 50.0), (60.0, 130.0)], total)
    out = concat_gaps(samples, SR, vad, cap_s=60.0)
    assert out.size == int(60.0 * SR)


def test_concat_gaps_empty_when_no_qualifying_gap():
    samples = np.zeros(SR, dtype=np.float32)
    vad = _vad([(0.0, 0.4)], 1.0)
    assert concat_gaps(samples, SR, vad).size == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/unit/test_exp1_gaps.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Write `eval/experiments/exp1_gap_noise.py`**

```python
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
    ANCHORS, DATA_DIR, SpendGuard, gemini_cost, load_truth, log_run,
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


def concat_gaps(samples: np.ndarray, sr: int, vad: VadMap,
                min_gap_s: float = 1.0, cap_s: float = 60.0) -> np.ndarray:
    parts, kept = [], 0.0
    for g in vad.gaps:
        dur = g.end - g.start
        if dur < min_gap_s:
            continue
        take = min(dur, cap_s - kept)
        if take <= 0:
            break
        parts.append(samples[int(g.start * sr): int((g.start + take) * sr)])
        kept += take
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def _noise_clips() -> dict[str, dict]:
    """9 synthetic noise clips + their truth, from the validation manifest."""
    import csv

    out = {}
    with open(VALIDATION_DIR / "validation_manifest.csv", newline="",
              encoding="utf-8-sig") as f:
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
    cost = gemini_cost(getattr(usage, "prompt_token_count", None),
                       getattr(usage, "candidates_token_count", None))
    return data, cost


def run_once(run_idx: int) -> dict:
    guard = SpendGuard()
    guard.check(EST_COST_PER_RUN)
    truth_anchors = load_truth()
    targets: list[tuple[Path, dict]] = [
        (DATA_DIR / n, {
            "background_noise_present": truth_anchors[n]["background_noise_present"],
            "background_noise_type": truth_anchors[n]["background_noise_type"],
        }) for n in ANCHORS
    ] + [(VALIDATION_DIR / n, t) for n, t in sorted(_noise_clips().items())]
    per_clip, cost = {}, 0.0
    for path, truth in targets:
        audio = load_audio(path)
        vad = analyze_vad(audio.samples, audio.sr)
        gaps = concat_gaps(audio.samples, audio.sr, vad)
        if gaps.size < 2 * audio.sr:  # <2s of gap audio: not applicable
            per_clip[path.name] = {"skipped": True, "gap_seconds": gaps.size / audio.sr,
                                   "truth": truth}
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
    payload = {"exp": "exp1_gap_noise", "run": run_idx, "cost_usd": cost,
               "per_clip": per_clip}
    log_run("exp1_gap_noise", run_idx, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    for i in range(1, args.runs + 1):
        p = run_once(i)
        hits = sum(1 for v in p["per_clip"].values()
                   if not v.get("skipped") and v["present_correct"])
        n = sum(1 for v in p["per_clip"].values() if not v.get("skipped"))
        print(f"run {i}: presence {hits}/{n}, ${p['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests, then live**

Run: `.venv/bin/pytest tests/unit/test_exp1_gaps.py -q` → `3 passed`
Run: `.venv/bin/python -m eval.experiments.exp1_gap_noise --runs 3`
Expected: 3 runs; call_001 skipped or predicted absent; the interesting rows are call_002/003 + the 9 synthetic clips. Cost ≈ $0.01–0.03 per run.

- [ ] **Step 5: Fast suite, ruff, commit**

```bash
git add eval/experiments/exp1_gap_noise.py tests/unit/test_exp1_gaps.py
git commit -m "feat(experiments): E1 gap-listening noise question"
```

---

### Task 4: E2 — audio few-shot intensity (leave-one-out)

**Files:**
- Create: `eval/experiments/exp2_fewshot.py`
- Test: `tests/unit/test_exp2_excerpt.py`

**Interfaces:**
- Consumes: `build_prompt` from `gemini_tone.py` (shipping prompt text reused verbatim), `encode_opus_ogg`, `analyze_vad`, common harness.
- Produces: `best_window(samples: np.ndarray, sr: int, vad: VadMap, win_s: float = 20.0) -> tuple[float, float]` (pure: the win_s window with max VAD speech-seconds, ties → earliest; if clip shorter than win_s returns (0, duration)); run logs `exp2_fewshot_run{1..3}.json`.

- [ ] **Step 1: Write the failing test for excerpt selection**

```python
# tests/unit/test_exp2_excerpt.py
import numpy as np

from autoace_audio.analyzers.vad import Segment, VadMap
from eval.experiments.exp2_fewshot import best_window

SR = 16000


def _vad(speech, total_s):
    return VadMap(speech=[Segment(a, b) for a, b in speech], gaps=[],
                  speech_ratio=0.5, max_gap_s=0.0,
                  long_silence_present=False, total_s=total_s)


def test_best_window_finds_densest_speech():
    # speech: thin at start, dense 40-60s
    vad = _vad([(2.0, 4.0), (40.0, 58.0)], 80.0)
    start, end = best_window(np.zeros(80 * SR, np.float32), SR, vad, win_s=20.0)
    assert 38.0 <= start <= 40.0 and end - start == 20.0


def test_best_window_ties_break_earliest():
    vad = _vad([(0.0, 10.0), (30.0, 40.0)], 60.0)
    start, _ = best_window(np.zeros(60 * SR, np.float32), SR, vad, win_s=20.0)
    assert start == 0.0


def test_best_window_short_clip_returns_whole():
    vad = _vad([(0.0, 5.0)], 12.0)
    assert best_window(np.zeros(12 * SR, np.float32), SR, vad, win_s=20.0) == (0.0, 12.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/unit/test_exp2_excerpt.py -q` → FAIL (module not found)

- [ ] **Step 3: Write `eval/experiments/exp2_fewshot.py`**

```python
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
    ANCHORS, DATA_DIR, SpendGuard, field_compare, gemini_cost, load_truth, log_run,
)

EST_COST_PER_RUN = 0.03  # 3 targets, each with ~40s exemplar audio on top
FIELDS = ["emotional_tone", "emotional_intensity"]


def best_window(samples: np.ndarray, sr: int, vad: VadMap,
                win_s: float = 20.0) -> tuple[float, float]:
    total = samples.size / sr
    if total <= win_s:
        return 0.0, total
    best_start, best_speech = 0.0, -1.0
    step = 1.0
    t = 0.0
    while t + win_s <= total:
        speech = sum(max(0.0, min(seg.end, t + win_s) - max(seg.start, t))
                     for seg in vad.speech)
        if speech > best_speech:  # strict > keeps earliest on ties
            best_start, best_speech = t, speech
        t += step
    return best_start, best_start + win_s


def _exemplar(name: str) -> tuple[bytes, str]:
    audio = load_audio(DATA_DIR / name)
    vad = analyze_vad(audio.samples, audio.sr)
    s0, s1 = best_window(audio.samples, audio.sr, vad)
    blob = encode_opus_ogg(audio.samples[int(s0 * audio.sr): int(s1 * audio.sr)], audio.sr)
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
        parts.append(f"The excerpt above is EXAMPLE {chr(65 + i)}: a different call whose "
                     f"correct emotional_intensity is '{intensity}'. It is calibration "
                     f"only -- do not classify it.")
    parts.append(types.Part.from_bytes(
        data=encode_opus_ogg(audio.samples, audio.sr), mime_type="audio/ogg"))
    parts.append("Now classify ONLY this final full call.\n\n"
                 + build_prompt(audio.samples.size / audio.sr, None, vad.speech_ratio))
    resp = client.models.generate_content(
        model=s.gemini_model, contents=parts,
        config=types.GenerateContentConfig(
            temperature=s.gemini_temperature,
            response_mime_type="application/json",
            response_schema=GEMINI_RESPONSE_SCHEMA,
        ),
    )
    data = json.loads(resp.text)
    usage = getattr(resp, "usage_metadata", None)
    return data, gemini_cost(getattr(usage, "prompt_token_count", None),
                             getattr(usage, "candidates_token_count", None))


def run_once(run_idx: int) -> dict:
    guard = SpendGuard()
    guard.check(EST_COST_PER_RUN)
    truth = load_truth()
    per_clip, cost = {}, 0.0
    for name in ANCHORS:
        data, c = classify_with_exemplars(name)
        cost += c
        pred = {"emotional_tone": data["emotional_tone"],
                "emotional_intensity": data["emotional_intensity"]}
        per_clip[name] = {"pred": pred,
                          "correct": field_compare(pred, truth[name], FIELDS),
                          "cost_usd": c}
    guard.add(cost)
    payload = {"exp": "exp2_fewshot", "run": run_idx, "cost_usd": cost,
               "per_clip": per_clip}
    log_run("exp2_fewshot", run_idx, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    for i in range(1, args.runs + 1):
        p = run_once(i)
        print(f"run {i}: ${p['cost_usd']:.4f}",
              {k: v["pred"]["emotional_intensity"] for k, v in p["per_clip"].items()})


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Unit tests, then live**

Run: `.venv/bin/pytest tests/unit/test_exp2_excerpt.py -q` → `3 passed`
Run: `.venv/bin/python -m eval.experiments.exp2_fewshot --runs 3`
Expected: intensity predictions per call per run; cost ≈ $0.02–0.04/run (exemplar audio roughly doubles input tokens).

- [ ] **Step 5: Fast suite, ruff, commit**

```bash
git add eval/experiments/exp2_fewshot.py tests/unit/test_exp2_excerpt.py
git commit -m "feat(experiments): E2 leave-one-out audio few-shot intensity"
```

---

### Task 5: E3 — devil's-advocate tone pass

**Files:**
- Create: `eval/experiments/exp3_advocate.py`

**Interfaces:**
- Consumes: shipping `classify_tone` (first pass), `build_prompt`, `GEMINI_RESPONSE_SCHEMA`, common harness.
- Produces: run logs `exp3_advocate_run{1..3}.json` with per-clip `{"first": {...}, "final": {...}, "flipped": bool, "correct": {...}}` — the report needs both the win count AND the regression count (correct→wrong flips).

- [ ] **Step 1: Write `eval/experiments/exp3_advocate.py`**

```python
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
    ANCHORS, DATA_DIR, SpendGuard, field_compare, gemini_cost, load_truth, log_run,
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


def advocate_pass(name: str, first_tone: str, first_intensity: str,
                  rationale: str) -> tuple[dict, float]:
    from google import genai
    from google.genai import types

    s = get_settings()
    client = genai.Client(api_key=s.gemini_api_key)
    audio = load_audio(DATA_DIR / name)
    blob = encode_opus_ogg(audio.samples, audio.sr)
    prompt = ADVOCATE_PROMPT.format(tone=first_tone, intensity=first_intensity,
                                    rationale=rationale or "(none recorded)")
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
    return data, gemini_cost(getattr(usage, "prompt_token_count", None),
                             getattr(usage, "candidates_token_count", None))


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
        cost += gemini_cost(first.raw.get("prompt_tokens"), first.raw.get("output_tokens"))
        rationale = str(first.raw.get("response", {}).get("rationale", ""))
        final, c2 = advocate_pass(name, first.tone.value, first.intensity.value, rationale)
        cost += c2
        pred = {"emotional_tone": final["emotional_tone"],
                "emotional_intensity": final["emotional_intensity"]}
        first_pred = {"emotional_tone": first.tone.value,
                      "emotional_intensity": first.intensity.value}
        per_clip[name] = {
            "first": first_pred,
            "final": pred,
            "flipped": pred["emotional_tone"] != first_pred["emotional_tone"],
            "correct": field_compare(pred, truth[name], FIELDS),
            "first_correct": field_compare(first_pred, truth[name], FIELDS),
        }
    guard.add(cost)
    payload = {"exp": "exp3_advocate", "run": run_idx, "cost_usd": cost,
               "per_clip": per_clip}
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
```

- [ ] **Step 2: Run live**

Run: `.venv/bin/python -m eval.experiments.exp3_advocate --runs 3`
Expected: per-run flip map; watch call_002 (does the advocate break the profanity fixation?) and calls 001/003 (regression risk). Cost ≈ $0.02–0.04/run.

- [ ] **Step 3: Fast suite, ruff, commit**

```bash
git add eval/experiments/exp3_advocate.py
git commit -m "feat(experiments): E3 devil's-advocate second tone pass"
```

---

### Task 6: E4 — full Gemini Flash tone arm

**Files:**
- Create: `eval/experiments/exp4_flash.py`

**Interfaces:**
- Consumes: `build_prompt`, `GEMINI_RESPONSE_SCHEMA`, `encode_opus_ogg`, common harness.
- Produces: run logs `exp4_flash_run{1..3}.json`; each log MUST record `{"model_id": str, "pricing": {"in_per_1m": float, "out_per_1m": float, "source": "<url or 'assumed, flag in study'>"}}`.

- [ ] **Step 1: Write `eval/experiments/exp4_flash.py`**

```python
"""E4: identical shipping prompt/schema, bigger model (gemini-3.1-flash).
The run log records the exact model id + the audio-token pricing used for
its cost column, with the price source, so the study table is auditable."""

import argparse
import json

from autoace_audio.analyzers.noise import analyze_noise
from autoace_audio.analyzers.tone.gemini_tone import GEMINI_RESPONSE_SCHEMA, build_prompt
from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import encode_opus_ogg, load_audio
from autoace_audio.config import get_settings
from eval.experiments.common import (
    ANCHORS, DATA_DIR, SpendGuard, field_compare, gemini_cost, load_truth, log_run,
)

FLASH_MODEL = "gemini-3.1-flash"
# VERIFY at implementation time from ai.google.dev/pricing and update BOTH
# numbers AND the source string; if the page is unreachable, keep these and
# set source to "assumed 2x lite -- flag in study doc".
FLASH_IN_PER_1M = 1.00
FLASH_OUT_PER_1M = 3.00
PRICING_SOURCE = "https://ai.google.dev/pricing (checked at run time)"

EST_COST_PER_RUN = 0.03
FIELDS = ["emotional_tone", "emotional_intensity", "speaker_overlap_present"]


def classify_flash(name: str) -> tuple[dict, float]:
    from google import genai
    from google.genai import types

    s = get_settings()
    client = genai.Client(api_key=s.gemini_api_key)
    audio = load_audio(DATA_DIR / name)
    vad = analyze_vad(audio.samples, audio.sr)
    noise = analyze_noise(audio.samples, audio.sr, vad)
    resp = client.models.generate_content(
        model=FLASH_MODEL,
        contents=[types.Part.from_bytes(data=encode_opus_ogg(audio.samples, audio.sr),
                                        mime_type="audio/ogg"),
                  build_prompt(audio.samples.size / audio.sr, noise.snr_db,
                               vad.speech_ratio)],
        config=types.GenerateContentConfig(
            temperature=s.gemini_temperature,
            response_mime_type="application/json",
            response_schema=GEMINI_RESPONSE_SCHEMA,
        ),
    )
    data = json.loads(resp.text)
    usage = getattr(resp, "usage_metadata", None)
    cost = gemini_cost(getattr(usage, "prompt_token_count", None),
                       getattr(usage, "candidates_token_count", None),
                       in_rate=FLASH_IN_PER_1M, out_rate=FLASH_OUT_PER_1M)
    return data, cost


def run_once(run_idx: int) -> dict:
    guard = SpendGuard()
    guard.check(EST_COST_PER_RUN)
    truth = load_truth()
    per_clip, cost = {}, 0.0
    for name in ANCHORS:
        data, c = classify_flash(name)
        cost += c
        pred = {"emotional_tone": data["emotional_tone"],
                "emotional_intensity": data["emotional_intensity"],
                "speaker_overlap_present": bool(data["speaker_overlap_present"])}
        per_clip[name] = {"pred": pred,
                          "correct": field_compare(pred, truth[name], FIELDS),
                          "cost_usd": c}
    guard.add(cost)
    payload = {"exp": "exp4_flash", "run": run_idx, "cost_usd": cost,
               "model_id": FLASH_MODEL,
               "pricing": {"in_per_1m": FLASH_IN_PER_1M, "out_per_1m": FLASH_OUT_PER_1M,
                           "source": PRICING_SOURCE},
               "per_clip": per_clip}
    log_run("exp4_flash", run_idx, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    for i in range(1, args.runs + 1):
        p = run_once(i)
        print(f"run {i}: ${p['cost_usd']:.4f}",
              {k: v["correct"] for k, v in p["per_clip"].items()})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify model id + pricing, then run live**

First verify the model exists and check posted pricing (WebFetch `https://ai.google.dev/pricing` or a 1-token trial call). Update `FLASH_MODEL`/`FLASH_IN_PER_1M`/`FLASH_OUT_PER_1M`/`PRICING_SOURCE` if reality differs — this is expected, not a deviation.
Run: `.venv/bin/python -m eval.experiments.exp4_flash --runs 3`
Expected: per-run correctness maps; cost ≈ 2–4× exp0.

- [ ] **Step 3: Fast suite, ruff, commit**

```bash
git add eval/experiments/exp4_flash.py
git commit -m "feat(experiments): E4 full Gemini Flash tone arm"
```

---

### Task 7: E5 — Deepgram diarization overlap

**Files:**
- Create: `eval/experiments/exp5_overlap.py`
- Test: `tests/unit/test_exp5_overlap_math.py`

**Interfaces:**
- Consumes: raw audio bytes (original files sent as-is; Deepgram sniffs format), `.env` `DEEPGRAM_API_KEY` (present), `classify_tone("dimensional", ...)` for the bonus, common harness.
- Produces: `turns_from_words(words: list[dict], max_intra_gap_s: float = 0.5) -> list[dict]` and `overlap_from_turns(turns: list[dict], min_overlap_s: float = 0.5, backchannel_max_s: float = 1.0, backchannel_max_words: int = 2) -> bool` (both pure, unit-tested); run log `exp5_overlap_run1.json` (single run — deterministic API).

- [ ] **Step 1: Write the failing tests for the pure overlap math**

```python
# tests/unit/test_exp5_overlap_math.py
from eval.experiments.exp5_overlap import overlap_from_turns, turns_from_words


def _w(word, start, end, speaker):
    return {"word": word, "start": start, "end": end, "speaker": speaker}


def test_turns_merge_consecutive_same_speaker_words():
    words = [_w("hi", 0.0, 0.3, 0), _w("there", 0.35, 0.6, 0), _w("yes", 2.0, 2.2, 1)]
    turns = turns_from_words(words)
    assert len(turns) == 2
    assert turns[0] == {"speaker": 0, "start": 0.0, "end": 0.6, "words": 2}
    assert turns[1]["speaker"] == 1


def test_turns_split_on_long_gap_same_speaker():
    words = [_w("a", 0.0, 0.2, 0), _w("b", 3.0, 3.2, 0)]
    assert len(turns_from_words(words)) == 2


def test_overlap_detects_real_crosstalk():
    turns = [
        {"speaker": 0, "start": 0.0, "end": 5.0, "words": 12},
        {"speaker": 1, "start": 4.0, "end": 8.0, "words": 9},  # 1.0s intersection
    ]
    assert overlap_from_turns(turns) is True


def test_overlap_ignores_backchannel():
    turns = [
        {"speaker": 0, "start": 0.0, "end": 5.0, "words": 12},
        {"speaker": 1, "start": 4.2, "end": 4.9, "words": 1},  # short "uh-huh"
    ]
    assert overlap_from_turns(turns) is False


def test_overlap_ignores_sub_threshold_intersection():
    turns = [
        {"speaker": 0, "start": 0.0, "end": 5.0, "words": 12},
        {"speaker": 1, "start": 4.8, "end": 8.0, "words": 10},  # 0.2s graze
    ]
    assert overlap_from_turns(turns) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/unit/test_exp5_overlap_math.py -q` → FAIL (module not found)

- [ ] **Step 3: Write `eval/experiments/exp5_overlap.py`**

```python
"""E5: measured overlap from Deepgram diarization (user-authorized recipient,
2026-07-17) instead of Gemini's judgment. Overlap = any cross-speaker turn
intersection >= 0.5s that isn't a bare back-channel (<= 1.0s own duration AND
<= 2 words). Thresholds are first-pass choices from the client's own
definition ("brief back-channel does not count"), NOT tuned on the eval.
Bonus (free, local): dimensional arm re-scored on customer-only audio."""

import argparse
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from autoace_audio.analyzers.tone.base import classify_tone
from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import load_audio
from eval.experiments.common import (
    ANCHORS, DATA_DIR, SpendGuard, field_compare, load_truth, log_run,
)

DG_URL = ("https://api.deepgram.com/v1/listen"
          "?model=nova-2&diarize=true&punctuate=false&smart_format=false")
DG_RATE_PER_MIN = 0.0043  # posted nova-2 prerecorded pay-as-you-go rate
EST_COST = 0.03  # ~4 audio-minutes total


def _dg_key() -> str:
    from dotenv import dotenv_values

    key = dotenv_values(".env").get("DEEPGRAM_API_KEY") or os.environ.get(
        "DEEPGRAM_API_KEY", "")
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY missing from .env")
    return key


def deepgram_words(path: Path) -> list[dict]:
    req = urllib.request.Request(
        DG_URL, data=path.read_bytes(),
        headers={"Authorization": f"Token {_dg_key()}",
                 "Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read())
    words = body["results"]["channels"][0]["alternatives"][0]["words"]
    return [{"word": w["word"], "start": w["start"], "end": w["end"],
             "speaker": w.get("speaker", 0)} for w in words]


def turns_from_words(words: list[dict], max_intra_gap_s: float = 0.5) -> list[dict]:
    turns: list[dict] = []
    for w in words:
        if (turns and turns[-1]["speaker"] == w["speaker"]
                and w["start"] - turns[-1]["end"] <= max_intra_gap_s):
            turns[-1]["end"] = w["end"]
            turns[-1]["words"] += 1
        else:
            turns.append({"speaker": w["speaker"], "start": w["start"],
                          "end": w["end"], "words": 1})
    return turns


def overlap_from_turns(turns: list[dict], min_overlap_s: float = 0.5,
                       backchannel_max_s: float = 1.0,
                       backchannel_max_words: int = 2) -> bool:
    for i, a in enumerate(turns):
        for b in turns[i + 1:]:
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


def customer_only_audio(samples: np.ndarray, sr: int,
                        turns: list[dict]) -> tuple[np.ndarray, int, str]:
    """Agent = speaker of the first turn (Erica opens every sample call).
    Returns (customer samples, customer speaker id, attribution note)."""
    if not turns:
        return samples, -1, "no turns; used full audio"
    agent = turns[0]["speaker"]
    speakers = {t["speaker"] for t in turns}
    if len(speakers) < 2:
        return samples, -1, "single speaker diarized; used full audio (ambiguous)"
    customer = next(s for s in sorted(speakers) if s != agent)
    parts = [samples[int(t["start"] * sr): int(t["end"] * sr)]
             for t in turns if t["speaker"] == customer]
    return np.concatenate(parts) if parts else samples, customer, "first-turn=agent rule"


def run_once(run_idx: int = 1) -> dict:
    guard = SpendGuard()
    guard.check(EST_COST)
    truth = load_truth()
    per_clip, minutes = {}, 0.0
    for name in ANCHORS:
        path = DATA_DIR / name
        audio = load_audio(path)
        minutes += audio.samples.size / audio.sr / 60.0
        words = deepgram_words(path)
        turns = turns_from_words(words)
        overlap = overlap_from_turns(turns)
        cust, cust_id, note = customer_only_audio(audio.samples, audio.sr, turns)
        vad = analyze_vad(cust, audio.sr)
        dim = classify_tone("dimensional", cust, audio.sr, vad, None)
        pred = {"speaker_overlap_present": overlap}
        per_clip[name] = {
            "pred": pred,
            "correct": field_compare(pred, truth[name], ["speaker_overlap_present"]),
            "n_words": len(words), "n_turns": len(turns),
            "attribution": note,
            "dimensional_customer_only": {
                "tone": dim.tone.value, "intensity": dim.intensity.value,
                "valence": dim.raw.get("valence"), "arousal": dim.raw.get("arousal"),
            },
        }
    cost = minutes * DG_RATE_PER_MIN
    guard.add(cost)
    payload = {"exp": "exp5_overlap", "run": run_idx, "cost_usd": cost,
               "audio_minutes": round(minutes, 2), "rate_per_min": DG_RATE_PER_MIN,
               "per_clip": per_clip}
    log_run("exp5_overlap", run_idx, payload)
    return payload


def main() -> None:
    argparse.ArgumentParser().parse_args()
    p = run_once(1)
    print(f"${p['cost_usd']:.4f}",
          {k: v["pred"] for k, v in p["per_clip"].items()})


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Unit tests, then live (single run)**

Run: `.venv/bin/pytest tests/unit/test_exp5_overlap_math.py -q` → `5 passed`
Run: `.venv/bin/python -m eval.experiments.exp5_overlap`
Expected: overlap predictions for the 3 anchors vs truth (false/true/true); dimensional customer-only AVD values in the log. Cost ≈ $0.02. Note: `dotenv_values` — `python-dotenv` ships with pydantic-settings; if the import fails, read the key by parsing `.env` lines directly (disclose in report).

- [ ] **Step 5: Fast suite, ruff, commit**

```bash
git add eval/experiments/exp5_overlap.py tests/unit/test_exp5_overlap_math.py
git commit -m "feat(experiments): E5 Deepgram diarization overlap + customer-only bonus"
```

---

### Task 8: Combined stack + run_all orchestrator

**Files:**
- Create: `eval/experiments/combined.py`, `eval/experiments/run_all.py`

**Interfaces:**
- Consumes: `read_runs`, `wins_field` from common; every exp module's `run_once`.
- Produces: `decide_stack() -> dict` (which levers won, from the run logs, via `wins_field`); run logs `combined_run{1..3}.json`; `run_all.py` executes exp0→exp5→combined in order, each skippable if its logs already exist (`--force` to redo), so the study is resumable without re-spending.

- [ ] **Step 1: Write `eval/experiments/combined.py`**

```python
"""Combined 'best stack': every lever that won its target field (2-of-3 rule
vs baseline) runs together. Tone/intensity source of truth = E4 flash if it
won any tone-family field else shipping lite; few-shot exemplars added if E2
won intensity; advocate pass wrapped last if E3 won tone; overlap from E5 if
it won; noise typing from E1 if it won presence on the noisy anchors.
This module REPORTS the stack decision in its log so the study table can
show exactly which levers the headline number contains."""

import argparse

from eval.experiments import (
    exp1_gap_noise, exp2_fewshot, exp3_advocate, exp4_flash, exp5_overlap,
)
from eval.experiments.common import (
    ANCHORS, SpendGuard, field_compare, load_truth, log_run, read_runs, wins_field,
)


def decide_stack() -> dict:
    base = read_runs("exp0_baseline")
    stack = {
        "flash": any(wins_field(base, read_runs("exp4_flash"), f, c)
                     for f in ["emotional_tone", "emotional_intensity",
                               "speaker_overlap_present"] for c in ANCHORS),
        "fewshot": any(wins_field(base, read_runs("exp2_fewshot"),
                                  "emotional_intensity", c) for c in ANCHORS),
        "advocate": any(wins_field(base, read_runs("exp3_advocate"),
                                   "emotional_tone", c) for c in ANCHORS),
        # E1/E5 are judged on their own evidence (different eval bases):
        "gap_noise": _gap_noise_won(),
        "deepgram_overlap": _overlap_won(),
    }
    return stack


def _gap_noise_won() -> bool:
    runs = read_runs("exp1_gap_noise")
    if not runs:
        return False
    noisy = ["call_002.ogg", "call_003.ogg"]
    per_run = [
        all(r["per_clip"].get(n, {}).get("present_correct") for n in noisy
            if not r["per_clip"].get(n, {}).get("skipped"))
        for r in runs
    ]
    return sum(per_run) >= 2


def _overlap_won() -> bool:
    runs = read_runs("exp5_overlap")
    if not runs:
        return False
    correct = [v["correct"]["speaker_overlap_present"]
               for v in runs[0]["per_clip"].values()]
    return sum(correct) == 3  # deterministic single run must be perfect to win


def run_once(run_idx: int, stack: dict) -> dict:
    truth = load_truth()
    guard = SpendGuard()
    guard.check(0.06)
    per_clip, cost = {}, 0.0
    for name in ANCHORS:
        # tone/intensity/overlap source
        if stack["fewshot"]:
            tone_data, c = exp2_fewshot.classify_with_exemplars(name)
        elif stack["flash"]:
            tone_data, c = exp4_flash.classify_flash(name)
        else:
            tone_data, c = exp4_flash.classify_flash(name) if stack["flash"] else \
                _shipping_classify(name)
        cost += c
        if stack["advocate"]:
            final, c2 = exp3_advocate.advocate_pass(
                name, tone_data["emotional_tone"], tone_data["emotional_intensity"],
                str(tone_data.get("rationale", "")))
            cost += c2
            tone_data = final
        pred = {"emotional_tone": tone_data["emotional_tone"],
                "emotional_intensity": tone_data["emotional_intensity"],
                "speaker_overlap_present": bool(tone_data["speaker_overlap_present"])}
        if stack["deepgram_overlap"]:
            dg = read_runs("exp5_overlap")[0]["per_clip"][name]["pred"]
            pred["speaker_overlap_present"] = dg["speaker_overlap_present"]
        per_clip[name] = {"pred": pred, "correct": field_compare(
            pred, truth[name],
            ["emotional_tone", "emotional_intensity", "speaker_overlap_present"])}
    guard.add(cost)
    payload = {"exp": "combined", "run": run_idx, "cost_usd": cost,
               "stack": stack, "per_clip": per_clip}
    log_run("combined", run_idx, payload)
    return payload


def _shipping_classify(name: str):
    from autoace_audio.analyzers.noise import analyze_noise
    from autoace_audio.analyzers.tone.base import classify_tone
    from autoace_audio.analyzers.vad import analyze_vad
    from autoace_audio.audio_io import load_audio
    from eval.experiments.common import DATA_DIR, gemini_cost

    audio = load_audio(DATA_DIR / name)
    vad = analyze_vad(audio.samples, audio.sr)
    noise = analyze_noise(audio.samples, audio.sr, vad)
    r = classify_tone("gemini", audio.samples, audio.sr, vad, noise.snr_db)
    data = dict(r.raw.get("response", {}))
    data.setdefault("emotional_tone", r.tone.value)
    data.setdefault("emotional_intensity", r.intensity.value)
    data.setdefault("speaker_overlap_present", bool(r.overlap_opinion))
    return data, gemini_cost(r.raw.get("prompt_tokens"), r.raw.get("output_tokens"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    stack = decide_stack()
    print("stack decision:", stack)
    for i in range(1, args.runs + 1):
        p = run_once(i, stack)
        print(f"run {i}: ${p['cost_usd']:.4f}",
              {k: v["correct"] for k, v in p["per_clip"].items()})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `eval/experiments/run_all.py`**

```python
"""Run the whole study in order, resumably: an experiment whose run logs
already exist is skipped unless --force. Sequential by design (16GB)."""

import argparse

from eval.experiments import (
    combined, exp0_baseline, exp1_gap_noise, exp2_fewshot, exp3_advocate,
    exp4_flash, exp5_overlap,
)
from eval.experiments.common import SpendGuard, read_runs

STAGES = [
    ("exp0_baseline", exp0_baseline, 3),
    ("exp1_gap_noise", exp1_gap_noise, 3),
    ("exp2_fewshot", exp2_fewshot, 3),
    ("exp3_advocate", exp3_advocate, 3),
    ("exp4_flash", exp4_flash, 3),
    ("exp5_overlap", exp5_overlap, 1),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for name, mod, runs in STAGES:
        done = len(read_runs(name))
        if done >= runs and not args.force:
            print(f"{name}: {done} runs exist, skipping")
            continue
        for i in range(done + 1, runs + 1):
            mod.run_once(i)
            print(f"{name} run {i} done; study total ${SpendGuard().total():.2f}")
    stack = combined.decide_stack()
    print("stack:", stack)
    if len(read_runs("combined")) < 3 or args.force:
        for i in range(len(read_runs("combined")) + 1, 4):
            combined.run_once(i, stack)
    print(f"STUDY COMPLETE. total spend ${SpendGuard().total():.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Fix the redundant conditional in combined.run_once**

The `else` branch above contains a leftover double-check (`exp4_flash.classify_flash(name) if stack["flash"] else _shipping_classify(name)` inside the `else` of the same condition). Simplify to:

```python
        if stack["fewshot"]:
            tone_data, c = exp2_fewshot.classify_with_exemplars(name)
        elif stack["flash"]:
            tone_data, c = exp4_flash.classify_flash(name)
        else:
            tone_data, c = _shipping_classify(name)
```

- [ ] **Step 4: Run the full study end-to-end (resumable; exp0–exp5 logs already exist from Tasks 2–7, so only combined runs live here)**

Run: `.venv/bin/python -m eval.experiments.run_all`
Expected: skip lines for exp0–exp5, a printed stack decision, 3 combined runs, `STUDY COMPLETE. total spend $X.XX` (must be < $10).

- [ ] **Step 5: Fast suite, ruff, commit**

```bash
git add eval/experiments/combined.py eval/experiments/run_all.py
git commit -m "feat(experiments): combined best-stack + resumable run_all orchestrator"
```

---

### Task 9: Study document + pitch page

**Files:**
- Create: `docs/experiments/2026-07-17-budget-accuracy-study.md`
- Create: `/Users/pratikyesare/Test/BUDGET-PITCH.md` (OUTSIDE the repo — never `git add` it)
- Modify: `README.md` (one line in the docs section linking the study)

**Interfaces:**
- Consumes: every run log in `out/experiments/` (`read_runs`), `docs/superpowers/specs/2026-07-17-budget-accuracy-study-design.md` (methodology source).

- [ ] **Step 1: Write the study document** with these exact sections, every number transcribed from run logs (no estimates anywhere):

```markdown
# Budget–Accuracy Study (2026-07-17)

## Why this study
[2 short paragraphs: shipping cost $0.00146/min vs $0.003 ceiling; the known
miss cluster; question: what does more spend buy? n=3 anchors — every result
here is directional evidence for a pilot, not a statistical guarantee.]

## Method (condensed from the design spec)
[Ablation ladder; 3x repeats; 2-of-3 improvement rule; leave-one-out
few-shot; spend guard; what each experiment is allowed to touch.]

## Baseline (shipping config, 3 runs)
[Table: per anchor x per field x per run — tone/intensity/overlap correctness;
mean cost/audio-min.]

## E1 gap-listening noise  [table + verdict paragraph]
## E2 audio few-shot intensity  [table incl. the call_001 medium-only-exemplars caveat]
## E3 devil's-advocate pass  [wins AND regressions, equal prominence]
## E4 Gemini Flash arm  [table + measured $/min at recorded pricing + source]
## E5 Deepgram diarization overlap  [table + the dimensional customer-only bonus]

## Combined best stack
[The stack decision (which levers), 3-run table, headline: "at $X.XXX/min,
the anchors score Y/Z on the judgment fields vs W/Z at baseline".]

## Cost vs accuracy summary
| config | $/audio-min (measured) | tone | intensity | overlap | noise presence (aug set) |
[one row per config; baseline first; combined last]

## Recommendation
[Tier framing: what fits inside the current $0.003 ceiling; what needs
~$0.005–0.007; what we do NOT recommend buying and why (negative results).]

## Limitations
[n=3; synthetic beds may be adversarial for both AED and Gemini; single-day
variance; thresholds in E5 are first-pass; call_002 tone may be label noise.]
```

- [ ] **Step 2: Write the pitch page** `/Users/pratikyesare/Test/BUDGET-PITCH.md` — one page, plain English, for presenting: the headline combined-stack number, the per-lever "what each dollar buys" list, the two negative results stated as savings ("we tested X and it does NOT pay — don't spend there"), and the ask (approve a pilot at $Y/min on a bigger labeled sample).

- [ ] **Step 3: Link the study from README** — add under the docs/limitations section: `See also: [Budget–accuracy study](docs/experiments/2026-07-17-budget-accuracy-study.md) — measured evidence for what a higher per-minute budget buys.`

- [ ] **Step 4: Verify no verbatim customer speech, lint, commit, push**

Run: `grep -ri "mama" docs/experiments/ README.md` → no matches (paraphrase rule).
Run: `make lint && make test`

```bash
git add docs/experiments/2026-07-17-budget-accuracy-study.md README.md
git commit -m "docs(experiments): budget-accuracy study — measured ablation of five spending levers"
git push
```

(Do NOT stage `Test/BUDGET-PITCH.md` — it lives outside the repo.)

---

## Self-Review

- **Spec coverage:** §2 protocol → Tasks 1–2 (baseline, repeats, rules in common.py); §3 E1–E5 → Tasks 3–7; combined → Task 8; §4 layout → Tasks 1–8 file structure; §5 deliverables → Task 9; §6 budget/safety → SpendGuard (Task 1) + per-task `guard.check/add` + global constraints; §7 honesty items → E2 caveat note, E3 regression logging, E5 attribution note, negative-results framing in Task 9.
- **Placeholder scan:** none — every step carries code or exact commands; Task 9's bracketed section notes are content instructions for a docs task, with the table schema fixed.
- **Type consistency:** `run_once(run_idx)` uniform across exp modules (E5's default arg included); `wins_field(baseline_runs, lever_runs, field, clip)` matches its Task 1 definition at every call site; `classify_with_exemplars`/`classify_flash`/`advocate_pass` signatures match between their defining tasks and Task 8's imports; combined.run_once's Step 3 fix is part of the task.
