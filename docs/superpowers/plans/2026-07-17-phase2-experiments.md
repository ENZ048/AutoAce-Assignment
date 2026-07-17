# Phase-2 Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Two greenlit hardening experiments that complete the accuracy-vs-cost curve: E6 (majority-vote harness for E1's gap-listening question) and E7 (tone self-consistency voting at sampling temperature), plus the docs update that turns the "$0.002 operating point" from projection into measurement (or honestly reports that it failed).

**Architecture:** Both experiments are new modules in `eval/experiments/` cloning the proven pattern: `common.py` SpendGuard + run logging + wins/loses helpers; import (never copy) the existing question/prompt builders; per-clip tokens + cost_usd; run-level model + nested pricing; 3 runs × 3 anchor calls; TDD with mock-level tests.

**Tech Stack:** Established — Python 3.12, google-genai, pytest. No new dependencies.

## Global Constraints

- User authorization: audio to Google (paid tier) ONLY for these two experiments (Deepgram not involved; no new vendors).
- SpendGuard before every live call; spend cap $10.00 (current $0.1581).
- Standing log amendments: per-clip `{"tokens": {"in": N, "out": N}}` + `cost_usd`; run-level `"model"` + nested `"pricing"` with `source`.
- Wins AND regressions equal prominence via `common.wins_field`/`common.loses_field` — never reimplemented.
- Never commit `data/`, `.env`, `.superpowers/`, `out/`. Conventional commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- TDD; `make test` + `make lint` green before every commit.

---

### Task 1: E6 — gap-listening majority-vote harness (`exp6_gap_vote.py`)

**Files:** Create `eval/experiments/exp6_gap_vote.py`, `tests/unit/test_exp6_gap_vote.py`.

**Question:** does asking E1's gap-listening question 3× per clip and taking the majority stabilize the win that was 3/3 in one session and 1/3 in another?

**Binding rules:**
- Import `ask_gemini_gaps` (and the gap-extraction helper) from `exp1_gap_noise` — the question text and audio-segment logic must be byte-identical to E1's. Same temperature as shipped (0.1). The variance being tested is call-to-call, not parameter-induced.
- Per clip per run: 3 independent gap calls → votes. Majority rules (pin exactly): `present` = ≥2 of 3 votes true. `type` = modal normalized string (lowercase/strip) among the present-true votes; if no modal winner among 2 true votes (they disagree), take the vote with higher self-reported noise confidence if the response carries one, else the first true vote. All ties toward absent.
- Per-clip log: all 3 raw votes (each with tokens + cost_usd) + the majority verdict + per-clip total cost.
- Anchors and expectations to report against: call_003 static (target: majority-confirmed in 3/3 runs → the harness works); call_001 (must stay absent every run — voting must not manufacture false positives); call_002 TV (known miss — report if voting changes it, in either direction).
- Compute and report: the measured $/audio-min of "shipping + voted gap question" (baseline $0.00146 + 3× measured gap-question marginal) — this is the operating-point number for the curve; state it explicitly.
- Runs: 3. Estimated cost ≤ $0.05 total; SpendGuard estimate $0.02/run.

**Steps:** failing mock tests (majority rules incl. both tie shapes, log shape, import-not-copy identity check on the question builder) → implement → live 3 runs → `make test`/`make lint` → commit `feat(experiments): E6 gap-listening majority-vote harness` → report per-run vote tables.

---

### Task 2: E7 — tone self-consistency vote (`exp7_tone_vote.py`)

**Files:** Create `eval/experiments/exp7_tone_vote.py`, `tests/unit/test_exp7_tone_vote.py`.

**Question:** does classic self-consistency (3 samples at diversity temperature, majority vote) beat the shipping single greedy call on tone/intensity? Baseline tone at temp 0.1 is run-deterministic, so voting at 0.1 would trivially null — the experiment is voting at **temperature 0.7** (pin this; it is the point of the test) vs the temp-0.1 single-call baseline.

**Binding rules:**
- Import `build_prompt` + `GEMINI_RESPONSE_SCHEMA` from the shipping arm (identity-asserted in tests); model = shipping `gemini-3.1-flash-lite`; ONLY the temperature and vote count differ from baseline.
- Majority on `emotional_tone`; tie (3-way split) → the vote with highest `tone_confidence`, further tie → first vote (deterministic). Same rule for `emotional_intensity` voted independently.
- Per-clip log: all 3 raw votes (tone, intensity, tone_confidence, tokens, cost_usd) + majority verdicts.
- Compare vs exp0 baseline via `wins_field`/`loses_field` on tone AND intensity; equal prominence; note per-clip vote dispersion (3-same / 2-1 / 3-way) — dispersion itself is a finding (measures how unstable the model is at 0.7 on these calls).
- Runs: 3. Estimated ≤ $0.06 total; SpendGuard estimate $0.02/run.

**Steps:** failing mock tests (majority + both tie rules, dispersion counter, temp wiring = 0.7 while baseline stays 0.1, log shape, prompt identity) → implement → live 3 runs → `make test`/`make lint` → commit `feat(experiments): E7 tone self-consistency vote at sampling temperature` → report vote-dispersion + verdict tables.

---

### Task 3: Docs — curve completion (`docs/experiments/` + pitch + study doc)

**Files:** Modify `docs/experiments/2026-07-17-budget-accuracy-study.md` (append "Phase 2: hardening experiments" section), `/Users/kishorrane/Test/BUDGET-PITCH.md` (update the curve section: projection → measured, or honest failure), README experiments line if the study doc's scope line changes.

**Binding rules:**
- Numbers transcribed from exp6/exp7 run logs only; source footnotes per table.
- The operating-point claim gets exactly one of two treatments: (a) E6 majority confirmed 3/3 runs → state the measured $/min operating point and mark the curve point MEASURED; (b) anything less → the pitch keeps the "do not use out loud" guardrail, and the study doc reports the voting harness result honestly (including if voting manufactured any false positive on call_001 — that would be a hard "don't ship").
- E7: whatever the result (win/null/negative + dispersion), it becomes a curve point with its measured $/min.
- BUDGET-PITCH.md is NEVER committed; study doc + README changes are committed: `docs(experiments): phase-2 hardening results` + trailer.
- Self-review: re-verify 4 random numbers vs logs, list them.

---

## Self-review notes

- Both experiments reuse proven modules by import with identity tests — no question/prompt drift possible.
- Tie rules pinned deterministically for reproducibility; temperature difference in E7 is the experiment, documented as such.
- Task 3 has an explicit honest-failure branch — the curve pitch is only upgraded on a 3/3 measured result.
