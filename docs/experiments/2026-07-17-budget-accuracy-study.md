# Budget–Accuracy Study (2026-07-17)

*Full source data: `out/experiments/` (gitignored; every number below is transcribed
from these run logs, one live-API run per file, or derived from logged tokens
where noted). Design source: `docs/superpowers/specs/2026-07-17-budget-accuracy-study-design.md`.
Total spend: **$0.1580927640625 of the $10.00 cap** (`out/experiments/spend.json`).*

## Why this study

The shipping backend costs **$0.00146/audio-minute** (measured bake-off figure,
`out/bakeoff.md`) against a client-set **$0.003/audio-minute ceiling** — roughly
half the budget is unspent. It also misses a known, disclosed cluster of
fields on the 3 labeled anchor calls: `emotional_tone` on `call_002.ogg`
(the model reads a single Spanish profanity phrase as decisive frustration
evidence, despite the rest of the call reading calm), `emotional_intensity`
one band low on 2 of 3 anchors, `background_noise_type` wording that is
best-effort/approximate, and `speaker_overlap_present` wrong on 1 of 3
anchors. The open question: if AutoAce were willing to spend more per
minute, what would that money actually buy back?

This study answers that with five live, measured spending levers — a bigger
second look at the same audio, a bigger model, exemplar audio, a
noise-focused sub-question, and a second vendor (Deepgram) — each run
against the same 3 labeled anchors, 3 times where the API is
temperature-sampled. **n=3 anchors is a very small sample.** Every result
below is directional evidence for a pilot decision, not a statistically
powered guarantee — a single clip flipping changes a lever's measured
accuracy by 33 points. Read the verdicts as "worth piloting on more data" or
"don't spend here," not as final numbers.

## Method (condensed from the design spec)

**Ablation ladder.** Baseline (shipping config) is measured first, 3 times,
before any lever runs — every experiment's delta is reported against this
measured distribution, not a single historical run. Five levers (E1–E5) each
add one spending mechanism on top of or in place of the baseline call; a
sixth stage (Combined) stacks whichever levers individually net a real win.

**Repeats and the 2-of-3 rule.** Every Gemini-dependent config runs 3× (the
API samples at `gemini_temperature=0.1`, so there is real, if small,
run-to-run variance). Deepgram's diarization call (E5) is deterministic and
runs once. A lever counts as a genuine "win" on a field×clip cell only under
`eval/experiments/common.wins_field`: the lever is correct in ≥2 of its 3
runs on a clip where baseline was correct in ≤1 of its 3 runs. Its mirror,
`loses_field`, catches the reverse (baseline mostly-right, lever
mostly-wrong) at the same threshold. A lever is included in the combined
stack only if wins strictly exceed losses **and** wins > 0 — a single
mechanical `wins_field` hit is not enough by itself (E3 and E4 each clear
one and are still excluded, below).

**Leakage.** E2's audio exemplars use leave-one-out: scoring call *k* draws
its two exemplars only from the other two anchors, never from *k* itself. No
threshold anywhere — in the shipping pipeline or in the experiment
harness — was tuned on this study's eval data.

**Spend guard.** A file-backed cumulative tracker (`out/experiments/spend.json`)
checks a projected cost before every live run and refuses to start one that
would cross a **$10.00 hard cap**, warning past $7.00. Every dollar below is
real, metered API spend, not an estimate.

**What each experiment is allowed to touch.** Experiment modules import the
shipping classifier/analyzer functions and pass overrides as *arguments*;
none of them mutate `config.py`'s shipping defaults or `fusion.py`. The
shipping pipeline's own behavior is unchanged by this study.

### Data provenance

- `exp0_baseline`, `exp1_gap_noise`, `exp2_fewshot`, and `exp3_advocate`'s
  run logs do not carry a `model` field. The model behind all four is stated
  from source, not the log: `src/autoace_audio/config.py` —
  `gemini_model: str = "gemini-3.1-flash-lite"`. `exp4_flash`, `exp5_overlap`,
  and `combined`'s logs do carry an explicit `model`/`model_id` field and are
  quoted directly from the log in the sections below.
- `exp3_advocate`'s per-clip log entries carry token counts
  (`tokens.first.{in,out}`, `tokens.final.{in,out}`) but no per-clip
  `cost_usd` (unlike every other experiment module). Per-clip cost is
  derived once, here, and reused throughout this document:

  ```
  cost_usd = (tokens.first.in + tokens.final.in) × $0.50/1e6
           + (tokens.first.out + tokens.final.out) × $1.50/1e6
  ```

  Cross-checked against the logged run-level total: `call_001.ogg` run 1
  (first in=1513/out=100, final in=1002/out=123) → $0.001592;
  `call_002.ogg` (first in=1613/out=110, final in=1113/out=141) →
  $0.0017395; `call_003.ogg` (first in=5039/out=108, final in=4535/out=155)
  → $0.0051815. Sum = **$0.008513**, exactly matching
  `exp3_advocate_run1.json`'s logged `cost_usd`.

## Baseline (shipping config, 3 runs)

Truth (`data/labels.csv`): `call_001.ogg`=upset/high/overlap **false**;
`call_002.ogg`=neutral/medium/overlap **true**; `call_003.ogg`=satisfied/medium/overlap
**true**.

| clip | tone pred | tone | intensity pred | intensity | overlap pred | overlap |
|---|---|---|---|---|---|---|
| call_001.ogg | upset | ✓ | medium | ✗ (truth high) | false | ✓ |
| call_002.ogg | frustrated | ✗ (truth neutral) | medium | ✓ | false | ✗ (truth true) |
| call_003.ogg | satisfied | ✓ | low | ✗ (truth medium) | true | ✓ |

**Identical on all 3 runs** — every field×clip cell landed the same way each
time, confirming these are systematic model behaviors at
`gemini_temperature=0.1`, not one-off sampling noise. Score: **5/9** correct
across tone+intensity+overlap (2/3 tone, 1/3 intensity, 2/3 overlap).

Cost: run 1 $0.0045685, run 2 $0.0045700, run 3 $0.0045670 (mean
**$0.0045685/run**; only output-token counts vary run to run, e.g.
`call_001.ogg` 100/106/102 out-tokens). $/audio-min: **$0.00146** (bake-off
headline, `out/bakeoff.md`) — this study's own baseline measurement, divided
by the 3 anchors' real combined duration (3.964018 audio-min, `ffprobe`, per
`study-task-6-report.md`), works out to **$0.0011525/audio-min**, a lower,
separately-measured figure from a different task/run of the same shipping
arm. Both numbers are real and pre-exist this document; neither is silently
preferred over the other — see the Cost summary footnote.

*Source: `out/experiments/exp0_baseline_run{1,2,3}.json`.*

## E1 — gap-listening noise question

**Mechanism:** a second, focused Gemini call fed only the clip's
concatenated VAD gap audio (≥1.0s segments, ≤60s total), asking directly
whether background noise is present, not the whole-call classification
prompt. Scored on presence only (type reported alongside truth, not scored
exact).

| clip | gap audio | pred present | pred type | truth present | presence correct |
|---|---|---|---|---|---|
| call_001.ogg | 16.1s | false | — | false | ✓ |
| call_002.ogg | 10.0s | false | — | true (TV) | ✗ |
| call_003.ogg | 40.6s | true | "static" | true (sharp static) | ✓ |
| 8 of 9 synthetic noise clips (aug_TV/electrical_hum/static × snr 18/10/2) | 15.5–16.3s each | false | — | true (all 9) | ✗ (8 of 9) |
| aug_static_snr2.wav (the 9th) | 16.0s | true | "traffic" | true (high severity) | ✓ (type wrong; borderline) |

**Identical across all 3 runs**: presence accuracy **3/12 per run (25%)**,
same predictions every time. Cost: run 1 $0.0037325, run 2 $0.0037190, run 3
$0.0037325 — **$0.0111840 total** for the 3-run, 12-clip standalone
experiment.

**Verdict — the study's one standalone win, with an equally-important
reliability caveat.** On its own 3-run measurement (this task), E1 is the
**only lever with wins > losses**: it is the first live confirmation, ever,
of `call_003.ogg`'s noise bed (prior whole-call prompting had denied noise
on this call in every test recorded in `docs/decisions.md`), and it
introduces no false positive on `call_001.ogg`. `call_002.ogg`'s TV noise
stays undetected (gap-only listening doesn't rescue it), and 8 of 9
synthetic noise beds are still denied outright even at the "high" severity
band — gap-listening does not generally solve noise-typing on synthetic
audio; its one real win is on genuine, non-synthetic background noise.

**Equal-prominence caveat (from re-running E1 inside the combined-stack
session, Study Task 8):** the exact same function, prompt, and audio, called
fresh three more times as part of the combined run, reproduced the
`call_003.ogg` "static" confirmation in only **1 of 3 runs** — down from the
clean 3/3 this task measured standalone. `gap_seconds` was identical across
all 3 combined-session runs (40.6s, confirming the audio input side is
stable); this is Gemini sampling variance on the gap-listening question
itself, not a code change. **This session-variability finding carries equal
weight to the win itself in this document**: E1's real accuracy on its one
success case is somewhere between 1/3 and 3/3 depending on which session you
measure, and it needs a reliability harness (e.g., multi-sample voting: ask
3 times, take the majority) before it should be trusted as a production
feature rather than a "sometimes-it-helps" bonus.

*Source: `out/experiments/exp1_gap_noise_run{1,2,3}.json` (standalone);
`out/experiments/combined_run{1,2,3}.json` (reliability re-measurement).*

## E2 — audio few-shot intensity

**Mechanism:** the shipping prompt gains two ~15–20s audio exemplars, each
labeled with its true intensity, drawn leave-one-out (scoring call *k* never
sees *k*'s own audio as an exemplar).

| clip | exemplar pool (other 2 anchors' intensity) | pred intensity | intensity correct | vs. baseline |
|---|---|---|---|---|
| call_001.ogg (truth **high**) | medium, medium — no same-band exemplar available | medium | ✗ | **byte-identical** |
| call_002.ogg (truth medium) | **high**, medium | medium | ✓ | **byte-identical** |
| call_003.ogg (truth medium) | **high**, medium | low | ✗ | **byte-identical** |

**Identical across all 3 runs, and identical to baseline on every clip,
every field (tone and intensity), every run** — not a partial shift, a
single run-to-run flicker, nothing. Formal verdict: **0 wins out of 6
field×clip cells** (`common.wins_field`).

**The call_001 medium-only-exemplars caveat, and why it doesn't fully
explain the null:** `call_001.ogg` is scored against a medium+medium pool —
the 3-anchor set's only "high"-labeled call is call_001 itself, so it
structurally never sees a same-band exemplar for its own true intensity.
But `call_002.ogg` and `call_003.ogg` **do** each get a genuine high+medium
contrastive pool (both include call_001's real "high" exemplar), and
neither moved either — `call_003.ogg` in particular had a valid high
exemplar in context and only needed to move one band up, and stayed at
`low`. The exemplar-pool gap is real but is not sufficient on its own to
explain a fully null result across all three clips.

**Verdict: clean, full negative result.** Leave-one-out audio few-shot
exemplars, as implemented, measurably changed nothing. This should be
reported as "no effect measured," not softened into a near-miss.

Cost: run 1 $0.0061665, run 2 $0.006156, run 3 $0.0061665 — **$0.018489
total**. $/audio-min ≈ **$0.001555** (mean run cost ÷ 3.964018 audio-min) —
more than baseline, for zero accuracy return.

*Source: `out/experiments/exp2_fewshot_run{1,2,3}.json`.*

## E3 — devil's-advocate tone pass

**Mechanism:** a second Gemini call, given the first pass's own verdict and
rationale, is asked to argue the strongest case for a *different* reading of
the customer, then give a final verdict. Final verdict overrides the first
pass.

| clip | first-pass tone (correct?) | final tone (correct?) | outcome |
|---|---|---|---|
| call_001.ogg | upset (✓) | neutral (✗) | **regression** — flips a right answer wrong, 3/3 runs |
| call_002.ogg | frustrated (✗) | neutral (✓) | **the win** — 3/3 runs |
| call_003.ogg | satisfied (✓) | frustrated (✗) | **regression** — flips a right answer wrong, 3/3 runs |

Every flip was unanimous across all 3 runs (no run-to-run split on any
clip). Secondary field: `call_002.ogg`'s previously-correct intensity
(medium) also regresses to wrong (low) in the same pass; `call_003.ogg`'s
previously-wrong intensity (low) incidentally becomes correct (medium) —
noted as secondary per the study's own protocol, since E3 is scored on tone.

**Verdict — net negative, reported with the win and the regressions at equal
weight.** The advocate pass does exactly what it was hypothesized to do on
`call_002.ogg`: challenged with its own reasoning, the model abandons the
profanity-fixated `frustrated` read and lands on the correct `neutral`,
every run. But the same challenge destabilizes two calls that were already
right — as implemented, **a second devil's-advocate pass is a net negative
for tone accuracy on this 3-anchor set: 1 win, 2 regressions.** Formal tone
score falls from baseline's 2/3 to E3's 1/3.

Cost: run 1 $0.008513, run 2 $0.0084965, run 3 $0.0084975 — **$0.025507
total** (per-clip costs derived from tokens; see Data provenance above).
$/audio-min ≈ **$0.002145** (mean run cost ÷ 3.964018 audio-min) — ~1.5× the
baseline bake-off figure, for a net-negative accuracy outcome.

*Source: `out/experiments/exp3_advocate_run{1,2,3}.json`.*

## E4 — Gemini Flash arm

**Pre-flight finding (Study Task 6):** the design's originally-named model,
`gemini-3.1-flash`, does not exist on this API key — confirmed by both a
full model-list scan and a direct `GetModel` lookup (hard `404`), and
independently corroborated by the absence of any "Gemini 3.1 Flash" line on
Google's own pricing page. Controller-selected substitute, used throughout
E4: **`gemini-3.5-flash`** (GA, no preview tag) at **$1.50/1M audio-input
tokens, $9.00/1M output tokens** (source:
`https://ai.google.dev/gemini-api/docs/pricing`, checked 2026-07-17 — both
figures logged per-run in `exp4_flash_run{1,2,3}.json`'s `pricing` block).

| clip | tone (3 runs) | intensity (3 runs) | overlap (3 runs) |
|---|---|---|---|
| call_001.ogg | upset, ✓ ×3 | medium, ✗ ×3 (truth high) | false,false,**true** — 2/3 ✓ |
| call_002.ogg | **neutral, ✓ ×3 (the win — baseline wrong ×3)** | **low, ✗ ×3 (regression — baseline right ×3)** | false ×3, ✗ (unchanged miss) |
| call_003.ogg | satisfied, ✓ ×3 | low, ✗ ×3 (truth medium) | **false ×3, ✗ (regression — baseline right ×3)** |

**Verdict — net negative, at over-ceiling cost.** E4 wins the exact same
field/clip E3 won (`call_002.ogg` tone), but trades it for two new full
regressions (`call_002.ogg` intensity, `call_003.ogg` overlap) plus a
partial wobble (`call_001.ogg` overlap, 2 of 3 runs). Net across the 9
field×clip cells: +1 win, −2 full regressions, −1 partial wobble, 5
unchanged. **The bigger model does not clearly perform better on this
3-anchor sample — it trades one known miss for two different ones.**

**Cost — measured, not estimated, and over the client's ceiling.** Real
token usage across the 3 anchors (237.841s = 3.964018 audio-min, `ffprobe`):
mean **$0.0157395/run → $0.0039706/audio-min**. That is **2.72×** the
shipping Lite arm's $0.00146/audio-min bake-off figure, and **exceeds the
$0.003/audio-min ceiling by ~32%** (not "slightly over" — a third over).

Cost: run 1 $0.0156225 ($0.003941/audio-min), run 2 $0.0157215
($0.003966/audio-min), run 3 $0.0158745 ($0.004005/audio-min). Total
**$0.0472185**.

*Source: `out/experiments/exp4_flash_run{1,2,3}.json`.*

## E5 — Deepgram diarization overlap

**Pricing finding (undisclosed in the original design, found in Task 7's
mandatory pre-flight check):** Deepgram's live pricing page
(`https://deepgram.com/pricing`, checked 2026-07-17) lists Nova-2
prerecorded pay-as-you-go at the design's assumed **$0.0043/min** — but
**separately** lists a **$0.0020/min Speaker Diarization add-on**, not
mentioned anywhere in the study's original design. Since `diarize=true` is
this experiment's entire premise, omitting the add-on would understate
exactly the cost this study exists to measure. **Effective rate used
throughout: $0.0063/audio-minute** (0.0043 + 0.0020), logged per-clip in
`exp5_overlap_run1.json`'s `pricing` block.

| clip | diarization result | overlap pred | truth | correct? | vs. baseline |
|---|---|---|---|---|---|
| call_001.ogg | 1 speaker found (fallback to full audio) | false | false | ✓ | unchanged |
| call_002.ogg | 1 speaker found (fallback to full audio); also undertranscribed — 13 words / 34.96s vs. call_003's 386 words / 171.9s | false | **true** | ✗ | unchanged miss (the target case, still broken) |
| call_003.ogg | 2 speakers correctly split (33 turns); 0 cross-speaker spans cleared the 0.5s threshold | false | **true** | ✗ | **regression** — baseline was right 3/3 |

**Verdict — 0 wins, 1 regression; the target case fails for a structural,
not a threshold, reason.** `call_002.ogg` — the call E5 was specifically
built to fix — never gets a chance: diarization collapses it (and
`call_001.ogg`) to a single detected speaker, making
`speaker_overlap_present=True` mechanically unreachable regardless of any
overlap-math threshold. On `call_003.ogg`, the one clip where diarization
*did* work correctly, word-timestamp overlap detection still found zero
qualifying crosstalk despite the human label — and it broke a previously
correct answer. **As specified and measured, Deepgram diarization overlap
does not improve on the shipping judgment on this sample, for a real
structural reason (diarization reliability), and costs $0.0063/audio-minute
for the attempt** — a genuine, previously-undisclosed cost finding equal in
weight to the accuracy result.

**Customer-only dimensional bonus — one honest line.** Re-running the local
dimensional (audeering wav2vec2) tone arm on Deepgram's diarized
customer-only audio formally wins intensity on 2 of 3 clips (vs. baseline's
1 of 3) — but the diarization collapse above means only `call_003.ogg` is a
*genuine* customer-only measurement; `call_001.ogg`'s win is confounded
(diarization fell back to full audio, not isolated customer speech, so it's
really "dimensional-on-full-audio," a different, already-explored
condition). Tone is not competitive (0 of 3). **Read this as promising,
needs more labeled data — not included in the stack.**

Cost: `call_001.ogg` $0.0032487, `call_002.ogg` $0.0036709, `call_003.ogg`
$0.0180517 — **$0.0249713 total** for the single deterministic run (3.9637
audio-min). Dimensional bonus arm: $0 marginal (local inference).

*Source: `out/experiments/exp5_overlap_run1.json`.*

## Combined best stack

**Stack decision (controller-binding, reproduced by a data-driven
`decide_stack()` reading every lever's on-disk logs):**

| Lever | Wins | Losses | Net | Included? |
|---|---|---|---|---|
| E1 gap-listening | 1 (`call_003` presence) | 0 | +1 | **YES — the only lever with wins > losses** |
| E2 few-shot intensity | 0 | 0 | 0 | no — clean null, 0/6 wins vs. baseline |
| E3 devil's-advocate | 1 (`call_002` tone) | 2 (`call_001`, `call_003` tone) | −1 | no — 1 win vs. 2 regressions on its own target field |
| E4 Flash tone source | 1 (`call_002` tone) | 2 (`call_002` intensity, `call_003` overlap) | −1 | no — same shape as E3; also 2.72× cost, over the $0.003/audio-min ceiling |
| E5 diarization overlap | 0 | 1 (`call_003` overlap) | −1 | no — 0 wins, 1 regression |
| E5 bonus (customer-only dimensional intensity) | — | — | — | no — not a planned stack component; its one clean win is confounded; recorded above as a lead, not a result |

**Combined = shipping baseline + E1 gap-listening, nothing else**
(`included_levers: ["E1"]`, verbatim from every `combined_run*.json`).

| clip | tone | intensity | overlap | vs. baseline |
|---|---|---|---|---|
| call_001.ogg | upset ✓ | medium ✗ | false ✓ | **byte-identical, all 3 runs** |
| call_002.ogg | frustrated ✗ | medium ✓ | false ✗ | **byte-identical, all 3 runs** |
| call_003.ogg | satisfied ✓ | low ✗ | true ✓ | **byte-identical, all 3 runs** |

**Tone, intensity, and overlap are byte-identical to baseline on all 9
clip×run cells across all 3 combined runs — E1 only ever touches noise
fields, confirmed live: stacking is safe, nothing leaked.** Score stays
5/9, identical to baseline.

**Noise presence — the E1 reliability finding surfaces directly here.**
`call_003.ogg`'s "static" confirmation (the entire reason E1 is in the
stack) landed in only **1 of the 3 combined runs** (run 3; runs 1–2 show
the same blind `present=false` result baseline itself shows). `call_002`'s
TV stays undetected in all 3 runs (unchanged, pre-existing). Net: the
combined stack still beats baseline on `call_003` noise presence (1/3 vs.
baseline's 0/3, since baseline has no path to detect it at all) — but it is
not the clean, reliable win Study Task 3's original 3/3 standalone
measurement suggested. **Restated here with the same prominence as in the
E1 section above, because it is this document's single most important
caveat: the study's one winning lever is measurably less reliable than its
first measurement, and that instability — not a fixed accuracy number — is
what a reader should take away about E1.**

**Headline:** at **$0.001431/audio-min** (mean $0.0056725/run ÷ 3.964018
audio-min; comparable to baseline's own $0.0011525/audio-min measured the
same way, or the $0.00146 bake-off headline — see the cost table footnote),
the anchors score **5/9** on the tone+intensity+overlap judgment fields,
identical to baseline's 5/9, plus a noise-presence result that ranges from
1/3 to 3/3 depending on session. **Stacking E1 costs essentially nothing
extra and never makes anything worse — the only open question is how
reliably its one win reproduces.**

*Scope note:* the combined stack's live run scores only the 3 real anchors
— it does not re-run the 9 synthetic noise-augmented clips E1 was also
measured against standalone, even though the original study design listed
that as in scope. This document's "noise presence (aug set)" figures in the
cost table below are E1's own standalone measurement, not a combined-stack
number.

*Source: `out/experiments/combined_run{1,2,3}.json`.*

## Cost vs accuracy summary

| config | $/audio-min (measured) | tone (n=3) | intensity (n=3) | overlap (n=3) | noise presence (aug set, n=9 synthetic) |
|---|---|---|---|---|---|
| **baseline** (shipping, `gemini-3.1-flash-lite`) | $0.00146 ¹ | 2/3 | 1/3 | 2/3 | not tested by this study ² |
| E1 gap-listening (marginal add-on) | +$0.00028 ³ | n/a (not targeted) | n/a | n/a | 1/9 synthetic; 2/3 real anchors |
| E2 few-shot intensity | $0.001555 | 2/3 (unchanged) | 1/3 (unchanged — 0 wins) | n/a | n/a |
| E3 devil's-advocate | $0.002145 | 1/3 (net −1) | 1/3 (composition changed, count unchanged) | n/a (not scored) | n/a |
| E4 Gemini Flash | $0.00397 | 3/3 (net +1) | 0/3 (net −1) | 1/3 majority-correct (net −1) | n/a |
| E5 Deepgram + dimensional bonus | $0.0063 ⁴ | 0/3 (bonus arm) | 3/3 raw / 2 formal wins, 1 confounded (bonus arm) | 1/3 (0 wins, 1 regression) | n/a |
| **combined** (baseline + E1) | $0.001431 | 2/3 (unchanged) | 1/3 (unchanged) | 2/3 (unchanged) | 1/3 on call_003 this session (3/3 in original standalone measurement) |

¹ Bake-off headline (`out/bakeoff.md`). This study's own `exp0_baseline`
measurement, using the same 3.964018-audio-min denominator used for every
other row, works out to $0.0011525/audio-min — lower, a pre-existing
discrepancy between two separate historical measurements of the same
shipping arm, not something this study changed (flagged in
`study-task-6-report.md`). Source: `out/experiments/exp0_baseline_run{1,2,3}.json`,
`out/bakeoff.md`.
² The shipping pipeline's own AED detector scores 11% presence on the same
9 synthetic clips (`README.md`, `out/validation_report.md`) — a different
mechanism (classical audio classifier, not Gemini), cited for context only,
not measured by an exp0 run in this study.
³ Isolated from `combined_run{1,2,3}.json`'s per-clip `gap_listening.cost_usd`
(mean $0.001108/run across the 3 anchors ÷ 3.964018 audio-min). E1's own
standalone 12-clip harness cost $0.0037–0.0038/run for a different, larger
clip set (3 anchors + 9 synthetic beds) and is not directly comparable to
this marginal figure — see the E1 section.
⁴ Deepgram bills a flat rate per audio-minute (not token-metered), so no
derivation is needed: $0.0043 base + $0.0020 diarization add-on, both
sourced in the E5 section.

**Note on audio billing rate:** `gemini_tone.py`'s own docstring assumes
Gemini bills audio input at **32 tok/s**. A reviewer's independent
reconstruction of E2's few-shot exemplar token delta (Study Task 4)
measured actual audio billing closer to **~25 tok/s** — the nominal
assumption used in cost documentation elsewhere in this repo is
conservative relative to what was actually measured on a live exemplar
delta. This does not change any dollar figure logged above (every cost in
this document comes from real, metered `usage_metadata`, not the nominal
rate), but it means napkin-math cost projections based on the 32 tok/s
docstring figure will run slightly high. (Source: `study-task-4-report.md`,
`progress.md` Study Task 4 entry.)

## Recommendation

**Fits inside the current $0.003/audio-min ceiling, recommended today:**
baseline + E1 (the combined stack), at $0.00143/audio-min — under half the
ceiling. It never makes anything worse and adds a real, if inconsistent,
noise-presence signal on one call class. **Condition:** harden E1 with a
reliability layer (e.g., ask the gap-listening question 3× and take a
majority vote) before leaning on it for anything client-facing — its
single-call reliability, measured twice now, ranges from 1/3 to 3/3.

**Technically under-ceiling but not recommended — explicit negative
results, stated as savings:** E2 few-shot intensity ($0.001555/audio-min,
zero measured effect) and E3 devil's-advocate ($0.002145/audio-min, net
negative on the field it targets). Both cost more than baseline for no net
accuracy gain; do not spend here.

**Needs ~$0.004–0.007/audio-min and is also not recommended — the
highest-cost negative results:** E4 Gemini Flash ($0.00397/audio-min, 2.72×
baseline, 32% over the ceiling, net negative on 3 of 9 scored cells) and E5
Deepgram diarization ($0.0063/audio-min effective, 0 wins/1 regression on
its target field). Neither a bigger model nor a second vendor bought back
accuracy at this sample size — both cost meaningfully more and left the
anchors no better off, in E4's case measurably worse on two fields it
wasn't even trying to fix.

**The credible next-dollar ask.** At this scale, model-side budget does
*not* buy accuracy: every lever that spent more on a bigger model or a
second model pass (E3, E4) net-regressed against the shipping baseline. The
one lever that netted a real win (E1) is also the cheapest one tested, and
its win needs a reliability harness before it's production-grade. The
highest-leverage place to spend a bigger budget is not a bigger model — it
is **more labeled anchor calls** (this entire study runs on 3) plus more
targeted-question levers built in E1's style (narrow, cheap, focused
sub-questions rather than bigger general-purpose passes), each validated
the same rigorous way this study was: measured, repeated, and reported
honestly including the losses.

## Limitations

- **n=3 labeled anchor calls.** Every accuracy figure above is directional
  evidence for piloting on more data, not a statistically powered result —
  a single clip flipping changes any given lever's measured accuracy by 33
  points. The 9 synthetic noise clips give E1 a larger sample for presence
  detection specifically, but they are synthetic SNR-mixed beds, not real
  recordings.
- **Synthetic noise beds may be adversarial to both AED and Gemini.** 8 of 9
  synthetic clips were denied by gap-listening even at "high" nominal
  severity; the shipping AED detector separately scores 11% presence on the
  same 9 clips (`README.md`). Both detectors work on the 2 real noisy
  anchors (`call_002`, `call_003`). This may be a real limitation of
  synthetically-mixed audio rather than evidence either detector is broken
  on real calls.
- **Session-to-session variance is real and material, not just single-run
  noise.** E1's `call_003.ogg` noise-presence win measured 3/3 in its
  standalone Study Task 3 run and 1/3 when the identical function, prompt,
  and audio were re-run three more times inside the Study Task 8 combined
  session — at the same `gemini_temperature=0.1` used everywhere else in
  this study, where most other levers reproduced perfectly deterministically
  across runs. This is the study's clearest evidence that "3 runs in one
  session" is not the same guarantee as "reliable across sessions."
- **E5's overlap-detection thresholds (0.5s minimum cross-speaker
  intersection, back-channel exclusion rules) are first-pass choices**,
  documented as such in the design, not tuned against this eval data — and
  the study found the target case fails upstream of where those thresholds
  would even apply (diarization never produces a second speaker on 2 of 3
  anchors).
- **`call_002.ogg`'s tone label may itself be a case of label noise, not
  purely a model failure.** Across 4 independent prompt-wording iterations
  (recorded in `docs/decisions.md`), the model consistently reads a single
  Spanish profanity phrase as decisive frustration evidence while its own
  rationale acknowledges the rest of the call is calm — this may be a
  genuine model-prior limitation, or the human label itself may be
  debatable; this study did not attempt to resolve which.
- **None of the five experiment modules' external-API call paths
  (`ask_gemini_gaps`, `classify_with_exemplars`, `advocate_pass`,
  `classify_flash`, `deepgram_words`) carry retry logic**, unlike the
  shipping pipeline's own Gemini call. No transient failure occurred during
  any live run in this study, so this gap was never exercised — a real
  transient failure on any of these paths remains untested.
- **Leakage-safe by design, not just by claim:** E2's exemplars are strict
  leave-one-out; no threshold in the shipping pipeline or any experiment
  module was tuned against this study's eval data at any point.
- **Spend:** $0.1580927640625 of the $10.00 cap used across all 7 stages
  (baseline + E1–E5 + combined) — nowhere near the $7.00 warn threshold,
  let alone the cap.
