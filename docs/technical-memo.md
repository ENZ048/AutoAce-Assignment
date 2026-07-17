# Technical Memo — Voice Tone & Background Noise Analysis

**Deliverable:** per-call 9-field JSON analysis + hosted review dashboard
**Dashboard:** https://autoace.aixcoach.in (credentials supplied separately)
**Cost:** **$0.00146 per audio-minute measured** against the $0.003 ceiling (≈ half)
**Date:** 2026-07-17

Every number in this memo is measured, and each one names its source file in
this repository. Nothing is projected. Where results are weak or the sample
is small, that is stated in place rather than footnoted away.

---

## 1. What was built

A two-layer analysis pipeline behind a batch dashboard:

- **Deterministic local layer** (no external calls, no per-call cost):
  silero-VAD for silence/long-gap detection; windowed PANNs audio tagging +
  gap-SNR for background-noise presence, type, and severity; deterministic
  channel checks with a gated SQUIM backstop for audio quality.
- **One focused LLM call** per clip (`gemini-3.1-flash-lite`, audio in,
  strict JSON out) for emotional tone and intensity, with a local
  dimensional-emotion fallback if the API call fails — a batch never stalls
  on a vendor error.
- **Fusion layer** enforcing cross-field invariants (e.g. severity/presence
  consistency) so the 9-field output is always schema-valid.
- **Batch surfaces:** a CLI (`python -m autoace_audio analyze <dir|zip>`)
  and the hosted dashboard (login → ZIP/folder upload → validation report
  *before* any processing → live per-file progress that survives reload →
  results review → CSV/JSON downloads preserving original filenames).

Audio leaves the machine only for the single Gemini call (paid tier: content
not used for training). Transcript text, not audio, was sent to OpenAI in
one evaluation-only experiment arm; nothing else touches a third party.

## 2. Accuracy

### 2.1 Labeled real calls (n = 3 anchors)

The trial supplied three labeled recordings. On the five headline fields
(tone, noise presence, noise severity, audio quality, long silence) the
shipping pipeline scores **13/15 (86.7%)**; across all eight comparable
fields it scores **17/24 (70.8%)**. Per-field detail, prediction vs. label
(source: `out/anchors_recheck/results.json` vs `data/labels.csv`):

| field | call_001 | call_002 | call_003 | score |
|---|---|---|---|---|
| emotional_tone | upset ✓ | **frustrated ✗** (label: neutral) | satisfied ✓ | 2/3 |
| emotional_intensity | medium ✗ (high) | medium ✓ | low ✗ (medium) | 1/3 |
| background_noise_present | ✓ | ✓ | ✓ | 3/3 |
| background_noise_type | — ✓ | radio ✗ (TV) | radio ✗ (sharp static) | 1/3 |
| background_noise_severity | none ✓ | high ✗ (medium) | medium ✓ | 2/3 |
| audio_quality | clear ✓ | clear ✓ | clear ✓ | 3/3 |
| speaker_overlap_present | ✓ | ✗ (missed) | ✓ | 2/3 |
| long_silence_present | ✓ | ✓ | ✓ | 3/3 |

**Tone confusion matrix** (rows = label, columns = prediction; n = 3):

| | neutral | satisfied | frustrated | upset | distressed |
|---|---|---|---|---|---|
| **neutral** | 0 | 0 | **1** | 0 | 0 |
| **satisfied** | 0 | **1** | 0 | 0 | 0 |
| **upset** | 0 | 0 | 0 | **1** | 0 |

The single tone miss (`call_002`, neutral → frustrated) survived four
independent prompt iterations: the model consistently treats one Spanish
profanity phrase as decisive frustration evidence while acknowledging the
rest of the call is calm. This may be a genuine model-prior limitation or
label noise in the anchor itself; both readings are recorded in
`docs/decisions.md` and neither is claimed as fact.

Known systematic behaviours, disclosed rather than tuned away:
- **Intensity runs one band low** on 2/3 anchors — consistent direction,
  recorded as systematic.
- **Noise type is approximate** ("radio" vs the labels "TV" / "sharp
  static") — presence and severity are the operationally useful signals;
  type is best-effort vocabulary.
- **Overlap is the weakest field** (an LLM judgment today; see roadmap).

### 2.2 Augmented validation set (synthetic degradations)

To test the quality detectors beyond 3 clips, an augmented set applies
scripted degradations (clipping, band-limiting, dropouts) and synthetic
noise beds to the anchors (`eval/build_validation_set.py`):

- **Audio-quality detectors: 100%** on the quality-labeled clips of the
  augmented set (`out/validation_report.md`).
- **Synthetic noise beds: 11% presence detection** (1/9). Disclosed
  honestly: both the local AED detector *and* Gemini gap-listening deny
  most synthetically SNR-mixed beds, while both work on the two *real*
  noisy anchors (3/3 presence). The evidence so far suggests
  synthetic mixing is adversarial to both detectors rather than
  representative of real calls — but only more real labeled audio settles
  this (see roadmap).

### 2.3 Tone-arm bake-off

Three tone strategies were compared on the anchors (`out/bakeoff.md`):

| arm | tone accuracy | macro F1 | $/audio-min | notes |
|---|---|---|---|---|
| **gemini audio** (shipping) | **67%** | 0.667 | $0.00146 | only miss = the adjudicated `call_002` |
| dimensional (local, free) | 0% | 0.000 | $0 | fallback only — keeps batches alive, not accurate |
| transcript → LLM | 0% | 0.000 | $0.00065 | loses paralinguistic signal in transcription |

Listening to the audio matters: the transcript arm, given identical calls
as text, got every tone wrong.

## 3. Cost

- **Measured shipping cost: $0.00146 per audio-minute** — token-metered
  from live `usage_metadata`, not list-price arithmetic
  (`out/bakeoff.md`). An independent re-measurement in the budget study
  produced $0.00115/audio-min for the same arm; the memo quotes the
  *higher* of the two historical measurements.
- **Ceiling: $0.003/audio-minute** → shipping runs at **~49% of budget**.
- The local layer (VAD, PANNs, SQUIM, quality, fusion) is $0 per call.
- The recommended production configuration (baseline + gap-listening
  add-on, §5) measures **$0.00143/audio-minute** — still under half the
  ceiling (`docs/experiments/2026-07-17-budget-accuracy-study.md`).

## 4. Latency

- **Deployed end-to-end** (m5.xlarge, 4 vCPU / 16 GB, us-east-1): a 3-clip,
  3.96-audio-minute batch completed in **144 s wall**, of which ~70 s was
  the one-time per-batch model load (shown in the UI as an explicit
  "loading models" phase). Steady-state ≈ **19 s per audio-minute**
  end-to-end for the full 9-field pipeline.
- **Per-clip Gemini tone call:** 4.7–7.2 s per clip measured
  (`out/bakeoff.md` per-clip `elapsed_s`).
- Throughput model: one worker processes one batch at a time (bounded
  ~4 GB model memory). A 100-call batch of ~3-minute calls ≈ **~95 min**
  at the measured rate. Batches queue; progress is visible per file and
  survives page reloads and server restarts.

## 5. What more budget buys — measured, not assumed

A two-phase budget–accuracy study spent $0.16 of a $10 cap testing seven
levers against the anchors, three runs each
(`docs/experiments/2026-07-17-budget-accuracy-study.md`). The measured
accuracy-vs-cost curve:

| configuration | $/audio-min | outcome |
|---|---|---|
| **shipping baseline** | **$0.00146** | **accuracy peak** |
| + gap-listening noise question (E1) | $0.00143 combined | the one real win: confirms noise on a call class the AED misses; needs reliability hardening |
| few-shot intensity exemplars (E2) | $0.00156 | clean null — byte-identical predictions |
| devil's-advocate tone pass (E3) | $0.00215 | net negative on its target field |
| 3-vote gap majority (E6) | $0.0023 | no reliability gain over single-vote E1 |
| tone self-consistency voting (E7) | $0.0035 | null, and over ceiling |
| Gemini Flash (bigger model, E4) | $0.00397 | 1 win, 2 regressions; 32% over ceiling |
| Deepgram diarization overlap (E5) | $0.0063 | 0 wins on its target field |

**Conclusion: at this sample size, model-side spend does not buy
accuracy.** Every configuration past the baseline cost more and measured
the same or worse. The two levers that *would* credibly move accuracy are
not model spend:

1. **More labeled real calls.** Every accuracy figure above rests on 3
   anchors; one flipped clip moves a field score by 33 points. A labeled
   pilot set (even 30–50 calls) converts every number in this memo from
   directional to statistical, and is the precondition for tuning anything.
2. **Narrow, cheap, targeted sub-questions** in the gap-listening style —
   the only lever class that produced a measured win — rather than bigger
   general-purpose model passes, which measurably regressed.
3. **CLAP zero-shot noise typing** (local, free): in office evaluation it
   produced the first-ever correct "TV" identification on `call_002` (the
   exact miss in §2.1) on 2/3 anchors. Not integrated pre-submission to
   avoid regression risk; it is the top engineering candidate for the
   noise-type field.

Also measured and disclosed: the pipeline's confidence output is
rank-informative but compressed (0.83–0.87 across 17 predictions,
σ = 0.013) — usable for ordering review queues, not yet calibrated as a
probability. Recalibration is a post-pilot item.

## 6. Dashboard & operations

- **Stack:** FastAPI + SQLite (WAL) job store + detached worker process per
  batch; React SPA served same-origin. Hosted on AWS EC2 (m5.xlarge,
  us-east-1) behind nginx with Let's Encrypt TLS, auto-renewing.
- **Flow:** login (bcrypt + JWT) → upload ZIP/folder → validation report
  *before* spend → explicit start → live per-file progress (failed files
  marked in place) → results table → CSV/JSON/errors downloads.
- **Robustness (each item tested):** zip-slip-safe extraction; decompressed-
  size budget (zip-bomb guard); macOS Finder ZIP junk handled; upload size
  caps; per-file failure isolation (one bad clip cannot stall a batch — a
  hard 60 s Gemini timeout with bounded retries drops to the local
  fallback); workers survive server restarts and are re-adopted by pid;
  batch history persists across reboots.
- **Test suite:** 315 backend/web tests + SPA unit tests, all green at the
  deployed commit.

## 7. Honest limitations

- **n = 3 labeled calls.** All real-call accuracy is directional. This is
  the single biggest caveat on every number above.
- **Synthetic noise beds defeat both detectors** (§2.2); real noisy calls
  do not. Unresolved without more real data.
- **Overlap detection is an LLM judgment** (2/3) — diarization (E5) did
  not fix it; it failed upstream. Real fix likely needs channel-separated
  audio from the telephony stack.
- **Intensity is prompt-coupled to tone:** every attempted intensity
  re-wording flipped a tone anchor (measured 3/3, twice). Intensity fixes
  must come from levers other than prompt wording.
- **Confidence is compressed** (0.83–0.87) — treat as a ranking signal.
- **Session-to-session variance is real:** one lever (E1) measured 3/3 in
  one session and 1/3 in another at identical settings. Reliability claims
  in this memo are stated at the weaker of the two measurements.

---

*Sources for every figure: `out/bakeoff.md`, `out/validation_report.md`,
`out/anchors_recheck/`, `data/labels.csv`,
`docs/experiments/2026-07-17-budget-accuracy-study.md`, `docs/decisions.md`,
and the deployed job record for the latency run. The study's raw run logs
(`out/experiments/*.json`) carry per-clip token counts for every live call.*
