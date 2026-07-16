# Decision log

Chronological record of every calibration/threshold decision made building the
AutoAce audio pipeline: pre-build seeds (made before any code existed), then
build-time findings (Tasks 5–9, as they were discovered while implementing each
analyzer), then the Task 11 evaluation-harness verdicts (which re-examined five
of the build-time calibrations against an augmented validation set and a live
tone bake-off). Evidence sources: `.superpowers/sdd/task-N-report.md` for
N in 5..11, `.superpowers/sdd/progress.md` (ledger), `out/bakeoff.md`,
`out/validation_report.md`.

---

## Pre-build decisions (seeded 2026-07-16, before implementation began)

- **long_silence threshold 10s**: AutoAce labeled a 7.4s dead-air gap `false`
  (call_003). 10.0s was chosen to sit comfortably above that real labeled-false
  gap.
- **Format sniffing via ffprobe, never extension**: the production smoke set
  (`data/test_recordings/`) contains `.mp3`-named files that are PCM WAVs
  inside — decoding must detect the real container/codec, not trust the
  filename.
- **Tone primary = gemini-3.1-flash-lite (paid tier, disclosed)**: the only
  promptable audio API verified under the $0.003/min ceiling (verified
  2026-07-16); multilingual (call_002 is Spanish); promptable to target the
  *customer's* tone specifically, not the AI agent's (Erica).
- **No audio few-shot in the Gemini prompt**: 3 example calls would add ~4
  audio-minutes of input tokens per request (~3x cost) — label definitions +
  DSP hints only, no few-shot anchors.
- **Overlap = Gemini audio judgment**: pyannote pretrained overlapped-speech
  detection refuted/unavailable on current pyannote 3.x (research 0-3 —
  claim didn't hold up); energy heuristics on mono mixes are unreliable.
  Disclosed as the weakest field in the memo.

---

## Build-time findings (Tasks 5–9)

### Task 5 — VAD (silero) tightening: the 8.9s/10.0s silence margin

Measured directly against the pre-build seed above: silero's speech timeline
on call_003 puts its largest real non-speech gap at **8.90s** — 1.1s *under*
the seeded 10.0s `long_silence_s` threshold, and the call is correctly labeled
`long_silence_present: false`. This is the exact real-world case the seed
decision was calibrated against; Task 5 confirmed the margin holds with silero
in the loop, not just in the abstract. `long_silence_s` was kept at 10.0
unchanged. (Task 11 later re-confirmed no regression: none of 18 augmented/
validation clips ever spuriously trip `long_silence_present`.)

### Task 6 — Noise presence: AED redesign from gap-only to windowed full-clip

**Original architecture (per spec §5.3, "AudioSet tagger on concatenated
non-speech segments") was implemented first, then proven unfixable by
threshold alone:**

- On the 3 labeled anchors, call_001 (no-noise label)'s noise floor under the
  gap-only design was `Animal` at mean prob **0.113**; call_002 (TV, medium
  label)'s true signal in that same gap-only evidence source topped out at
  `Music` **0.040** — call_002's true-signal ceiling sits *below* call_001's
  noise floor. No single global `aed_prob_threshold` can flag call_002 present
  without also flagging call_001 present; this is a mathematical
  impossibility under concatenated-non-speech-gap evidence, not a tuning gap.
- Root cause, found by comparing evidence sources: the true signal is far
  stronger when speech is present. call_002's `Radio` scores **0.489** in
  speech-concurrent audio vs. **0.040** (`Music`) in its non-speech gaps alone
  — the TV bleeds through continuously *under* the customer's speech, so a
  non-speech-only design structurally starves the classifier of its best
  evidence.
- Naive per-segment (non-concatenated) pooling was tried as an alternative and
  rejected: it recovers call_002's signal (one clean 0.70s gap alone scores
  `Music` 0.310) but spikes false positives on short/out-of-distribution
  fragments — call_003's per-segment max-pool surfaced `Clip-clop 0.413`,
  `Horse 0.390` from sub-1.5s fragments, a known CNN behavior on short
  out-of-distribution inputs.

**Decision (controller-directed rework):** replace concatenated-non-speech-gap
AED with **windowed full-clip AED** (`aed_window_s=5.0`, `aed_hop_s=2.5`, 50%
overlap) **+ sustained-support pooling** (`aed_min_support_s=5.0`, i.e. ≥2
independently-spaced activated windows), calibrated against the 3 anchors with
an explicit floor (`aed_prob_threshold` not permitted below 0.20).

- **Threshold sweep** (cached per-window probabilities, swept cheaply): at every
  threshold from 0.35 down to the 0.20 floor, call_001 never sustains anything
  (support stays 0.0s) — the windowed architecture gives a wide, robust margin
  against false positives that the old architecture could never offer. call_002
  already sustains `Radio` comfortably at the original 0.35 (4/13 windows clear
  it, 10.0s support vs. 5.0s floor) — so `aed_prob_threshold` was **kept at
  0.35**, not lowered (the controller's rule to lower it was conditional on
  0.35 missing call_002; it doesn't).
- **`MASKED_CLASSES` extended** with `Sidetone` and `Busy signal` — both are
  telephony call-channel artifacts (the caller's own voice echoed back;
  call-progress tones), the same category as the already-masked
  `Telephone`/`Dial tone`/DTMF entries, and were outranking the true noise
  signal in call_002/call_003's gaps before being masked (call_003's `Busy
  signal` was at 0.320, dangerously close to threshold, for the wrong reason).
- **call_003 ("sharp static") stays a documented `xfail`, not a passing
  assertion.** `present=True` technically holds (2 of 68 windows clear 0.35,
  the thinnest possible margin), but the sustained class is always `Radio`,
  never `Static`/`White noise`/`Pink noise`, at any threshold tested down to
  the floor. CNN14 appears to have no static/crackle response for this noise
  family — its confusions are impulsive/transient classes (`Firecracker`,
  `Machine gun`) that resemble static crackle in mel-spectrogram features.
  `test_call_003_static_detected` asserts the meaningful outcome
  (`present and type_label == "static"`), which fails honestly and is marked
  `xfail(strict=False)`, with the pipeline's audio-LLM noise opinion in fusion
  as the disclosed backstop for this case.
- **Two windowing bugs found by a reviewer, fixed with residual-ownership
  weighting**: (1) the tail-anchor window (added so trailing audio is never
  left unscored) could sit much closer than `hop_s` to its predecessor — a
  172s clip's last two windows were only 2.0s apart (60% overlap) yet each was
  credited a flat 2.5s support, letting one spike double-count as two
  independent detections; (2) the fixed 5.0s support floor was unreachable on
  clips shorter than one window (support capped at the clip's own length).
  Fixed via `_support_weights`: each window after the first owns only
  genuinely-new time (`min(hop_s, start_i - start_{i-1})`), and
  `effective_floor = min(aed_min_support_s, sum(weights))` scales the floor
  down on short clips instead of demanding support a clip can never provide.
  Verified against real call_003 output: its two activated `Radio` windows
  (37.5s and 95.0s, 57.5s apart) were never near each other — its
  `present=True` rested on genuinely independent evidence, not the bug. 10 new
  pure-logic unit tests added; one of them caught a real regression (a
  `zip(..., strict=True)` call that could never succeed on unequal-length
  iterables) before it ever reached a real call.
- **call_002's severity reads `high`** (SNR 0.28dB) **vs. the label `medium`**
  — a one-band discrepancy in the independent, untouched SNR-based severity
  math, deferred to the eval task rather than hand-tuned on a single anchor.
  (Task 11 later confirmed this is the same pre-existing labeler-vs-metric
  disagreement, not new evidence of a threshold bug — see below.)

### Task 7 — Quality: SQUIM demoted to noise-conditioned backstop, rolloff retuned, memory-driven window cut

**SQUIM PESQ conflates ambient noise with channel quality — proven, not
assumed.** Measured on the 3 real, human-labeled-`clear` anchor calls (original
60s scoring window): PESQ 2.11 (call_001) / 1.64 (call_002) / 2.09 (call_003)
— all well under the spec's naive 3.0 "clear" floor, with call_002 (the
noisiest-but-technically-*undistorted* call) scoring the *worst* of the three.
Cross-checked against the completely independent, non-ML SNR calculation from
`noise.py` (pure RMS-ratio math, zero shared code with SQUIM): SNR 23.12dB /
0.28dB / 10.65dB for calls 001/002/003 — the SNR ranking (001 > 003 > 002)
**exactly matches** the PESQ ranking and the SI-SDR ranking. A perceptual-
quality/intelligibility metric tracking ambient SNR rank-for-rank, on a field
the client's own protocol defines as *technical-channel-only, independent of
background noise*, is decisive evidence SQUIM is measuring the wrong axis for
this field — not a bug, a domain mismatch.

- **Decision:** demote SQUIM from primary evidence to a **noise-conditioned
  severe backstop** — it may only escalate a call to `severely_impaired` when
  `pesq < pesq_severe_backstop (1.3)` **AND** `snr_db > snr_no_excuse_db
  (15.0)`, i.e. only when the background is clean enough that noise cannot
  explain a catastrophic PESQ. Primary evidence becomes 4 deterministic,
  channel-only signals computed straight from the waveform + VAD timeline:
  clipping ratio (`clipdetect`, hard override to `severely_impaired`
  regardless of everything else), dropout runs (hard near-zero runs strictly
  inside VAD speech segments), spectral rolloff/bandwidth (muffle detector),
  and speech-segment RMS (low-volume detector). Worst-triggered level wins;
  default `clear`.
- **Spectral rolloff retuned with measured anchor evidence.** Real telephony
  speech on the 3 labeled-clear anchors measured rolloff **1248Hz** (call_001),
  **1024Hz** (call_002), **1591Hz** (call_003) — nowhere near a naive
  "wideband speech, well above 2.2kHz" assumption. The original
  `rolloff_severe_hz`/`rolloff_slight_hz` values (2200/1200Hz) would have
  false-triggered the muffle detector on every real anchor. Retuned to
  **900/600Hz**, below all three measured anchor values with real margin.
- **SQUIM memory scales superlinearly with input length — measured, not
  estimated:** 0.44GB @ 5s, 1.5GB @ 15s, 4.6GB @ 30s, ~**14GB @ 60s**. The
  original 60s scoring window swamped a 16GB machine into a swap storm (13.76GB
  resident + 5.2GB swap, observed directly), which retroactively explains an
  earlier office-Mac session's day-long apparent "slow model loading" — it was
  actually a memory blowup, not a slow model. **Fix:** scoring window cut from
  the middle 60s to the middle **15s** (~1.5GB peak). Since PESQ only feeds the
  `<1.3` backstop gate, a shorter representative window is safe by margin.
  15s-window anchor PESQ values (superseding the 60s-window numbers above for
  any future comparison): call_001 1.95, call_002 2.18, call_003 2.34 — all
  three rate `clear` (matching their labels), backstop correctly gated off on
  every anchor. Full per-call evidence: rolloff 1248/1024/1591Hz, dropouts
  0.0/min on all three, clipping ratios ~1e-6 (4-5 orders of magnitude below
  the 0.02 override).
- **Known tradeoff, checked and found not to matter (Task 11):** a 15s
  positional (not VAD-aware) window is blind to channel damage entirely
  outside it on long clips. Confirmed no observed miss: `aug_clip.wav`
  (clipping) and both dropout augmentation clips (185.6s each) all scored
  correctly because clipping/dropout detection scans the *full* clip via VAD,
  never a windowed subset — SQUIM's window choice only affects its own gated
  backstop path, which never needed to fire for either augmentation type.

### Task 8 — Tone arms: transformers pin, Gemini prompt iteration, dimensional's no-diarization limitation

- **`transformers` pin — a real, silent-corruption bug, not a version nit.**
  The task 1–7 pin (`>=4.44`) resolves today to `5.14.1`, which crashes the
  dimensional arm's model load (`AttributeError: 'EmotionModel' object has no
  attribute 'all_tied_weights_keys'` — transformers 5.x rewrote model-loading
  internals incompatibly with the audeering custom `Wav2Vec2PreTrainedModel`
  subclass). Downgrading to the original floor, `4.44.2`, avoids the crash but
  is *worse*: it loads "successfully" while **silently** leaving the Wav2Vec2
  encoder's `pos_conv_embed` weight-norm parameters randomly initialized
  instead of restored from the checkpoint — no exception, no warning surfaced
  to the caller, and every dimensional-arm prediction would have been silently
  wrong. Repinned to `transformers>=4.46,<5` (tested against `4.57.6`: no
  crash, no missing-weights warning). `transformers` is used nowhere else in
  the codebase, so the pin is fully localized.
- **Gemini prompt iteration for call_002 ("frustrated" vs. label "neutral") —
  4 rounds, capped, genuine model-prior limitation.** Baseline `build_prompt`
  missed all 3 anchor labels. Iterating the surrounding "Rules:" text (never
  the enum definitions themselves) across 4 rounds fixed call_003 (round 2)
  and call_001 (round 3), but call_002 never moved: the model's own rationale
  each time acknowledges "the rest of the call is calm/low-intensity" yet
  still scores `frustrated`, anchored entirely on a single Spanish
  slang/profanity phrase ("mamá huevo," used once, briefly) despite explicit,
  increasingly forceful instructions to require independent sustained
  escalation evidence before letting one crude word move the score. Stopped
  at the agreed 4-iteration cap. Controller adjudication: accept as a known
  model-prior limitation / possible label-noise case, do not iterate
  `build_prompt` further, mark `xfail(strict=False)`. No audio few-shot
  examples derived from the 3 test-fixture calls were added (would be
  circular — guarantees a pass without generalizing). (Task 11's bake-off
  later reconfirmed this is gemini's *only* miss on the 3-call sample, with no
  prompt change — see the Task 11 section below.)
- **Dimensional arm's no-diarization limitation — real, reproducible, not a
  code bug.** The weak valence-ordering sanity check
  (`call_001.valence < call_003.valence`, i.e. the upset call should read
  lower valence than the satisfied call) fails deterministically on real data:
  call_001 (labeled upset) measures valence **0.6694**, call_003 (labeled
  satisfied) measures **0.5790** — backwards. `classify()`/`_avd()` are
  byte-identical to the brief's code, including feeding the model *all*
  speech in a clip (agent + customer mixed, no diarization) exactly as the
  arm's own documented limitation states. call_003 is a 172s call dominated
  by long, calm agent turns; mixing that much agent speech into the "speech"
  average plausibly dilutes the customer-specific valence signal, while
  call_001's brief, agitated exchange isn't diluted the same way. No `va_*`
  threshold was touched to force a pass (the test compares raw valence
  directly, before `map_va()` even runs). Controller adjudication: accept as
  the arm's documented limitation, `xfail(strict=False)`.

### Task 9 — Fusion rules: thin-margin type override (zero live triggers, kept)

Two cross-field rules for `background_noise_type` when AED and the Gemini
audio-LLM's own noise opinion disagree:

- **Rule A** (brief-specified): AED reports absent but the LLM opinion reports
  present → fill in noise from the LLM's opinion (through `concise_label()`,
  fixed in review round 1 so a raw AudioSet class name like `"Television"`
  never leaks past fusion unformatted as `"TV"` would).
- **Rule B** (new, `support_s`/`support_floor_s`-driven): when AED's presence
  margin is thin (`support_s - support_floor_s <= aed_hop_s`) **and** the LLM
  agrees noise is present but names a *different* type, defer to the LLM's
  type label. Built specifically around call_003 (AED's `'radio'` vs. the
  label "sharp static," AED margin exactly at the minimum 2-window floor —
  see Task 6 above).

**Rule B has had zero live trigger opportunities.** call_003's own real
Gemini call denies `background_noise_present` outright
(`{"present": false, "type": ""}`), so `llm_present` is `False` and the rule
correctly abstains — it requires the LLM to *agree presence, disagree type*,
not deny presence altogether. Task 11 later ran this same check across 9
synthetic noise-augmented clips (TV/static/hum, all 3 severity tiers,
including the loudest tested case) and found Gemini denies presence on **all
9** — reproducing call_003's behavior as a genuine, reproducible blind spot
for continuous ambient noise, not a one-off. Rule B therefore never fires
across the 3 real anchors or the 9 augmented clips. **Verdict: keep the rule
unchanged** — it is unit-tested directly in both directions with synthetic
inputs (`tests/unit/test_fusion.py`), cheap, narrowly-gated, and could
plausibly help on some other real call; it has simply had no opportunity to
in this sample. `background_noise_type` should be disclosed in the client
memo as AED's best-effort guess, cross-checked by the LLM only when the LLM
independently agrees noise is present (which it has not, for any
continuous/ambient noise case observed so far).

---

## Task 11 — Evaluation harness + calibration verdicts

Seeded by Task 11 (evaluation harness + tone bake-off), per the project
ledger's deferred-calibration list. This section re-examines five build-time
calibrations above against a fresh augmented validation set and a live tone
bake-off.

Evidence sources: `out/validation_report.md` (18-clip augmented validation set,
built by `eval/build_validation_set.py`, run through the real pipeline with the
`gemini` tone arm) and `out/bakeoff.md` (3-call tone bake-off, arms
gemini/dimensional/transcript). Raw per-clip predictions used below are from
`out/validation/results.json`.

### 1. Noise severity bands (`snr_none_db=20`, `snr_low_db=15`, `snr_medium_db=5`)

**Verdict: KEEP unchanged. No systematic evidence found; one pre-existing
disagreement documented, unchanged.**

- The augmented validation set adds 9 noise clips (TV/static/electrical-hum beds
  mixed into the clean call_001 anchor at nominal target SNRs 18/10/2 dB, one bed
  per severity tier).
- **A real construction bug was found and fixed** while running this for real:
  the brief's `_mix_at_snr` referenced the WHOLE clip's RMS as the signal-power
  anchor. call_001 is only 45% speech by duration with near-silent gaps, so its
  whole-clip RMS (0.166) sits well below its speech-only RMS (0.246) — using the
  diluted figure under-gained the injected noise relative to what
  `analyze_noise.snr_db` (speech RMS vs gap RMS, the same function the pipeline
  itself uses) measures. Measured before the fix: nominal (18, 10, 2) dB ->
  measured (21.3, 16.5, 10.6) dB for the TV bed alone. Fixed by referencing the
  clean signal's own VAD speech-segment RMS instead (see
  `eval/build_validation_set.py::_mix_at_snr`).
- **Even after the fix, the augmented harness could not exercise the severity
  bands as evidence**, for a reason independent of the SNR thresholds: AED
  (PANNs CNN14) fails to recognize these noise beds as ANY AudioSet class with
  enough confidence/sustained support to clear `aed_prob_threshold=0.35` +
  `aed_min_support_s=5.0` in 8 of 9 clips. Top classes were consistently
  unrelated ("Animal", "Music", "Vehicle", "Bird" — never "Television",
  "Static", or "Hum"), with duration-weighted mean probabilities of 0.03-0.18,
  far below threshold. Since `severity_from_snr` only runs when `present=True`,
  the severity bands themselves were never reached for 8/9 clips — the miss is
  an AED-vs-synthetic-audio recognition gap, not a threshold-tuning case.
  Real per-clip evidence: `background_noise_present` accuracy 11% (1/9),
  `background_noise_severity` accuracy 0% (9/9) on the augmented noise rows —
  but the failure mode is "AED never fires," not "AED fires with the wrong
  band."
- The ONE clip that did cross the presence threshold (`aug_static_snr2.wav`,
  nominal target "high") measured 13.91 dB via `analyze_noise.snr_db` — a
  MEDIUM-band value, not high — confirming that nominal-target-based truth
  labels do not reliably correspond to what the detector measures once mixed,
  even post-fix (mixing noise at a fixed gain doesn't hold a stable dB gap once
  the noise itself materially changes the RMS during speech). This is a harness
  construction limit, not evidence the 5 dB / 15 dB boundaries are wrong.
- Cross-referencing the only REAL anchor with actual field noise: call_002
  measures 0.28 dB SNR (task 7/9 finding, reconfirmed here via
  `call_002_seg0.wav` -> `present=True, severity=high`) against a client label of
  "medium" — a pre-existing, already-documented labeler-vs-metric definitional
  disagreement (task 7), not new evidence of a threshold bug. No second real
  anchor disagreement surfaced.
- **Grouped-by-source-call check (no leakage):** the only source clip for all 9
  noise augmentations is call_001 (the no-noise anchor); the only other noise
  evidence is call_002/call_003 (already used to calibrate these bands in tasks
  6/7). No cross-fold information was used to make this decision.
- Per the amendment's explicit rule ("if the augmented truth shows the bands
  systematically off, retune WITH evidence; if not, keep and document the
  disagreement") — there is no systematic band-level disagreement to retune
  against, only a detection-gate limitation. Lowering `aed_prob_threshold`
  toward its floor (0.20) to force these synthetic clips to register was
  considered and rejected: the real anchor (call_002) does not miss at 0.35 (see
  config.py's existing rationale), so lowering it would trade real-world
  precision for a synthetic-test artifact, which is exactly the untuning this
  task was told not to do.

**Follow-up recorded for whoever revisits this next:** validating AED-gated
severity bands needs REAL noise recordings (or at minimum audio that PANNs'
AudioSet-trained classifier actually recognizes), not synthetic
white-noise/two-tone-hum beds — the current augmented set can validate the SNR
*measurement* pipeline (now fixed) but not the AED *presence* gate + severity
band combination end-to-end.

### 2. call_003 noise type ("radio" via AED vs client label "sharp static") and fusion Rule B's thin-margin override

**Verdict: KEEP Rule B unchanged; disclose `background_noise_type` as
approximate in the client memo.**

- Direct evidence gathered this task: Gemini denies `background_noise_present`
  on ALL 9 synthetic noise-augmented clips (TV/static/hum, all 3 severity
  tiers), including the loudest tested case (nominal target SNR=2 dB). This
  reproduces call_003's known real-call behavior (task 9: "Gemini denies noise
  presence outright for call_003") across a 9-clip battery, not just the one
  historical case — a genuine, reproducible blind spot for continuous ambient
  noise (static/hum/TV-bleed), independent of loudness in the range tested.
- Because Gemini never reports presence in any of these clips, fusion's Rule A
  (accept the LLM's presence opinion when AED denies it) never fires, and Rule
  B's thin-margin type-override condition (`present AND llm_present AND
  llm_type != aed_type`) requires `llm_present=True`, which also never occurs
  here. Rule B therefore has zero live trigger cases across the 3 original
  anchors AND all 9 augmented noise clips — consistent with, and reinforcing,
  task 9's finding.
- Rule B is unit-tested directly (synthetic monkeypatched inputs,
  `tests/unit/test_fusion.py`) and is cheap, narrowly-gated, defensive code — no
  evidence it causes harm, only that it has not yet had an opportunity to help
  in this sample.
- **Verdict: keep the rule as-is** (it's correct and someday-useful — a
  different real call could plausibly hit its trigger condition), but the
  client memo should state plainly that `background_noise_type` is AED's raw
  AudioSet-label guess, cross-checked by the audio-LLM only when the LLM
  actually agrees noise is present (which it has not, on any static/hum-like
  case observed so far) — i.e., disclose `type` as approximate/best-effort, not
  independently verified, for continuous/ambient noise categories.

### 3. Long-silence margin (`long_silence_s=10.0`, call_003's labeled-false gap measures 8.9s)

**Verdict: no action. Confirmed no regression.**

- No augmented clip (18 clips, including two ~185.6s looped dropout clips)
  triggers `long_silence_present=True`; all measure `False`, matching truth (none
  of the augmentations are designed to test this field, so no clip should ever
  trip it). This confirms the existing 10.0s threshold's ~1.1s margin above
  call_003's real 8.9s labeled-false gap is undisturbed by anything added this
  task.
- No new evidence bearing directly on the exact margin was generated (the
  augmented set doesn't construct a gap near the 8.9s/10.0s boundary); this
  remains an open item for a future task if a synthetic near-boundary gap clip
  is ever built.

### 4. SQUIM 15s-window tradeoff vs full-clip deterministic detectors

**Verdict: no action. Confirmed deterministic detectors are unaffected by clip
position/length.**

- `aug_clip.wav` (30.9s, clipping-degraded) -> correctly rated
  `severely_impaired` (clipping override fires; clipping ratio is measured over
  the FULL clip, not a 15s window).
- `aug_dropout_slight.wav` / `aug_dropout_severe.wav` (185.6s each, looped
  source + inserted dropouts) -> correctly rated `slightly_impaired` /
  `severely_impaired` respectively. The dropout detector scans the entire VAD
  speech timeline across the full 185.6s clip, not a windowed subset, and
  correctly counted the inserted dropouts regardless of where in the (much
  longer than 15s) clip they landed.
- Both deterministic-detector-driven quality augmentations scored 100% (3/3)
  against truth in `out/validation_report.md`'s `audio_quality` field, with no
  clip long enough to displace the damage entirely outside SQUIM's middle-15s
  window causing a miss (clipping/dropouts are full-clip signals by
  construction, per quality.py's architecture — SQUIM's narrow window only
  affects its own gated backstop-escalation path, which never needed to fire
  for either augmentation). No missed quality damage observed; SQUIM's window
  choice is unaffected by this finding.

### 5. Gemini call_002 "frustrated" vs label "neutral" (adjudicated known miss)

**Verdict: quantified via bake-off; no prompt changes (per this task's scope).**

Tone bake-off (3 real labeled calls, `--arms gemini dimensional transcript`,
`out/bakeoff.md`):

| arm | tone acc | macro F1 | $ / audio-min | proc s / audio-min |
|---|---|---|---|---|
| gemini | 67% | 0.667 | $0.00146 | 7.3/min |
| dimensional | 0% | 0.000 | $0 (local) | 6.0/min |
| transcript | 0% | 0.000 | $0.00065 (whisper local $0 + OpenAI text, metered) | 12.1/min |

- gemini's one miss (of 3) is exactly this case: `call_002.ogg` predicted
  `frustrated`, true `neutral`. This is the SAME adjudicated miss from task 9 —
  no new information changes that adjudication, and no prompt change was made
  (explicitly out of scope for this task).
- gemini is unambiguously the bake-off winner on this sample (67% vs 0%/0%),
  reinforcing it as the shipping tone arm. n=3 is a very small sample (each
  arm's macro F1 swings hard on a single clip) — this is a directional signal,
  not a statistically powered comparison; the memo should present it as such.
- `dimensional` and `transcript` both score 0/3 on this tiny sample (every
  single prediction wrong, not merely a soft near-miss on all three) — real
  measured results, not a code bug (raw predictions are in `out/bakeoff.md`'s
  JSON block).
- **Note on run-to-run variance:** the table above is the final, code-fixed
  regeneration (`out/bakeoff.md` on disk); an earlier run (recorded in
  `progress.md`'s Task 11 ledger entry, before a bake-off scoring-bug fix
  described in `task-11-report.md`) measured gemini at 9.6 proc-s/audio-min,
  dimensional at 12.5, and transcript at $0.00081/16.7 proc-s/audio-min for the
  same 3 calls with the same accuracy/F1/predictions. The accuracy figures,
  predicted labels, and $/audio-min for gemini are identical between runs; the
  proc-time deltas are normal live-API-latency/local-model wall-clock wobble
  across repeated runs, not a code or calibration change — call out both
  figures here rather than silently picking one, per the project's disclosure
  norm.
