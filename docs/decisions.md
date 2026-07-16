# Calibration decisions

Seeded by Task 11 (evaluation harness + tone bake-off), per the project ledger's
deferred-calibration list. Task 12 completes this file with any further findings.

Evidence sources: `out/validation_report.md` (18-clip augmented validation set,
built by `eval/build_validation_set.py`, run through the real pipeline with the
`gemini` tone arm) and `out/bakeoff.md` (3-call tone bake-off, arms
gemini/dimensional/transcript). Raw per-clip predictions used below are from
`out/validation/results.json`.

---

## 1. Noise severity bands (`snr_none_db=20`, `snr_low_db=15`, `snr_medium_db=5`)

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

---

## 2. call_003 noise type ("radio" via AED vs client label "sharp static") and
   fusion Rule B's thin-margin override

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

---

## 3. Long-silence margin (`long_silence_s=10.0`, call_003's labeled-false gap
   measures 8.9s)

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

---

## 4. SQUIM 15s-window tradeoff vs full-clip deterministic detectors

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

---

## 5. Gemini call_002 "frustrated" vs label "neutral" (adjudicated known miss)

**Verdict: quantified via bake-off; no prompt changes (per this task's scope).**

Tone bake-off (3 real labeled calls, `--arms gemini dimensional transcript`,
`out/bakeoff.md`):

| arm | tone acc | macro F1 | $ / audio-min | s / clip |
|---|---|---|---|---|
| gemini | 67% | 0.667 | $0.00146 | 9.6/min |
| dimensional | 0% | 0.000 | $0 (local) | 12.5/min |
| transcript | 0% | 0.000 | $0.00081 (whisper local $0 + OpenAI text, metered) | 16.7/min |

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
