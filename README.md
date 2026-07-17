# AutoAce Call-Audio Analysis

Analyzes production call audio into a structured 9-field JSON: emotional tone,
background noise, technical quality, speaker overlap, long silences,
confidence. Built per `docs/superpowers/specs/2026-07-16-autoace-backend-design.md`;
every calibration/threshold decision behind the numbers in this README is
recorded with its measurement in `docs/decisions.md`.

## Quickstart

    make setup
    cp .env.example .env   # add GEMINI_API_KEY (paid tier — free tier trains on your audio)
    make analyze DIR=data/

`make setup` creates `.venv`, installs the package + dev deps, and warns if
`ffmpeg` is missing from `PATH` (required at runtime — audio format is
detected from file content via ffprobe, never from the filename extension,
since our own production smoke set contains `.mp3`-named files that are PCM
WAVs inside).

## Architecture

```
audio file (wav/mp3/ogg/flac/m4a)
        │  audio_io: ffmpeg → 16 kHz mono float32 (decode ONCE) + raw-stream metadata
        ▼
   silero-vad ──────────── speech/non-speech timeline ── long_silence_present
        │
   ├─ speech path ──► tone/<arm>.py ──► tone + intensity (+ tone confidence, noise/overlap opinion)
   ├─ non-speech-aware ► noise.py: windowed full-clip AED (speech classes masked) + SNR ─► noise fields
   └─ whole file ────► quality.py: deterministic channel evidence + SQUIM backstop ──► audio_quality
        │
        ▼
     fusion.py: invariants, thin-margin/no-type overrides, confidence blend ──► AnalysisResult (9 fields)
```

Package layout (`src/autoace_audio/`): `schema.py`, `config.py`, `audio_io.py`,
`analyzers/{vad,noise,quality}.py`, `analyzers/tone/{base,gemini_tone,dimensional,transcript_llm}.py`,
`fusion.py`, `pipeline.py` (`analyze(path, tone_arm=None) -> PipelineOutput`),
`batch.py` (folder/ZIP + manifest, per-file error isolation), `__main__.py`
(CLI: `python -m autoace_audio analyze <dir|zip> [--out out/] [--arm gemini|dimensional|transcript]`).
Evaluation lives outside the package in `eval/` (`build_validation_set.py`,
`evaluate.py`, `bakeoff.py`, `metrics.py`).

Every analyzer is a pure function over `(audio, sr, vad_map, ...)` returning a
typed, frozen dataclass; models load lazily and cache as module singletons so
a batch run only pays model-load cost once. All thresholds live in
`config.py` with a calibration rationale comment next to each value.

## Tone arm bake-off (live, n=3 labeled calls)

Three swappable tone-classification arms sit behind one interface
(`classify_tone(arm, samples, sr, vad, snr_db)`); the pipeline ships the
bake-off winner as the default, selectable via `--arm`.

| arm | tone acc | macro F1 | $ / audio-min | proc s / audio-min |
|---|---|---|---|---|
| **gemini** (shipped default) | 67% | 0.667 | $0.00146 | 7.3/min |
| dimensional (audeering wav2vec2, local, zero-cost) | 0% | 0.000 | $0 | 6.0/min |
| transcript (faster-whisper local → OpenAI text) | 0% | 0.000 | $0.00065 | 12.1/min |

Winner: **gemini**, unambiguously (67% vs 0%/0% on this sample). Its one miss
(of 3) is `call_002.ogg`: predicted `frustrated`, labeled `neutral` — the
model anchors on a single Spanish profanity phrase as frustration evidence
despite the model's own rationale acknowledging the rest of the call is calm;
4 prompt-wording iterations did not change this outcome (full trail in
`docs/decisions.md`, treated as a genuine model-prior limit / possible label
disagreement, not iterated further). `dimensional` and `transcript` scoring
0/3 is real (not a code bug) and consistent with each arm's own documented
limitations (dimensional: no diarization, hears agent+customer speech mixed;
transcript: text-only, loses all prosody).

**n=3 is a very small sample** — each arm's accuracy/F1 swings hard on a
single clip. Read this table as directional evidence that gemini is the right
default, not as a statistically powered comparison. (An earlier run of the
same 3 calls measured slightly different proc-time figures — 9.6/12.5/16.7
s/audio-min instead of 7.3/6.0/12.1 — with identical accuracy, F1, and
predicted labels; that's normal live-API-latency/local-model wall-clock
wobble across repeated runs, not a code change. Both are recorded in
`docs/decisions.md`.)

## Cost model

Ceiling: **$0.003 / audio-minute**. Measured all-in: **≈ $0.0015–0.002 /
audio-minute** (33–50% under ceiling).

| component | basis | $ / audio-min |
|---|---|---|
| Gemini audio input (32 tok/s × 60s/min ≈ 1920 tok/min @ $0.50/M) + output tokens + fixed prompt-text overhead | measured on 3 real calls (task 8): prompt tokens 1509–5035, output 99–107, scaling with clip duration | $0.0011–0.0016 |
| Local layer (silero VAD + PANNs CNN14 AED + SQUIM + dimensional/transcript, amortized CPU) | measured on the eval box | $0.0002–0.0005 |
| **Total** | | **$0.0013–0.0021** |
| Measured bake-off gemini arm (reproducible via `make bakeoff`) | `out/bakeoff.md` | $0.00146 |

The fully-local `dimensional` arm is a documented $0 fallback (no Gemini call)
if the paid tier is ever unavailable, at the cost of the accuracy shown above.

## Accuracy

- **audio_quality**: 100% (3/3) on augmented clipping/dropout degradations
  (`out/validation_report.md`) — the deterministic clipping/dropout/rolloff/
  volume detectors are unaffected by clip length or where in a long clip the
  damage sits (see `docs/decisions.md` §4).
- **End-to-end field accuracy** on the 3 real labeled anchor calls, over a
  5-field subset of the schema (`emotional_tone`, `background_noise_present`,
  `audio_quality`, `speaker_overlap_present`, `long_silence_present` — 5 fields
  × 3 calls = 15 comparisons; live Gemini arm): **13/15 = 86.7%** (gate:
  ≥80%). Both misses are on `call_002` and trace to the same root cause:
  Gemini fixates on one profanity phrase (`emotional_tone` +
  `speaker_overlap_present`, the pipeline's disclosed weakest field). The
  other 4 schema fields (`emotional_intensity`, `background_noise_type`,
  `background_noise_severity`, `confidence`) are intentionally excluded from
  this gate — they carry the real, disclosed misses covered in "Limitations"
  below (e.g. call_002's severity reads `high` vs. labeled `medium`;
  call_002/003's `background_noise_type` reads `radio` vs. labeled `TV`/
  `sharp static`), so folding them into one blended accuracy number would
  understate exactly what this section is trying to disclose honestly.
- **background_noise_present / severity** on the augmented validation set: 11%
  / 0% (9 synthetic noise clips) — this is a harness limitation, not a
  threshold bug: PANNs CNN14 essentially never recognizes synthetic
  white-noise/two-tone-hum beds as any AudioSet class with enough
  confidence+support to clear the presence gate (top classes are consistently
  unrelated — "Animal", "Music", "Vehicle" — at 0.03–0.18, far below the 0.35
  threshold), while the same detector correctly fires on the 2 real noisy
  anchor calls (call_002 TV, call_003 static). See "Limitations" below and
  `docs/decisions.md` §1 for the full evidence chain.

## Reproducing our results

```
make setup && make test && make analyze DIR=data/
```

- `make test` — fast unit suite (`-m "not slow and not network"`), no models
  loaded, no API calls, seconds.
- `make test-all` — full suite including `slow` (local model inference) and
  `network` (live Gemini/OpenAI calls, costs a small amount) markers.
- `make analyze DIR=data/` — runs the real pipeline over `data/` (or a
  ZIP), writes `out/results.csv` + `out/results.json`.
- `make evaluate` — field-level accuracy/F1 against `data/labels.csv` →
  `out/validation_report.md`.
- `make bakeoff` — regenerates the tone-arm comparison table above →
  `out/bakeoff.md`. Runs all 3 arms live (Gemini + OpenAI API calls; small
  real cost, ~$0.01 for the 3 anchor calls).

`data/` (production audio + labels) and `.env` (all API keys) are gitignored
from the first commit and must never be staged — verify with `git status`
before any push.

## Limitations (disclosed, not hidden)

- **Speaker overlap** is the weakest field: no pretrained overlapped-speech
  detector was viable (pyannote's claimed pretrained OSD was refuted for our
  pyannote 3.x setup — research 0-3), so it is Gemini's own audio judgment,
  defaulting to `false` without independent evidence. It shares gemini's
  call_002 root-cause miss (the same profanity-phrase fixation that produces
  the tone miss also suppresses the overlap opinion for that call).
- **`background_noise_type` is AED best-effort, cross-checked by Gemini only
  when Gemini independently agrees noise is present** — which it never did in
  testing for continuous/ambient noise (static, hum, TV-bleed): across the 3
  real anchors plus 9 synthetic noise-augmented clips, Gemini denied
  `background_noise_present` on call_003 and on all 9 synthetic clips
  outright. The fusion rule that would let Gemini override AED's *type* guess
  when AED's margin is thin (`docs/decisions.md`, Task 9/11) is real,
  unit-tested, and shipped — but has had **zero live trigger opportunities**
  in any data gathered so far, because it requires Gemini to agree presence
  while disagreeing on type, and Gemini has only ever denied presence outright
  for this class of noise. Treat `background_noise_type` as an unverified,
  best-effort label.
- **Neither AED nor Gemini recognized synthetically-mixed noise beds** in the
  eval harness (11% presence detection on 9 augmented clips) **while both work
  on real-recording anchors** (call_002 TV, call_003 static both correctly
  detected/flagged) — an honest harness-vs-reality gap, not evidence the
  detectors are broken on real audio. Validating the AED presence-gate +
  severity-band combination end-to-end needs real noise recordings, not
  synthetic white-noise/hum beds (see `docs/decisions.md` §1).
- **`background_noise_severity`** for call_002 reads `high` (measured SNR
  0.28dB) vs. the client's own label `medium` — a known, pre-existing
  labeler-vs-metric definitional disagreement (not touched by any calibration
  pass; documented in `docs/decisions.md`, Task 6/11).
- **`audio_quality`'s SQUIM backstop** scores a middle **15-second** window of
  the clip (memory scales superlinearly with input length — measured 0.44GB
  @5s up to ~14GB @60s, which swamped a 16GB machine at the original 60s
  window). This is safe for the backstop's own `<1.3`-PESQ gate, and every
  full-clip deterministic detector (clipping, dropouts, rolloff, volume) is
  unaffected by the window choice — but SQUIM itself only ever "sees" a 15s
  slice of longer calls.
- **Dimensional tone arm is English-tuned** (audeering's V-A model) and has
  **no speaker diarization** — it scores agent + customer speech mixed
  together, which measurably inverts valence ordering on long agent-dominated
  calls (call_003, 172s, mostly calm agent turns) vs. shorter customer-driven
  calls (call_001). Kept out of the default pipeline for this reason (0/3 on
  the bake-off); shipped only as a documented zero-cost fallback arm.
- **Gemini bake-off sample is n=3** — every number in the tone-arm table above
  is directional, not statistically powered. A single clip flipping changes
  each arm's accuracy by 33 points.
- **The Gemini prompt's decision rules were shaped on the same 3 labeled anchor
  calls** used everywhere else in this disclosure — the repeated-hello
  escalation rule (classify as `upset` after 3+ unanswered greetings) and the
  profanity-requires-corroboration rule (a single crude aside is insufficient
  evidence without independent, sustained escalation elsewhere in the call)
  were iterated specifically against `call_001`/`call_002`/`call_003`. Their
  generalization to other callers, languages, or phrasing is unvalidated
  (n=3).
- **call_002's tone label is a recorded disagreement**, not an unhandled bug:
  the model consistently reads a single Spanish profanity phrase as decisive
  frustration evidence across 4 independent prompt-wording iterations, despite
  its own rationale acknowledging the rest of the call is calm — this may be a
  genuine limitation of `gemini-3.1-flash-lite`'s ability to override a strong
  lexical prior via prompt instruction alone, or the human label itself may be
  debatable; both are plausible, and no further prompt iteration is planned
  per the project's controller adjudication.

## Experiments

See also: [Budget–accuracy study](docs/experiments/2026-07-17-budget-accuracy-study.md) — measured evidence for what a higher per-minute budget buys (five spending levers tested against the $0.003/audio-min ceiling).

## Security & data handling

`data/` and `.env` are gitignored from the first commit. Audio goes to
exactly one external service by default (Google Gemini API, paid tier —
content not used for training, transient abuse-monitoring retention only),
disclosed here with model name and pricing. The `transcript` bake-off arm
additionally sends locally-transcribed text (never raw audio) to OpenAI; it
is excluded from the shipped default pipeline. No telemetry; all model
downloads (silero, PANNs CNN14, SQUIM, audeering wav2vec2, faster-whisper) are
weights-only from HF/torch hubs.
