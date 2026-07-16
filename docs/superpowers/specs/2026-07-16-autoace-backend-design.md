# AutoAce Call-Audio Analysis — Backend Design

**Date:** 2026-07-16 · **Status:** Approved · **Scope:** Analysis engine + batch CLI (dashboard is a separate, later spec)

## 1. Context

AutoAce AI's technical trial requires a system that analyzes production call audio (customer ↔ AutoAce's AI voice agent) and emits a structured 9-field JSON per clip, evaluated on a hidden test set. Full brief: `voice_tone_background_noise_dashboard_trial.pdf`.

Hard constraints:

- **Cost:** ≤ $0.003 per audio-minute for the final production approach.
- **Confidentiality:** production audio never enters git or any non-disclosed third-party service. The one external API used (Gemini, paid tier) is disclosed in the memo with pricing and retention terms.
- **Reproducibility:** AutoAce must be able to run the system on new audio from the repo alone.
- **Scoring reality:** emotional tone, background noise, and technical audio quality are scored *independently*. Known traps: loudness ≠ emotion; poor audio quality ≠ background noise.

Facts about the data that shape this design (measured from the 3 labeled sample calls):

- Calls are customer ↔ TTS agent ("Erica"). The schema targets the **customer's** emotion; the always-calm agent voice must not dilute classification.
- The sample set is bilingual (EN + ES) → tone classification must be multilingual.
- A 7.4s dead-air stretch is labeled `long_silence_present: false` → the "long silence" bar is ~10s+.
- A call with "sharp static" noise is labeled `audio_quality: clear` → noise and quality pipelines must not share evidence.
- All sample labels carry `confidence: 0.82` (placeholder) → confidence is ours to calibrate.

## 2. Goals / Non-goals

**Goals**

1. `analyze(path) -> AnalysisResult` — one clip in, validated 9-field result out.
2. Batch CLI: folder or ZIP + CSV manifest (`name,result_json`) in → per-file results + CSV/JSON export out, with per-file failure isolation.
3. Tone bake-off harness producing the accuracy/cost/latency comparison that (a) selects the shipped classifier and (b) fills the memo's "compare two materially different approaches" requirement.
4. Evaluation harness: macro F1, per-class metrics, confusion matrix on a leakage-safe validation set.

**Non-goals (deferred)**

- Web dashboard, auth, hosting (separate spec after backend works).
- Fine-tuning or training custom models (3 labeled calls make it pointless).
- Real-time/streaming analysis; this is batch.

## 3. Output contract

`schema.py` defines `AnalysisResult` (pydantic v2, enums exact per brief):

| Field | Type | Source of truth |
|---|---|---|
| `emotional_tone` | enum: neutral/satisfied/frustrated/upset/distressed | Tone classifier (customer speech only) |
| `emotional_intensity` | enum: low/medium/high | Tone classifier |
| `background_noise_present` | bool | AED on non-speech segments |
| `background_noise_type` | string ("" when absent) | AED dominant class → concise label |
| `background_noise_severity` | enum: none/low/medium/high | SNR (speech RMS vs non-speech RMS) |
| `audio_quality` | enum: clear/slightly_impaired/severely_impaired | SQUIM MOS/PESQ mapping + clipdetect override |
| `speaker_overlap_present` | bool | Overlap detector (see §5.4) |
| `long_silence_present` | bool | VAD gap > threshold (default 10s) |
| `confidence` | float 0–1 | Fusion (calibrated, §6) |

Invariants enforced in `fusion.py`: `background_noise_present == false` ⇒ `type == ""` and `severity == "none"`; severity > none ⇒ present == true. Serialization matches the brief's example byte-for-byte in field order.

## 4. Architecture

```
audio file (wav/mp3/ogg/flac/m4a)
        │  audio_io: ffmpeg → 16 kHz mono float32 (decode ONCE) + raw-stream metadata
        ▼
   silero-vad ──────────── speech/non-speech timeline ── long_silence_present
        │
   ├─ speech path ──► tone/<arm>.py ──► tone + intensity (+ tone confidence)
   ├─ non-speech path ► noise.py: AED (speech-classes masked) + SNR ─► noise fields
   └─ whole file ────► quality.py: SQUIM + clipdetect ──► audio_quality
        │                       overlap.py ──► speaker_overlap_present
        ▼
     fusion.py: invariants, overrides, confidence ──► AnalysisResult
```

Module map (package `src/autoace_audio/`): `schema.py`, `config.py`, `audio_io.py`, `analyzers/{vad,noise,quality,overlap}.py`, `analyzers/tone/{base,gemini_tone,dimensional,transcript_llm}.py`, `fusion.py`, `pipeline.py`, `batch.py`. Evaluation lives outside the package in `eval/`.

Every analyzer: pure function over `(audio: np.ndarray, sr: int, vad_map)` → typed dataclass. Models load lazily and cache as module singletons (batch amortization). All thresholds live in `config.py` with the calibration rationale next to each value — nothing magic inline.

## 5. Component specs

### 5.1 `audio_io.py`
ffmpeg subprocess (stdin/stdout pipes, no temp files — pattern proven in our Calling-Agent backend) → 16 kHz mono float32 numpy + duration. Rejects unreadable/zero-length input with `DecodeError`. Also retains the pre-resample stream info (sample rate, channels, codec) for the quality analyzer's bandwidth heuristics. **Format is detected from content (ffprobe), never from the filename extension** — our own production smoke set contains `.mp3`-named files that are PCM WAV inside; the hidden set may do the same.

### 5.2 `vad.py` — silero-vad
Speech timestamps (natively supports 8k/16k; >30× realtime CPU). Outputs: speech segments, non-speech gaps, total speech ratio. `long_silence_present` = any single mid-call gap > `LONG_SILENCE_S` (default **10.0**; calibrated: AutoAce labeled a 7.4s gap false). Pin silero v5 as fallback if v6 misbehaves on telephony audio (known G.711 regression reports).

### 5.3 `noise.py` — AED + SNR
- **Type:** AudioSet-taxonomy tagger run on concatenated non-speech segments, with speech-adjacent classes masked (Speech, Conversation, Narration, Shout, Whispering, Throat clearing…). Sustained (≥2 s cumulative) top class above probability threshold → mapped to a concise human label ("TV", "office chatter", "road noise", "static"…). Primary model: **PANNs CNN14** (torch, single-framework stack); YAMNet (TF) documented as the swap-in alternative behind the same interface — final pick benchmarked during implementation on the sample calls.
- **Presence:** true iff a non-masked class passes the sustained threshold (never inferred from SNR alone).
- **Severity:** SNR = 10·log10(P_speech/P_nonspeech) over VAD-segmented RMS; mapping (calibratable): >20 dB → none, 15–20 → low, 5–15 → medium, ≤5 → high. Sample calls provide two known "medium" anchors.

### 5.4 `overlap.py` — weakest field, staged plan
1. Timebox (30 min): try pyannote overlapped-speech detection / diarization-derived overlap regions on current pyannote 3.x. (Research refuted the "natively ships pretrained OSD" claim 0-3 — verify, don't trust.)
2. Fallback: Gemini judgment (same call as tone, zero extra cost) cross-checked with an energy/spectral-flux heuristic on speech segments.
Decision + measured accuracy on the 3 labeled calls (2 true / 1 false) recorded in `docs/decisions.md`. Cumulative-overlap threshold ≈ >3 s or >5% of speech time, so back-channel "uh-huh" doesn't trigger it.

### 5.5 `quality.py` — SQUIM + clipdetect
Torchaudio-SQUIM objective (PESQ/STOI/SI-SDR, reference-free, CPU, verified working on torch 2.13). Mapping (initial, calibrated on augmented set): PESQ ≥ 3.0 → clear; 2.0–3.0 → slightly_impaired; < 2.0 → severely_impaired, blended with STOI for intelligibility. **Override:** `clipdetect` frame-clipping ratio above threshold forces `severely_impaired` regardless of MOS (clipping survives normalization; naive peak checks don't see it). NISQA noted as alternative; not primary (older torch-compat friction). Judges the *technical channel only* — noise evidence is never an input.

### 5.6 `tone/` — three arms behind one interface
`ToneClassifier.classify(audio, sr, vad_map, context) -> ToneResult(tone, intensity, p_dist | None, notes)`

- **Arm A — `gemini_tone.py` (expected primary):** `gemini-3.1-flash-lite` (verified available on the project key), one structured-JSON call per clip: audio + label definitions verbatim from the brief + 3 labeled calls as few-shot anchors + DSP hints (SNR, loudness stats) + explicit instruction *"call between an AI agent (Erica) and a customer — classify the CUSTOMER's emotional tone; do not infer emotion from loudness."* `response_mime_type=application/json` with schema. Timeout 60 s, 3 retries exponential backoff. Cost ≈ $0.0011–0.0016/min interactive (32 tok/s audio billing).
- **Arm B — `dimensional.py` (zero-cost cross-check):** audeering `wav2vec2-large-robust-12-ft-emotion-msp-dim` → arousal/valence/dominance on *speech segments*; V-A threshold mapping: satisfied = V high; neutral = V mid ∧ A low-mid; frustrated = V low-mid ∧ A mid; upset = V low ∧ A high; distressed = V very-low ∧ A very-high. Intensity from arousal bands. Known limits (recorded honestly in the memo): English-tuned; can't isolate the customer without diarization; thresholds tuned on the labeled data.
- **Arm C — `transcript_llm.py` (bake-off only):** faster-whisper (multilingual) transcript + prosody stats → OpenAI text model → same enums. Kept out of the default pipeline unless it wins.

The shipped pipeline uses the bake-off winner; runner-up optionally kept as a disagreement flag feeding confidence.

## 6. Fusion & confidence

`fusion.py` merges typed analyzer outputs → enforces §3 invariants → applies overrides (clipping ⇒ severely_impaired; empty non-speech evidence ⇒ noise absent) → computes `confidence`: weighted blend of tone-arm probability (or Gemini self-report normalized against bake-off-measured reliability), AED margin, and SNR distance from severity boundaries; squashed to [0.05, 0.98], calibrated on the validation set so stated confidence ≈ empirical accuracy (reliability-diagram check in `eval/`).

## 7. Validation methodology

- **Leakage rule:** all segments from one call stay on the same side of any split (grouped / leave-one-call-out).
- **Set construction** (`eval/build_validation_set.py`): (a) segment the 3 calls into 15–45 s clips, hand-label each against the brief's definitions; (b) synthetic augmentation with *known* ground truth — overlay ESC-50/Freesound noise types (TV, chatter, traffic, typing, music) onto clean segments at controlled SNRs (severity labels for free), and apply controlled degradations (clipping, band-limiting, dropouts, gain) for quality labels. Augmentation recipes are seeded/reproducible and documented.
- **Metrics** (`eval/evaluate.py`): macro F1 + accuracy for tone; per-field accuracy/F1 for the rest; confusion matrices → memo. Report validation numbers only from held-out folds, never train-set (brief explicitly demands this).
- **Bake-off** (`eval/bakeoff.py`): arms A/B/C on identical clips; accuracy, per-class F1, $ per audio-minute (measured tokens × live prices), wall-clock per clip. Output: one markdown table.
- **Unlabeled smoke set:** 11 real production recordings from our own calling agent (`data/test_recordings/`, 70 s–11.5 min, mixed WAV/mislabeled-extension files, likely Hindi/English) — used for format-robustness, throughput measurement on long clips, and eyeball sanity checks; never for accuracy claims.

## 8. Error handling

- Typed exceptions: `DecodeError`, `AnalyzerError(component)`, `ToneClassifierError`; batch layer catches per file — one corrupt file yields a per-file error record (`name, error, reason`), batch continues (brief tests this).
- Gemini failure after retries → pipeline degrades: deterministic fields still emitted, tone falls back to Arm B, confidence capped ≤ 0.4, note recorded. Total failure only on undecodable audio.
- Batch layer validates manifest ↔ files both directions before processing (missing file / missing row → reported, not fatal).
- All model downloads pinned to exact revisions; first-run downloads cached to `models_cache/`.

## 9. Cost & latency model (memo inputs)

Per audio-minute, final pipeline: Gemini 3.1 Flash-Lite ≈ $0.0011–0.0016 (interactive; Batch API ≈ half — documented as the production-scale path); local models ≈ $0.0002–0.0005 amortized CPU (measured on the eval box); **total ≈ $0.0013–0.002 → ~35–55% under the ceiling with the fully-local Arm B fallback at <$0.001 documented.** Latency target: ≤ 0.3× realtime per clip on 4 CPU cores excluding network; measured numbers reported per clip and per audio-minute in the memo.

## 10. Security & data handling

- `data/` (production audio + labels) and `.env` (all keys) are gitignored from the very first commit; verified via `git status` before every push.
- Audio goes to exactly one external service (Google Gemini API, paid tier — content not used for training; transient abuse-monitoring retention), disclosed in the memo with model name, pricing, and terms link. STT bake-off vendors receive audio only during experiments, disclosed likewise, and are excluded from the shipped pipeline.
- No telemetry; model downloads from HF/torch hubs are weights-only, pinned.

## 11. Engineering standards

- Python 3.12, `pyproject.toml`, `ruff` (lint+format), `pytest`; `Makefile`: `setup / test / lint / analyze / evaluate / bakeoff`.
- Type hints throughout; analyzers documented with input/output contracts; README with architecture sketch, 3-command quickstart, cost table.
- Tests: unit (V-A mapping boundaries, SNR→severity, silence gap logic, manifest validation, schema round-trip — no model loads); integration (3 sample calls e2e vs `labels.csv`; Gemini-dependent tests marked `@pytest.mark.network`).
- Conventional commits, each leaving tests green: `chore: scaffold` → `feat(schema)` → `feat(audio-io)` → `feat(vad)` → `feat(noise)` → `feat(quality)` → `feat(overlap)` → `feat(tone)` → `feat(pipeline+batch)` → `feat(eval+bakeoff)` → `docs(memo)`.

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Gemini tone accuracy unbenchmarked on call audio | Bake-off measures it before we commit; Arm B is the wired fallback |
| Overlap detection has no verified tool | Staged plan §5.4, timeboxed, honest memo disclosure |
| Spanish/multilingual clips | Arm A native multilingual; Arm C whisper multilingual; Arm B limitation documented |
| TTS agent dilutes emotion | Prompt-level speaker targeting (A); diarization-gated SER only if time allows (B) |
| Hidden set differs from samples (formats, lengths) | ffmpeg-normalized ingest; format matrix in tests; thresholds in config |
| Key leakage | .env gitignored day one; keys were shared in chat — rotate after the trial |

## 13. Deferred

Dashboard (upload/auth/progress/download UI), hosting, and the Batch-API cost optimization each get their own spec once the backend is proven against the labeled calls.
