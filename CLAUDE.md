# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make setup        # create .venv, install .[web,dev] (torch, fastapi, …); warns if ffmpeg missing
make test         # fast unit suite: pytest -m "not slow and not network" (seconds, no models/API)
make test-all     # full suite incl. slow (local model inference) + network (live Gemini/OpenAI, small $)
make lint         # ruff check + ruff format --check over src/ tests/ eval/
make analyze DIR=data/   # run the pipeline over a dir or ZIP → out/results.{csv,json}
make evaluate     # field accuracy/F1 vs data/labels.csv → out/validation_report.md
make bakeoff      # regenerate the 3-arm tone comparison → out/bakeoff.md (live API, ~$0.01)
make web          # build the SPA into webapp/dist, then serve API+SPA on :8000
make web-dev      # API with --reload; pair with `cd webapp && npm run dev` (Vite HMR, proxies /api)
```

- **Run one test:** `.venv/bin/pytest tests/unit/test_batch.py::test_flat_zip_processes_files -q`
- **Web tests only:** `.venv/bin/pytest tests/web/ -q` (needs the `[web]` extra; note the whole
  `tests/` tree imports fastapi at collection time, so a `.[dev]`-only venv fails to collect — always `make setup`).
- **SPA tests:** `cd webapp && npx vitest run`
- Python 3.12 + ffmpeg/ffprobe on PATH are required at runtime.

## Architecture

Two independent products share one repo and one venv:

1. **`src/autoace_audio/`** — the analysis pipeline (CLI + importable). `analyze(path, tone_arm=None)`
   in `pipeline.py` is the single entry point; `batch.py` wraps it for folders/ZIPs.
2. **`src/dashboard/`** — a FastAPI + React (`webapp/`) batch-review UI that *wraps* the pipeline.
   It must never reimplement pipeline logic — it calls `batch.run_batch`.

### Pipeline data flow (`pipeline.py` orchestrates)

`audio_io.py` decodes **once** to 16 kHz mono float32 (format sniffed from content via ffprobe,
never the extension). Then analyzers run as **pure functions** over `(audio, sr, vad_map, …)`
returning frozen dataclasses: `analyzers/vad.py` (silence), `analyzers/noise.py` (windowed PANNs
AED + gap-SNR), `analyzers/quality.py` (deterministic checks + gated SQUIM backstop), and one of
three swappable tone arms behind `analyzers/tone/base.py` (`gemini_tone` is the shipped default;
`dimensional` is a local $0 fallback; `transcript` is a bake-off arm). `fusion.py` applies
cross-field invariants and produces the schema-valid 9-field `AnalysisResult`.

- **Models are lazy module-level singletons** — a batch pays load cost once. `pipeline.preload_models()`
  warms them up front so the dashboard can show a "loading models" phase instead of a stuck first file.
- **All calibration thresholds live in `config.py`** with a rationale comment next to each value.
  Every threshold decision is recorded in `docs/decisions.md` (client-visible; keep it updated).
- **Cost ceiling is a hard constraint: ≤ $0.003/audio-minute.** Shipping config measures $0.00146.
  Any change touching the Gemini call or adding a vendor must re-check this.

### Dashboard concurrency model (the subtle part)

- **One batch at a time.** `runner.dispatch_once` (called each second by the app's background task)
  starts the oldest queued job only if nothing is running. Each batch runs in a **detached
  subprocess** (`python -m dashboard.worker`, `start_new_session=True`) — *not* multiprocessing,
  so a server restart doesn't kill an in-flight batch; on restart `sweep_orphans` re-adopts a live
  worker by pid. Job state machine lives in `store.py` (SQLite, WAL; two writers: API + worker).
- **`DASHBOARD_STUB_ANALYZE=1`** swaps in canned results (no models/keys/network) — use it for
  dashboard/UI work so you aren't paying for Gemini calls.
- Uploaded batches + results live under `DASHBOARD_DATA_DIR` (default `data/`), outside the git
  checkout on the server, so `git pull` never touches user data. Deleting a batch removes its DB row
  and job dir; `audit.py` keeps an append-only `audit.jsonl` at the data-dir root that delete can't reach.

## Conventions

- **TDD** — write the failing test first, watch it fail, then implement (see the dense existing tests).
- **Never commit** `data/`, `.env`, `out/` (all gitignored) — they hold production audio and API keys.
- **No verbatim customer speech** in committed files, logs, or test fixtures — paraphrase.
- Audio may go **only** to Google Gemini (paid tier). The `transcript` arm sends locally-transcribed
  *text* to OpenAI and is excluded from the shipped pipeline. Any external API must stay disclosed in
  the README / `docs/technical-memo.md`.
- Two-space-indent line length is 100 (ruff). Run `make lint` before committing.

## Context

Paid client trial for AutoAce (see `README.md` and `docs/technical-memo.md`). Ground truth is only
**3 labeled anchor calls** — accuracy numbers are directional, not statistical, and are always
disclosed as such. When adding a lever, prove it the way `docs/experiments/` does: measured,
repeated, and reported honestly including the losses.
