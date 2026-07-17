# Dashboard Track — Kickoff Brief

*(For the Claude session working in this checkout. Branch: `dashboard`. The backend/accuracy session works in the sibling `AutoAce-Assignment` checkout on `main` — coordinate merges through it.)*

## Mission

Build the hosted evaluation dashboard for the AutoAce trial (brief §7 of `voice_tone_background_noise_dashboard_trial.pdf`, copy in the parent `Test/` folder). AutoAce must be able to: log in with provided credentials → upload a batch (folder or ZIP of audio + one CSV manifest `name,result_json`) → watch validation + per-file progress → review per-call results → download results as CSV/JSON preserving original filenames. A single malformed file must never kill a batch.

**Fresh sizing fact from the client (2026-07-17): a typical batch ≈ 100 calls.** Design for 100 comfortably, degrade gracefully beyond.

## Ground rules (non-negotiable, inherited from the project)

1. Work on branch `dashboard` in THIS checkout only. Never commit to `main`. Push with `git push -u origin dashboard`.
2. NEVER commit `data/`, `.env`, `.superpowers/`, `out/` (gitignored — verify `git status` before every commit).
3. Follow the superpowers flow: brainstorm → design approval → spec → plan → subagent-driven implementation. FOREGROUND agents (user preference).
4. Conventional commits ending with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
5. Confidentiality: production audio goes only to Google Gemini paid tier (+ Deepgram, authorized) — the dashboard itself must not introduce new third-party services without asking the user. No verbatim customer speech in committed files.
6. HOSTING IS THE USER'S DECISION (their EC2/Docker vs an isolated box) — brainstorm both, ask them, don't assume.

## What the backend already gives you (don't rebuild any of it)

- `autoace_audio.pipeline.analyze(path, tone_arm=None) -> PipelineOutput` — one clip → validated `AnalysisResult` + diagnostics dict (duration, cost-relevant token counts, timings).
- `autoace_audio.batch.run_batch(input_path, out_dir, tone_arm=None, analyze_fn=analyze, progress_cb=None) -> BatchReport` — accepts a folder OR `.zip`, validates the manifest both directions (missing/extra files → warnings list), per-file failure isolation (`report.errors`), writes `results.csv` (`name,result_json`), `results.json`, `errors.csv`. `progress_cb(done, total, name)` is your progress hook.
- `autoace_audio.schema.AnalysisResult` — the exact client output contract; `.to_result_json()` is byte-ordered per the brief.
- `make setup` builds the venv (transformers pinned `>=4.46,<5` — don't fight it). `make test` = 120 fast tests green.
- Models auto-download on first use; peak pipeline memory ~4GB; **process files sequentially or with small worker counts — do NOT run 100 concurrent analyses.**

## Sizing math for 100-call batches (design inputs, measured numbers)

- Measured cost: ~$0.00146/audio-min (Gemini 3.1 Flash-Lite arm). 100 calls × ~2 min avg ≈ **$0.30/batch** — negligible.
- Throughput: local layer + Gemini ≈ roughly 20–40s per typical call end-to-end on a modest CPU box, sequential → **100 calls ≈ 35–70 min sequential; ~3 workers ≈ 15–25 min**. Implication: uploads must return immediately with a job id; processing is an async background job with live progress; results persist server-side; the browser session must be able to close and come back. (Exact per-call timing: measure on the target box during implementation — diagnostics carry `elapsed_s`.)
- Upload size: 100 calls × ~1–5MB ≈ up to ~500MB per ZIP — stage to disk, stream the unzip, never buffer whole uploads in RAM; enforce a max-size with a clear error.
- Concurrency caution: PANNs/SQUIM/silero are CPU-hungry and models are process-wide singletons — worker processes (not threads) if parallelizing, 2–4 max on a 4-core box.

## Strongly suggested starting points (proven patterns, user's own code)

- Auth: JWT + bcrypt pattern from the user's production backend — `/Users/kishorrane/Calling-Agent-Production/Calling-Agent-Backend-V6/utils/security.py` + `utils/password.py` + `routes/auth.py`. One admin user provisioned via env is enough for the trial (AutoAce gets one set of credentials).
- Stack recommendation to bring INTO brainstorming (not pre-decided): FastAPI + a minimal server-rendered or small-SPA UI; background jobs via a simple process-pool job runner or RQ — justify against the 100-call async requirement. The user's React admin dashboard (`Calling-Agent-Admin-Dashboard-v6`: `Login.jsx`, `CallLogs.jsx` results table, native `<audio>` player, `axiosClient.js`) is trimmed-down reusable if a SPA is chosen.
- Deploy: repo already has `Dockerfile`-ready patterns in the user's production project (`Dockerfile` with ffmpeg baked, gunicorn conf, docker-compose) — mirror them.

## Deliverable checklist (from the trial brief §7 — the dashboard is 10% of the score, batch handling under "production practicality" another 10%)

- [ ] Hosted, browser-accessible, login with credentials we hand AutoAce
- [ ] Folder or ZIP upload + one CSV manifest; manifest↔files validation reported clearly BEFORE processing
- [ ] Batch progress/completion status visible; survives page reload
- [ ] Per-file failure isolation with named file + reason (backend already provides this — surface it)
- [ ] Results table per file, exact schema fields; downloadable CSV and JSON preserving original filenames
- [ ] Stays up through their evaluation period; documented start/stop + credential rotation for the memo

## Coordination protocol with the backend session

- The backend session owns `main`. When the dashboard reaches a reviewable milestone, push `dashboard` and tell the user — the backend session runs the merge review (its reviewers know the codebase invariants).
- If you need a backend change (new hook, extra diagnostics), DON'T edit `src/autoace_audio/` on this branch — write the request in `docs/DASHBOARD-BACKEND-REQUESTS.md`, commit, and the user relays it. Prevents merge hell.
- `git pull origin main` + rebase your branch when the backend lands accuracy improvements (they're isolated to `eval/experiments/` + occasional `config.py` — low conflict risk).

## Suggested opening prompt for this session

> Read docs/DASHBOARD-KICKOFF.md fully. Then run the superpowers brainstorming skill for the dashboard design: explore this repo's batch/pipeline interfaces first, ask me the open questions one at a time (hosting is mine to decide), propose 2–3 approaches, and don't write any code until I approve the design. We're on branch `dashboard`.
