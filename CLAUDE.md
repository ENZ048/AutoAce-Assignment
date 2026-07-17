# CLAUDE.md — Dashboard Track Instructions

You are the **dashboard session** for the AutoAce trial. This checkout (`AutoAce-Dashboard/`, branch `dashboard`) is yours. The backend/accuracy session works in the sibling `AutoAce-Assignment/` checkout and owns `main` — you never push to `main`.

## Read first, in order
1. `docs/DASHBOARD-KICKOFF.md` — mission, sizing math (client confirmed ~100 calls/batch), backend interfaces, deliverable checklist. It is your requirements source.
2. The trial brief §7 (hosted dashboard requirements): `../voice_tone_background_noise_dashboard_trial.pdf`.
3. Skim `src/autoace_audio/batch.py` and `pipeline.py` — the two entry points you wrap. Do not rebuild what they already do.

## Process (mandatory)
- Superpowers flow: **brainstorming → user-approved design → spec → plan → subagent-driven implementation.** No code, no scaffolding before design approval. The user has seen this flow all project — don't shortcut it.
- Run subagents in the **FOREGROUND** (user preference: background silence reads as "stuck").
- TDD per task; conventional commits; every commit message ends with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Before EVERY commit: `git status` must show nothing from `data/`, `.env`, `.superpowers/`, `out/`, `models_cache/`. These are gitignored — keep them that way.

## Hard boundaries
- **Branch discipline:** commit and push only to `dashboard` (`git push -u origin dashboard`).
- **Never modify `src/autoace_audio/`** (backend code). If you need a backend change (extra hook, diagnostics field), write it into `docs/DASHBOARD-BACKEND-REQUESTS.md`, commit, and tell the user — the backend session implements it on `main`, you rebase.
- **Confidentiality:** production call audio goes ONLY to Google Gemini (paid tier) and Deepgram — both already authorized. The dashboard must not add any new third-party service (analytics, CDN fonts, error trackers, storage) without asking the user first. No verbatim customer speech in committed files or logs.
- **Decisions that belong to the USER (ask, don't assume):** hosting target (their EC2/Docker vs an isolated box), the credentials scheme handed to AutoAce, anything spending money, submission timing.
- **Secrets:** `.env` in this folder has live keys (Gemini/Deepgram/OpenAI etc.). Never print, never commit, never send to any service. Keys get rotated before client handoff.

## Engineering facts you must respect
- Batches are ~100 calls (~500MB ZIP worst case): uploads return a job id immediately; processing is an async background job with persistent progress (page reload must not lose it); stage uploads to disk, never whole-in-RAM.
- Pipeline memory peaks ~4GB per worker and models are process-wide singletons: **worker processes, 2–4 max**; never run 100 concurrent analyses.
- `run_batch(input_path, out_dir, tone_arm=None, analyze_fn=analyze, progress_cb=None)` already does: ZIP extraction, manifest↔files two-way validation with warnings, per-file failure isolation, `results.csv`/`results.json`/`errors.csv` writing. Your job is auth + upload + job orchestration + progress + review UI + downloads around it.
- Environment: venv is prebuilt (`make test` → 120 passed). `transformers` is pinned `>=4.46,<5` — do not "upgrade" it. Python 3.12; ffmpeg required.
- Auth pattern to reuse: JWT + bcrypt from `/Users/kishorrane/Calling-Agent-Production/Calling-Agent-Backend-V6/utils/security.py` + `utils/password.py` (user's own production code — copying the pattern is encouraged).

## Coordination with the backend session
- Rebase on `main` when it moves (`git fetch origin && git rebase origin/main`) — backend changes land in `eval/experiments/` mostly; conflicts should be rare.
- When a milestone is reviewable: push, then tell the user "dashboard ready for merge review" — the backend session runs the merge review on `main`.
- This file and `docs/DASHBOARD-KICKOFF.md` are internal — they must NOT survive into the final client-visible repo (pre-handoff prune list; the backend session tracks it).

## Definition of done (trial §7)
Login with credentials we hand AutoAce → ZIP/folder + CSV manifest upload → validation report BEFORE processing → live batch progress that survives reload → per-file results in the exact 9-field schema → per-file errors named with reasons → CSV + JSON downloads preserving original filenames → deployed at a stable URL through the evaluation period.
