# Dashboard track — session handoff state (internal, pre-handoff prune list)

Written 2026-07-17 while switching machines. Branch `dashboard`, HEAD `1709b31`, merge-base with `main` = `3cbf9a4` (31 commits). Read together with `CLAUDE.md`, the spec (`docs/superpowers/specs/2026-07-17-dashboard-design.md`), and the plan (`docs/superpowers/plans/2026-07-17-dashboard.md`).

## Where things stand

**All 12 plan tasks are implemented, per-task reviewed (spec + quality), and every review fix round is applied.** Execution used subagent-driven development: fresh implementer per task, adversarial per-task reviewer, fix subagents, re-reviews. Commits per task:

| Task | Commits |
|---|---|
| 1 settings/deps/hash helper | 70c4cbc, a588f45 |
| 2 SQLite job store | eb893f1, 5d6cb43 |
| 3 auth (bcrypt+JWT) | d49a722, 2391937, df39022, 31a3cb7 |
| 4 zipsafe extraction | d6fca74, 706eafd (+ `docs/DASHBOARD-BACKEND-REQUESTS.md` filed: `_find_manifest` case-sensitivity) |
| 5 app factory + login | a7b38d8, 9b52513 (httpx2 for warning-free TestClient) |
| 6 upload + validate | 10f2f3f, a0ed590 |
| 7 runner/dispatcher | e8b5da7, d29fc98 (pid-liveness orphan detection, status-guarded terminal writes) |
| 8 job routes | 1d1a66f, a79ebaf (status-gated artifact reads, out/ cleared on rerun) |
| 9 webapp scaffold | 27807bd, f268a83 |
| 10 jobs list + upload UI | fdbdfb1, c85e063 |
| 11 job detail UI | d15d73c, a9e2d7b |
| 12 SPA serving + e2e | 200b591, cec0255 (percent-encoded traversal fix), 1709b31 (/api JSON 404 guard) |

**Gates last verified by the controller on 2026-07-17 (this machine):**
- `.venv/bin/pytest tests/web/ -q -W error` → 59 passed, zero warnings
- `make test` → 178 passed / 13 deselected (backend 119 untouched + 59 web)
- `make lint` → ruff check + format clean (76 files)
- `cd webapp && npx vitest run` → 4/4; `npm run build` → clean
- Playwright e2e smoke (`webapp/e2e/smoke.spec.js`) passed twice against a live stub-analyze server (login → upload → validation report → start → live queue → results → download)

## What remains (in order)

### 1. Final whole-branch review (was about to dispatch when paused)

Dispatch one code-reviewer subagent on the most capable available model, per `superpowers:requesting-code-review`'s template, with:
- Range: base `3cbf9a4` → head `1709b31`. Regenerate the review package on the new machine: `<superpowers-skill-dir>/subagent-driven-development/scripts/review-package 3cbf9a4 1709b31` (or `git diff --stat` + `git diff -U10` to one file). Tell the reviewer to SKIP `webapp/package-lock.json` hunks.
- Requirements: the spec + plan paths above; trial §7 checklist in spec §1; hard constraints (`src/autoace_audio/` untouched — verify; self-contained; no speech in logs; data under gitignored `data/`).
- Emphasis: cross-task integration (upload→validate→confirm→dispatch→worker→progress→results→download as one flow; status-string/transition consistency across api/store/runner; SPA assumptions vs real API shapes; every `/api` route except login auth-guarded), security (upload handling, path safety, auth), concurrency core (dispatcher/worker/WAL).
- Ask it to TRIAGE the carried-forward per-task Minors below: each → fix-before-merge or accept, with reasoning. Dispatch ONE fix subagent afterward with the complete must-fix list (never one fixer per finding).

**Carried-forward Minors for triage (from the per-task reviews; the gitignored ledger `.superpowers/sdd/progress.md` on the old machine has the same list):**
1. hash_password CLI argv guard untested.
2. store: warnings-merge only tested from empty list; `interrupted` transition untested directly; silent no-op mutators lack debug logs; duplicate `create_job` IntegrityError undocumented.
3. auth: two tests bundle multiple assertions; username-check timing side channel (single fixed admin).
4. zipsafe: POSIX-specific backslash-safety assumption; no empty/dirs-only/dup-name zip tests (probe-verified OK).
5. api: login-failure sleep occupies a threadpool worker; dead `except HTTPException` branch in upload; some in-function test imports; no security logging on upload rejects; folder-mode `batch_root.txt` not `.resolve()`d.
6. runner: running-set-before-start ordering; redundant sleeps in one test; module-global `_processes`; worker except read-then-write not atomic (suggest `store.set_status_if_running` if a cancel route ever lands).
7. routes: delete/rerun TOCTOU; silent rmtree in delete; rerun-failed/delete-queued branches not route-tested; `/errors` and `/download` lack explicit 409-before-artifact tests; UnicodeDecodeError uncaught in results parse.
8. webapp: 401 substring match in axios interceptor; router v7 future-flag warnings; jobs empty-state copy; ENUM_CHIP coverage vs `schema.py` value domain; manifest-detection string match fragility; JobPage refresh lacks alive-guard (transient); no component tests for the four Task-11 components.
9. serving: security headers absent on unhandled 500s; no header assertions on SPA/static responses; bare `/api` (no slash) falls to SPA shell (trivial).

### 2. Real-pipeline acceptance (plan Task 12 Step 7 — controller-run, manual)

`.env` with real keys (GEMINI etc. — transfer `.env` to the new machine YOURSELF, securely; it is gitignored and must never travel via git), `DASHBOARD_*` vars set, `DASHBOARD_STUB_ANALYZE` unset → `make web` → browser: zip the 3 sample calls from `data/test_recordings/` + `labels.csv`, upload, confirm validation report, watch live queue (first file slow — models load once), cross-check the 9-field rows against known pipeline outputs, download all three artifacts, reload once mid-run (queue must restore). ~$0.01 Gemini spend. (`data/` is also local-only — copy the sample calls to the new machine yourself if needed.)

### 3. Finish the branch

`superpowers:finishing-a-development-branch`: branch is already pushed (this handoff pushed it); after 1-2 pass, tell the user **"dashboard ready for merge review"** — the backend session runs the merge review on `main` (per CLAUDE.md coordination protocol). Relay the `docs/DASHBOARD-BACKEND-REQUESTS.md` entry to the user for the backend session.

## New-machine environment setup

1. Clone / pull branch `dashboard` (this doc travels with it).
2. Python: create venv + `pip install -e '.[web,dev]'` (repo `Makefile` has the setup pattern; Python 3.12, ffmpeg required).
3. Web: `cd webapp && npm install` (node ≥20; repo used v25).
4. Copy `.env` securely by hand. Regenerate any needed `DASHBOARD_*` values (`python -m dashboard.hash_password '<pw>'`).
5. Sanity gates before resuming: `make test`, `make lint`, `cd webapp && npx vitest run && npm run build`.
6. Note: `.superpowers/sdd/` (task briefs/reports/ledger) is gitignored and stays on the old machine; briefs regenerate via the superpowers `task-brief` script from the plan; this doc supersedes the ledger for resume purposes.

## Standing user preferences for whoever resumes

Foreground subagents (background silence reads as "stuck"); deployment/hosting remains OUT of scope for this track's session (tentative later target: fresh isolated box); the Calling-Agent project is off-limits — this project stays fully self-contained; commit trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` on every commit; `git status` clean of `data/`, `.env`, `.superpowers/`, `out/`, `models_cache/` before every commit.
