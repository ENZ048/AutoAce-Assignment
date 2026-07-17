# Budget–Accuracy Study — Design

*2026-07-17. Goal: measured proof of what a higher per-minute budget buys, presentable to AutoAce. Approved scope: all five levers including Deepgram (new, user-authorized audio recipient). Spend cap: $10 total, tracked from real usage metadata. Deliverables: client-visible study doc + private pitch page.*

## 1. Purpose and framing

The shipping backend costs ~$0.0015/audio-min against a $0.003 ceiling and misses a known cluster of fields (tone on call_002, intensity one-band-low on 2/3 anchors, noise type wording, overlap on 1/3). This study measures — with live runs, N=3 repeats, and logged costs — how much of that cluster each of five spending levers recovers. Output is an ablation table: per lever, its accuracy delta on the fields it targets and its measured incremental $/audio-min.

Non-goals: changing the shipping pipeline's defaults; retuning any calibrated threshold; producing statistical significance from n=3 anchors (the study says "directional" wherever it is).

## 2. Evaluation protocol

**Baseline first.** The shipping configuration is re-run 3× over the 3 anchors before any experiment, logging per-field outputs and cost. All experiment deltas are reported against this measured baseline distribution, not against a single historical run.

**Repeats.** Every Gemini-dependent config runs 3× (temperature and service variance). Deterministic configs (Deepgram overlap math, dimensional arm) run once. Report tables show all runs, not just means; a field "improves" only if it is right in ≥2 of 3 runs where baseline was wrong in ≥2 of 3.

**Eval sets per experiment** (each lever is scored only on the fields it can affect, on data whose truth we actually hold):

| Experiment | Data | Scored fields |
|---|---|---|
| E1 gap-listening | 3 anchors + the 9 synthetic noise clips (data/validation) | background_noise_present, severity (exact); type reported verbatim beside truth |
| E2 few-shot intensity | 3 anchors | emotional_intensity |
| E3 devil's advocate | 3 anchors | emotional_tone (intensity secondary) |
| E4 Flash arm | 3 anchors | emotional_tone, emotional_intensity, speaker_overlap_present |
| E5 diarization overlap | 3 anchors | speaker_overlap_present; bonus: dimensional-arm tone on customer-only audio |
| Combined stack | 3 anchors + 9 noise clips | all of the above |

**Leakage rules.** E2's audio exemplars use leave-one-out: when scoring call k, the two exemplar excerpts come only from the other two anchors. No threshold anywhere is tuned on the eval data during this study.

## 3. The five experiments

**E1 — Gap-listening noise question.** New focused Gemini call: concatenate the clip's VAD gap segments (≥1.0s each, up to 60s total), re-encode opus, prompt: "These are the between-speech moments of one phone call. Describe any background sound: is meaningful noise present, what type (concise label), constant or intermittent?" Structured JSON (present: bool, type: str, character: str). Fusion simulation (offline, in the experiment harness — NOT in shipping fusion.py): where AED and gap-listening disagree on presence, report both; where AED is present-with-thin-margin or type-less, adopt gap-listening type. Hypothesis: unlike full-call prompting (0 live noise confirmations ever), speech-free audio lets Gemini hear the bed. Runs 3×. Clips whose gaps total <2s are recorded as "not applicable" rather than scored.

**E2 — Audio few-shot intensity anchors.** Prompt gains two short audio exemplars (~15–20s excerpts, chosen once per source call at build time by a deterministic rule: the 20s window containing the maximum VAD speech-seconds, ties broken earliest) labeled with their ground-truth intensity, inserted before the target clip: "Example A is intensity=high. Example B is intensity=medium. Now classify the following call." Leave-one-out per §2. Measures: intensity accuracy; cost delta (exemplar audio tokens measured from usage metadata). Prompt text otherwise identical to shipping.

**E3 — Devil's-advocate tone pass.** Two-call chain: (1) shipping classify; (2) second Gemini call with the same audio + "A first analysis concluded {tone}/{intensity} because {rationale}. Argue the strongest case for a DIFFERENT reading of the customer, then give your final verdict" — structured JSON, final verdict wins. Measures: tone accuracy, flip behavior (does call_002 move; do correct calls stay correct — regression risk is the interesting number), 2× audio cost measured.

**E4 — Full Gemini Flash arm.** Identical shipping prompt/schema, model `gemini-3.1-flash` (live-verify exact id + audio pricing at run time; record both in the run log). Measures: tone/intensity/overlap accuracy and the real per-minute cost from its own token billing.

**E5 — Deepgram diarization overlap.** Send each anchor (once) to Deepgram prerecorded API with diarization enabled; compute overlap from speaker-turn intervals: overlap_present = any cross-speaker interval intersection ≥ 0.5s that is not a bare back-channel (intersecting segment's own duration < 1.0s AND fewer than 3 words → ignored), thresholds documented as first-pass choices, not tuned on the eval. Measures: overlap accuracy vs labels (false/true/true), agreement with Gemini's opinion. Bonus (free): re-run the dimensional arm on customer-only concatenated audio (customer = the diarized speaker with lower total speaking time overlap with Erica's known first-turn voice; if speaker attribution is ambiguous, run both and report) — measures whether diarization rescues the local fallback arm.

**Combined stack.** The levers that individually improved their target fields (decided by the ≥2-of-3 rule) run together: e.g. Flash-or-lite + few-shot + advocate for tone/intensity, gap-listening for noise, Deepgram for overlap. 3×. This row is the headline of the pitch: "at $X/min, the anchors score Y."

## 4. Code layout

```
eval/experiments/
  __init__.py          # empty
  common.py            # run logging (out/experiments/<exp>_<runN>.json), cost accounting,
                       # cumulative spend guard (abort > $10, warn > $7), anchor loading,
                       # per-field compare helpers
  exp1_gap_noise.py    # each exp module: build_config() + run(runs: int) -> results
  exp2_fewshot.py
  exp3_advocate.py
  exp4_flash.py
  exp5_overlap.py
  run_all.py           # baseline + E1..E5 + combined, in order, resumable per-experiment
```

Rules: experiments import shipping code (classify_tone, analyzers, encode_opus_ogg, prompts) and pass overrides as ARGUMENTS; they never mutate config defaults or fusion. Deepgram key read from .env (present). Every API response's raw usage metadata lands in the run log. out/experiments/ is untracked (out/ is gitignored). No experiment file imports models at module scope. New deps: `deepgram-sdk` (or plain httpx call if the SDK fights us — implementer's choice, disclosed).

## 5. Deliverables

1. `docs/experiments/2026-07-17-budget-accuracy-study.md` (committed, client-visible): methodology (this design, condensed), baseline table, per-experiment tables (all runs shown), the cost-vs-accuracy summary, tiered recommendation (what $0.003 buys / what ~$0.005–0.007 buys), limitations (n=3, synthetic beds, single-day variance).
2. `Test/BUDGET-PITCH.md` (NOT in repo): one page, plain English, for the user's presentation.
3. Run logs under out/experiments/ for reproducibility (untracked; the study doc embeds the numbers).

## 6. Budget & safety

- Hard cap $10; guard in common.py sums measured cost cumulatively across all experiments (Gemini tokens at live rates; Deepgram at its posted per-minute rate) and refuses to start a run that projects past the cap.
- Audio recipients: Google paid tier + Deepgram only (user-authorized 2026-07-17). OpenAI continues to receive transcripts only (not used in this study).
- data/, out/, .env stay untracked; the study doc quotes no verbatim customer speech (paraphrase rule from the final review stands).
- 16GB machine: experiments run sequentially; no experiment loads more than the shipping pipeline already does (Deepgram is API-side).

## 7. Risks / honest expectations

- E1 may show Gemini *still* denies synthetic beds (they may sound artificial even in isolation) — that is itself a publishable result: "noise typing is not purchasable at any nearby price; disclosed limitation stands."
- E3 may regress correct calls (advocate flips a right answer) — the regression count is reported with the same prominence as the win.
- E5's speaker-attribution heuristic may misfire on the 2-speaker-with-TV call; the "run both, report both" fallback keeps it honest.
- n=3 anchors: every accuracy claim in the deliverables carries the caveat inline, and the combined-stack table is presented as directional evidence justifying a *pilot* at higher budget, not a guarantee.
