"""All tunable thresholds, with calibration rationale. Values are initial and
revisited by eval/ against the labeled + augmented validation set."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- API keys / models ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"  # verified available; ~$0.0011-0.0016/audio-min
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"  # Arm C bake-off only
    tone_arm: str = "gemini"  # gemini | dimensional | transcript

    # --- VAD / silence ---
    # AutoAce labeled a 7.4s dead-air stretch long_silence=false -> bar is above that.
    long_silence_s: float = 10.0
    # Silero defaults tightened for telephony: <250ms blips are usually artifacts;
    # 300ms merges stutters.
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 300

    # --- Noise severity via SNR (speech RMS vs non-speech RMS, dB) ---
    snr_none_db: float = 20.0   # > this: no meaningful interference
    snr_low_db: float = 15.0    # (low..none]: audible, doesn't interfere
    snr_medium_db: float = 5.0  # (medium..low]: occasionally interferes; <= : high

    # --- AED (PANNs CNN14) ---
    # Windowed full-clip AED (see analyzers/noise.py + task-6-report.md): CNN14 scores
    # a sliding window every aed_hop_s seconds; a class only counts as present if it's
    # sustained across enough hop-weighted window time, not a one-window spike.
    # Measured sweep {0.35, 0.30, 0.275, 0.25, 0.225, 0.20} against the 3 labeled
    # anchors (task-6-report.md): call_001 (no-noise) never sustains any class at any
    # threshold down to the 0.20 floor (peak mean 0.055 "Animal", support 0.0s
    # throughout); call_002 (TV, medium) already sustains "Radio" at the original
    # 0.35 (mean 0.229 across all windows, 4 of 13 windows individually clear 0.35,
    # support 10.0s >> the 5.0s floor). Per the controller's rule ("start at 0.35; if
    # call_002 misses, lower toward the widest margin, floor 0.20") — it doesn't
    # miss, so kept at 0.35 rather than lowering with no measured need.
    aed_prob_threshold: float = 0.35
    aed_min_support_s: float = 5.0  # >=2 activated windows at aed_hop_s=2.5s; kills
    # single-window CNN spikes on short/out-of-distribution fragments (measured on the
    # old per-gap-segment diagnostic: call_003 fragments spiked "Clip-clop" 0.41 /
    # "Horse" 0.39 / "Run" 0.32 on <1.5s slices — a single window must not be enough
    # on its own; see task-6-report.md).
    aed_window_s: float = 5.0  # CNN14 is trained on ~10s AudioSet clips; 5s balances
    # enough spectral context per window against localizing which part of a long call
    # actually carries the noise (call_003 runs 172s — one clip-wide read would blur
    # a short noise burst into silence).
    aed_hop_s: float = 2.5  # 50% overlap between consecutive windows; also the
    # per-window "sustained" time credit used by aed_min_support_s above.

    # --- Quality: deterministic channel evidence PRIMARY; SQUIM demoted to a
    # noise-conditioned backstop (task 7 rework -- full rationale, per-call evidence
    # table, and threshold calibrations in task-7-report.md). Original design scored
    # SQUIM PESQ/STOI bands directly; measured on the 3 labeled-clear anchor calls
    # with the original 60s scoring window, PESQ (2.11/1.64/2.09) ranked EXACTLY
    # with independent, non-ML background SNR from noise.py (23.1/0.3/10.7 dB)
    # (the shipped 15s window measures 1.95/2.18/2.34 -- ranking shuffles, but every
    # value still clears the backstop floor) -- SQUIM is responding to ambient noise, not
    # channel distortion, but audio_quality is scored INDEPENDENT of background
    # noise (all 3 noisy-yet-clean-channel calls are labeled "clear"). PESQ/STOI
    # bands therefore cannot be primary evidence for this field. Replaced with 4
    # deterministic, channel-only signals computed straight from the waveform + VAD
    # speech timeline; worst-triggered level wins, default CLEAR. SQUIM keeps a
    # narrow, gated role: it may only escalate a call when the background is clean
    # enough that noise CANNOT be the excuse for a catastrophic PESQ (real channel
    # damage, not a noise confound).
    #
    # Anchors pass all 4 deterministic detectors + the backstop with real measured
    # margin -- see the per-call evidence table appended to task-7-report.md.
    clipping_ratio_max: float = 0.02  # >2% clipped frames -> severely_impaired
    # override. UNCHANGED from the original design (real clipdetect API:
    # total_clipped_samples / total_samples, see quality.py) -- measured ~1e-6 on
    # all 3 anchors, 4-5 orders of magnitude of margin, no calibration case.

    # Hard near-zero runs (|sample|<1e-4, >=50ms) that start+end strictly inside a
    # VAD speech segment, normalized to occurrences per minute of speech. All 3
    # anchors measure 0.0/min (real dropouts are rare in these clips) -- huge margin
    # under both floors; synthetic degradations in eval/ will exercise the boundary
    # directly once that harness lands (task brief).
    dropout_high_per_min: float = 4.0
    dropout_low_per_min: float = 1.0

    # 95th-percentile-energy spectral rolloff (energy-weighted mean across 32ms/16ms
    # -hop speech frames). CALIBRATED DOWN from an initial 1200/2200 (task-7-report.md
    # rework log): those values FALSE-TRIGGERED on all 3 anchors when measured for
    # real -- call_001/002/003 measure 1248/1024/1591 Hz, and call_002 even trips the
    # original 1200 "severe" floor. This is real speech physics, not a detector bug
    # (confirmed via synthetic sanity checks in test_quality_logic.py: a <900Hz-only
    # synthetic signal correctly measures low, white noise correctly measures high)
    # -- energy-weighting toward the loudest frames pulls the aggregate toward
    # vowel-dominated frames, and voiced speech concentrates most of its energy well
    # below 2kHz even when perfectly clean (LTASS). Retuned with real measured
    # margin: 900 clears the worst anchor (call_002, 1024Hz) by 124Hz (~12%); 600
    # clears it by 424Hz (~41%). Still a 3-anchor sample with no labeled non-clear
    # example to calibrate the upper discrimination against -- revisit once eval/'s
    # synthetic degradations land (see task-7-report.md).
    rolloff_severe_hz: float = 600.0
    rolloff_slight_hz: float = 900.0

    # Speech-segment RMS in dBFS. All 3 anchors measure comfortably above both
    # floors (see task-7-report.md) -- no calibration case.
    volume_severe_dbfs: float = -45.0
    volume_slight_dbfs: float = -35.0

    # SQUIM backstop gate: only escalates when PESQ is catastrophic AND the
    # background is clean enough that noise cannot be the excuse. No anchor trips
    # this (15s-window measurements): call_001's SNR (23.1dB) clears
    # snr_no_excuse_db but its PESQ (1.95) doesn't clear this floor; call_002/003's
    # SNR (0.3/10.7dB) is below snr_no_excuse_db so the backstop is gated off
    # regardless of their PESQ (2.18/2.34) -- exactly the noise-confound case it
    # exists to ignore.
    pesq_severe_backstop: float = 1.3
    # Deliberately the same boundary as noise.py's snr_low_db ("audible, doesn't
    # interfere"): above this, background noise is not loud enough to plausibly
    # explain a catastrophic PESQ, so a bad score must be real channel damage.
    snr_no_excuse_db: float = 15.0

    # --- Dimensional tone mapping (audeering A/V/D in [0,1]) ---
    # Region boundaries in the valence-arousal plane; initial values from the
    # audeering model's roughly-centered output distribution (~0.5 mean), to be
    # calibrated on the labeled + augmented validation set in eval/. Intensity
    # bands follow the brief: low=subtle, medium=clear+sustained, high=escalated.
    va_satisfied_v: float = 0.60
    va_upset_v: float = 0.40
    va_upset_a: float = 0.60
    va_distressed_v: float = 0.30
    va_distressed_a: float = 0.75
    va_frustrated_v: float = 0.45
    va_frustrated_a_min: float = 0.40
    intensity_a_low: float = 0.45
    intensity_a_high: float = 0.65

    # --- Fusion confidence ---
    # Never emit 0.0/1.0 (calibration honesty: we always carry some uncertainty).
    # Degraded cap 0.40: when the tone arm failed and a fallback answered, the tone
    # fields are best-effort — confidence must signal that to the evaluator.
    confidence_floor: float = 0.05
    confidence_ceiling: float = 0.98
    tone_degraded_confidence_cap: float = 0.40


@lru_cache
def get_settings() -> Settings:
    return Settings()
