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

    # --- Quality (SQUIM + clipdetect) ---
    # Initial bands; the 3 sample calls are all labeled "clear" -> calibrate against them.
    pesq_clear: float = 3.0
    pesq_slight: float = 2.0
    stoi_floor: float = 0.75          # below this, degrade one level
    clipping_ratio_max: float = 0.02  # >2% clipped frames -> severely_impaired override

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
