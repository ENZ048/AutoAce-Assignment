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
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 300

    # --- Noise severity via SNR (speech RMS vs non-speech RMS, dB) ---
    snr_none_db: float = 20.0   # > this: no meaningful interference
    snr_low_db: float = 15.0    # (low..none]: audible, doesn't interfere
    snr_medium_db: float = 5.0  # (medium..low]: occasionally interferes; <= : high

    # --- AED (PANNs CNN14) ---
    aed_prob_threshold: float = 0.35  # per user's converging research doc
    aed_min_support_s: float = 2.0    # sustained evidence, not a blip

    # --- Quality (SQUIM + clipdetect) ---
    # Initial bands; the 3 sample calls are all labeled "clear" -> calibrate against them.
    pesq_clear: float = 3.0
    pesq_slight: float = 2.0
    stoi_floor: float = 0.75          # below this, degrade one level
    clipping_ratio_max: float = 0.02  # >2% clipped frames -> severely_impaired override

    # --- Dimensional tone mapping (audeering A/V/D in [0,1]) — initial, calibrated in eval ---
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
    confidence_floor: float = 0.05
    confidence_ceiling: float = 0.98
    tone_degraded_confidence_cap: float = 0.40


@lru_cache
def get_settings() -> Settings:
    return Settings()
