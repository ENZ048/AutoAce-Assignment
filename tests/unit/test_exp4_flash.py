"""Mock-level tests for exp4: prompt/schema reuse from the shipping arm
(byte-identical, not copied), the gemini-3.5-flash model id + controller-
verified pricing wiring (not Flash-Lite's, not the rejected preview
candidate's), and the run-log shape (model_id + model + pricing at the top
level, per-clip tokens + cost_usd). No network, no local data/ dependency,
no model loads -- every collaborator exp4_flash imports is monkeypatched at
its imported name (same convention as test_exp3_advocate.py)."""

import json

import numpy as np
import pytest
from eval.experiments import common
from eval.experiments import exp4_flash as exp4

from autoace_audio.analyzers.tone import gemini_tone
from autoace_audio.config import Settings

# ---------------------------------------------------------------------------
# prompt/schema reuse: must be the SAME object as the shipping arm imports,
# not a copy -- a copy could silently drift from the shipping prompt.
# ---------------------------------------------------------------------------


def test_build_prompt_is_the_shipping_function_not_a_copy():
    assert exp4.build_prompt is gemini_tone.build_prompt


def test_response_schema_is_the_shipping_schema_not_a_copy():
    assert exp4.GEMINI_RESPONSE_SCHEMA is gemini_tone.GEMINI_RESPONSE_SCHEMA


# ---------------------------------------------------------------------------
# model id + pricing constants: the controller-decided substitute for the
# brief's dead placeholder id, and its verified rates (not Lite's, not the
# recorded-but-not-implemented E4b preview candidate's).
# ---------------------------------------------------------------------------


def test_flash_model_is_the_controller_approved_substitute():
    assert exp4.FLASH_MODEL == "gemini-3.5-flash"
    # brief's original placeholder -- confirmed 404 at pre-flight, never used
    assert exp4.FLASH_MODEL != "gemini-3.1-flash"


def test_flash_pricing_matches_controller_verified_rates_not_lite_or_preview():
    assert pytest.approx(1.50) == exp4.FLASH_IN_PER_1M
    assert pytest.approx(9.00) == exp4.FLASH_OUT_PER_1M
    # must not silently fall back to Flash-Lite's shipping rates...
    assert exp4.FLASH_IN_PER_1M != common.GEMINI_LITE_IN
    assert exp4.FLASH_OUT_PER_1M != common.GEMINI_LITE_OUT
    # ...or the E4b preview candidate's rates (recorded in the report, not implemented)
    assert (exp4.FLASH_IN_PER_1M, exp4.FLASH_OUT_PER_1M) != (1.00, 3.00)
    assert exp4.PRICING_SOURCE.startswith("https://")
    assert "2026-07-17" in exp4.PRICING_SOURCE
    assert pytest.approx(0.02) == exp4.EST_COST_PER_RUN


# ---------------------------------------------------------------------------
# classify_flash: what actually gets sent to (a faked) Gemini, and what it
# returns -- Client is faked, google.genai.types stays real (pure data
# construction, no network), same convention as test_exp3_advocate.py.
# ---------------------------------------------------------------------------


class _FakeAudio:
    samples = np.zeros(16000 * 20, dtype=np.float32)  # 20s @ 16kHz
    sr = 16000


class _FakeUsage:
    def __init__(self, in_tok, out_tok):
        self.prompt_token_count = in_tok
        self.candidates_token_count = out_tok


class _FakeResp:
    def __init__(self, data, in_tok, out_tok):
        self.text = json.dumps(data)
        self.usage_metadata = _FakeUsage(in_tok, out_tok)


class _FakeVad:
    speech_ratio = 0.6


class _FakeNoise:
    def __init__(self, snr_db):
        self.snr_db = snr_db


def _fake_settings() -> Settings:
    return Settings(_env_file=None, gemini_api_key="fake-key-for-test")


_FULL_DATA = {
    "emotional_tone": "neutral",
    "emotional_intensity": "medium",
    "speaker_overlap_present": True,
}


def _patch_classify_collaborators(monkeypatch, resp, snr_db=12.5):
    """Fakes every collaborator classify_flash touches, returns the list its
    fake Gemini client will record generate_content() calls into."""
    calls: list[dict] = []

    class _FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return resp

    class _FakeGenaiClient:
        def __init__(self, api_key=None):
            self.models = _FakeModels()

    monkeypatch.setattr(exp4, "get_settings", _fake_settings)
    monkeypatch.setattr(exp4, "load_audio", lambda path: _FakeAudio())
    monkeypatch.setattr(exp4, "analyze_vad", lambda samples, sr: _FakeVad())
    monkeypatch.setattr(exp4, "analyze_noise", lambda samples, sr, vad: _FakeNoise(snr_db))
    monkeypatch.setattr(exp4, "encode_opus_ogg", lambda samples, sr: b"fake-ogg-bytes")
    monkeypatch.setattr("google.genai.Client", _FakeGenaiClient)
    return calls


def test_classify_flash_sends_the_flash_model_id(monkeypatch):
    resp = _FakeResp(_FULL_DATA, 2000, 120)
    calls = _patch_classify_collaborators(monkeypatch, resp)

    exp4.classify_flash("call_001.ogg")

    assert calls[0]["model"] == "gemini-3.5-flash"


def test_classify_flash_prompt_matches_shipping_build_prompt_output_exactly(monkeypatch):
    resp = _FakeResp(_FULL_DATA, 2000, 120)
    calls = _patch_classify_collaborators(monkeypatch, resp, snr_db=12.5)

    exp4.classify_flash("call_001.ogg")

    sent_prompt = calls[0]["contents"][1]
    expected = gemini_tone.build_prompt(_FakeAudio.samples.size / _FakeAudio.sr, 12.5, 0.6)
    assert sent_prompt == expected


def test_classify_flash_uses_flash_pricing_not_lite_pricing(monkeypatch):
    resp = _FakeResp(_FULL_DATA, 2000, 120)
    _patch_classify_collaborators(monkeypatch, resp)

    data, cost, tokens = exp4.classify_flash("call_001.ogg")

    assert data == _FULL_DATA
    assert tokens == {"in": 2000, "out": 120}
    expected_cost = (2000 * 1.50 + 120 * 9.00) / 1e6
    assert cost == pytest.approx(expected_cost)
    # sanity: must NOT equal what lite pricing would have produced for the same tokens
    lite_cost = common.gemini_cost(2000, 120)
    assert cost != pytest.approx(lite_cost)


def test_classify_flash_handles_missing_usage_metadata(monkeypatch):
    """None token counts must count as zero cost, not raise (matches
    common.gemini_cost's contract, exercised end-to-end here)."""
    resp = _FakeResp(_FULL_DATA, None, None)
    _patch_classify_collaborators(monkeypatch, resp)

    data, cost, tokens = exp4.classify_flash("call_001.ogg")

    assert tokens == {"in": None, "out": None}
    assert cost == 0.0


# ---------------------------------------------------------------------------
# run_once: log shape (model_id, model, pricing, per-clip tokens + cost_usd)
# and the standing amendment's guard-before/add-after ordering.
# ---------------------------------------------------------------------------


def test_run_once_logs_model_pricing_and_per_clip_tokens_and_cost_in_guard_order(
    monkeypatch, tmp_path
):
    truth = {
        "call_001.ogg": {
            "emotional_tone": "upset",
            "emotional_intensity": "high",
            "speaker_overlap_present": False,
        },
        "call_002.ogg": {
            "emotional_tone": "neutral",
            "emotional_intensity": "medium",
            "speaker_overlap_present": True,
        },
        "call_003.ogg": {
            "emotional_tone": "satisfied",
            "emotional_intensity": "medium",
            "speaker_overlap_present": True,
        },
    }
    fake_returns = {
        "call_001.ogg": (
            {
                "emotional_tone": "upset",
                "emotional_intensity": "high",
                "speaker_overlap_present": False,
            },
            0.010,
            {"in": 1500, "out": 100},
        ),
        "call_002.ogg": (
            {
                "emotional_tone": "neutral",
                "emotional_intensity": "medium",
                "speaker_overlap_present": True,
            },
            0.011,
            {"in": 1600, "out": 110},
        ),
        "call_003.ogg": (
            {
                "emotional_tone": "satisfied",
                "emotional_intensity": "low",
                "speaker_overlap_present": True,
            },
            0.030,
            {"in": 5000, "out": 105},
        ),
    }

    def _fake_classify_flash(name):
        return fake_returns[name]

    order: list[tuple[str, float]] = []

    class _OrderedGuard:
        def check(self, projected_usd):
            order.append(("check", projected_usd))

        def add(self, cost_usd):
            order.append(("add", cost_usd))

    monkeypatch.setattr(exp4, "load_truth", lambda: truth)
    monkeypatch.setattr(exp4, "classify_flash", _fake_classify_flash)
    monkeypatch.setattr(exp4, "SpendGuard", lambda: _OrderedGuard())
    # log_run (called for real) writes via common.OUT_DIR -- redirect it so this
    # mock-level test never touches the real out/experiments/ directory.
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)

    payload = exp4.run_once(1)

    assert payload["exp"] == "exp4_flash"
    assert payload["model_id"] == "gemini-3.5-flash"
    assert payload["model"] == "gemini-3.5-flash"  # standing amendment: new field
    assert payload["pricing"] == {
        "in_per_1m": 1.50,
        "out_per_1m": 9.00,
        "source": exp4.PRICING_SOURCE,
    }

    pc = payload["per_clip"]
    assert pc["call_001.ogg"]["tokens"] == {"in": 1500, "out": 100}
    assert pc["call_001.ogg"]["cost_usd"] == pytest.approx(0.010)
    assert pc["call_003.ogg"]["tokens"] == {"in": 5000, "out": 105}
    assert pc["call_003.ogg"]["cost_usd"] == pytest.approx(0.030)
    assert pc["call_003.ogg"]["correct"] == {
        "emotional_tone": True,
        "emotional_intensity": False,
        "speaker_overlap_present": True,
    }
    assert payload["cost_usd"] == pytest.approx(0.010 + 0.011 + 0.030)

    # log actually written to disk (log_run runs for real, redirected via OUT_DIR)
    on_disk = json.loads((tmp_path / "exp4_flash_run1.json").read_text())
    assert on_disk["model"] == "gemini-3.5-flash"
    assert on_disk["per_clip"]["call_002.ogg"]["tokens"] == {"in": 1600, "out": 110}

    # SpendGuard consulted BEFORE any spend, charged with the measured total AFTER
    assert order[0] == ("check", exp4.EST_COST_PER_RUN)
    assert order[-1] == ("add", pytest.approx(0.010 + 0.011 + 0.030))
