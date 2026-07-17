"""Mock-level tests for exp3: the advocate-prompt assembly (what advocate_pass
actually sends to Gemini, including the empty-rationale fallback) and the
flip/keep/regression bookkeeping in run_once, including the standing
amendment's per-pass token logging. No network, no local data/ dependency,
no model loads -- every collaborator exp3_advocate imports is monkeypatched
at its imported name (same convention as test_pipeline.py / test_gemini_parse.py)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from eval.experiments import common
from eval.experiments import exp3_advocate as exp3
from eval.experiments.exp3_advocate import ADVOCATE_PROMPT

from autoace_audio.analyzers.tone.base import ToneResult
from autoace_audio.config import Settings
from autoace_audio.schema import EmotionalIntensity, EmotionalTone

# ---------------------------------------------------------------------------
# advocate-prompt assembly: pure string formatting
# ---------------------------------------------------------------------------


def test_advocate_prompt_interpolates_verdict_and_rationale():
    prompt = ADVOCATE_PROMPT.format(
        tone="frustrated", intensity="medium", rationale="raised voice, sighed twice"
    )
    assert "emotional_tone=frustrated, emotional_intensity=medium." in prompt
    assert "Its reasoning: raised voice, sighed twice" in prompt
    assert "DIFFERENT" in prompt
    assert "FINAL verdict" in prompt
    assert "Return JSON only." in prompt


def test_advocate_prompt_states_all_five_tone_definitions():
    prompt = ADVOCATE_PROMPT.format(tone="neutral", intensity="low", rationale="x")
    for label in ("neutral=", "satisfied=", "frustrated=", "upset=", "distressed="):
        assert label in prompt


# ---------------------------------------------------------------------------
# advocate_pass: what actually gets sent to (a faked) Gemini, and what it
# returns -- Client is faked, google.genai.types stays real (pure data
# construction, no network), same convention as test_gemini_parse.py.
# ---------------------------------------------------------------------------


class _FakeAudio:
    samples = np.zeros(1600, dtype=np.float32)
    sr = 16000


class _FakeUsage:
    def __init__(self, in_tok, out_tok):
        self.prompt_token_count = in_tok
        self.candidates_token_count = out_tok


class _FakeResp:
    def __init__(self, data, in_tok, out_tok):
        self.text = json.dumps(data)
        self.usage_metadata = _FakeUsage(in_tok, out_tok)


def _fake_settings() -> Settings:
    return Settings(_env_file=None, gemini_api_key="fake-key-for-test")


def _patch_advocate_collaborators(monkeypatch, resp):
    """Fakes every collaborator advocate_pass touches, returns the list its
    fake Gemini client will record generate_content() calls into."""
    calls: list[dict] = []

    class _FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return resp

    class _FakeGenaiClient:
        def __init__(self, api_key=None):
            self.models = _FakeModels()

    monkeypatch.setattr(exp3, "get_settings", _fake_settings)
    monkeypatch.setattr(exp3, "load_audio", lambda path: _FakeAudio())
    monkeypatch.setattr(exp3, "encode_opus_ogg", lambda samples, sr: b"fake-ogg-bytes")
    monkeypatch.setattr("google.genai.Client", _FakeGenaiClient)
    return calls


def test_advocate_pass_embeds_first_pass_verdict_in_the_sent_prompt(monkeypatch):
    resp = _FakeResp({"emotional_tone": "neutral", "emotional_intensity": "medium"}, 900, 60)
    calls = _patch_advocate_collaborators(monkeypatch, resp)

    exp3.advocate_pass("call_002.ogg", "frustrated", "medium", "raised voice")

    prompt = calls[0]["contents"][1]
    assert "emotional_tone=frustrated, emotional_intensity=medium." in prompt
    assert "Its reasoning: raised voice" in prompt


def test_advocate_pass_falls_back_to_none_recorded_for_empty_rationale(monkeypatch):
    resp = _FakeResp({"emotional_tone": "neutral", "emotional_intensity": "medium"}, 900, 60)
    calls = _patch_advocate_collaborators(monkeypatch, resp)

    exp3.advocate_pass("call_002.ogg", "frustrated", "medium", "")

    prompt = calls[0]["contents"][1]
    assert "Its reasoning: (none recorded)" in prompt


def test_advocate_pass_returns_data_cost_and_per_pass_tokens(monkeypatch):
    """Standing amendment: advocate_pass must surface its own token usage so
    run_once can log it per-clip per-pass -- not cost-only."""
    resp = _FakeResp({"emotional_tone": "neutral", "emotional_intensity": "medium"}, 900, 60)
    _patch_advocate_collaborators(monkeypatch, resp)

    data, cost, tokens = exp3.advocate_pass("call_002.ogg", "frustrated", "medium", "x")

    assert data == {"emotional_tone": "neutral", "emotional_intensity": "medium"}
    assert tokens == {"in": 900, "out": 60}
    assert cost == pytest.approx((900 * 0.50 + 60 * 1.50) / 1e6)


# ---------------------------------------------------------------------------
# run_once: flip/keep/regression bookkeeping + per-pass token logging
# ---------------------------------------------------------------------------


class _FakeGuard:
    def __init__(self):
        self.checked: list[float] = []
        self.added: list[float] = []

    def check(self, projected_usd):
        self.checked.append(projected_usd)

    def add(self, cost_usd):
        self.added.append(cost_usd)


def _sequence(values):
    """classify_tone doesn't receive the clip name -- run_once calls it once
    per ANCHORS iteration, in order, so a plain queue dispatches correctly."""
    it = iter(values)
    return lambda *a, **kw: next(it)


def _tone_result(tone, intensity, in_tok, out_tok, rationale="because"):
    return ToneResult(
        tone=EmotionalTone(tone),
        intensity=EmotionalIntensity(intensity),
        confidence=0.8,
        raw={
            "response": {"rationale": rationale},
            "prompt_tokens": in_tok,
            "output_tokens": out_tok,
        },
    )


def test_run_once_flags_keep_win_flip_and_regression_flip_per_clip(monkeypatch, tmp_path):
    """Three clips, three outcomes: call_001 stays correct (keep), call_002's
    wrong first-pass tone gets flipped to correct (the hoped-for win),
    call_003's correct first-pass tone gets flipped to wrong (the regression
    the brief/spec say must be reported with equal prominence, spec S7)."""
    truth = {
        "call_001.ogg": {"emotional_tone": "upset", "emotional_intensity": "high"},
        "call_002.ogg": {"emotional_tone": "neutral", "emotional_intensity": "medium"},
        "call_003.ogg": {"emotional_tone": "satisfied", "emotional_intensity": "medium"},
    }
    first_by_clip = {
        "call_001.ogg": _tone_result("upset", "high", 1500, 100),
        "call_002.ogg": _tone_result("frustrated", "medium", 1600, 110),
        "call_003.ogg": _tone_result("satisfied", "medium", 1550, 105),
    }
    # (final verdict dict, advocate-pass cost, advocate-pass tokens)
    final_by_clip = {
        "call_001.ogg": (
            {"emotional_tone": "upset", "emotional_intensity": "high"},
            0.0010,
            {"in": 1500, "out": 90},
        ),
        "call_002.ogg": (
            {"emotional_tone": "neutral", "emotional_intensity": "medium"},
            0.0011,
            {"in": 1600, "out": 95},
        ),
        "call_003.ogg": (
            {"emotional_tone": "upset", "emotional_intensity": "medium"},
            0.0012,
            {"in": 1550, "out": 92},
        ),
    }
    advocate_calls: list[tuple] = []

    def _fake_advocate_pass(name, first_tone, first_intensity, rationale):
        advocate_calls.append((name, first_tone, first_intensity, rationale))
        return final_by_clip[name]

    guard_box: list[_FakeGuard] = []

    def _fake_spend_guard():
        g = _FakeGuard()
        guard_box.append(g)
        return g

    monkeypatch.setattr(exp3, "load_truth", lambda: truth)
    monkeypatch.setattr(exp3, "load_audio", lambda path: _FakeAudio())
    monkeypatch.setattr(exp3, "analyze_vad", lambda samples, sr: object())
    monkeypatch.setattr(
        exp3, "analyze_noise", lambda samples, sr, vad: SimpleNamespace(snr_db=10.0)
    )
    monkeypatch.setattr(exp3, "classify_tone", _sequence([first_by_clip[n] for n in exp3.ANCHORS]))
    monkeypatch.setattr(exp3, "advocate_pass", _fake_advocate_pass)
    monkeypatch.setattr(exp3, "SpendGuard", _fake_spend_guard)
    # log_run (called for real) writes via common.OUT_DIR -- redirect it so this
    # mock-level test never touches the real out/experiments/ directory.
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)

    payload = exp3.run_once(1)
    pc = payload["per_clip"]

    # call_001: first pass right, advocate keeps it -- no flip
    assert pc["call_001.ogg"]["flipped"] is False
    assert pc["call_001.ogg"]["correct"] == {"emotional_tone": True, "emotional_intensity": True}
    assert pc["call_001.ogg"]["first_correct"] == {
        "emotional_tone": True,
        "emotional_intensity": True,
    }

    # call_002: first pass wrong, advocate flips it right -- the win
    assert pc["call_002.ogg"]["flipped"] is True
    assert pc["call_002.ogg"]["correct"] == {"emotional_tone": True, "emotional_intensity": True}
    assert pc["call_002.ogg"]["first_correct"] == {
        "emotional_tone": False,
        "emotional_intensity": True,
    }

    # call_003: first pass right, advocate flips it wrong -- the regression
    assert pc["call_003.ogg"]["flipped"] is True
    assert pc["call_003.ogg"]["correct"] == {"emotional_tone": False, "emotional_intensity": True}
    assert pc["call_003.ogg"]["first_correct"] == {
        "emotional_tone": True,
        "emotional_intensity": True,
    }

    # standing amendment: per-clip, per-pass token logging (not cost-only)
    assert pc["call_001.ogg"]["tokens"] == {
        "first": {"in": 1500, "out": 100},
        "final": {"in": 1500, "out": 90},
    }
    assert pc["call_002.ogg"]["tokens"] == {
        "first": {"in": 1600, "out": 110},
        "final": {"in": 1600, "out": 95},
    }
    assert pc["call_003.ogg"]["tokens"] == {
        "first": {"in": 1550, "out": 105},
        "final": {"in": 1550, "out": 92},
    }

    # run_once threads pass-1's verdict+rationale into the advocate call
    assert ("call_002.ogg", "frustrated", "medium", "because") in advocate_calls

    # cost sums both passes across all 3 clips; SpendGuard checked then charged
    expected_cost = sum(
        common.gemini_cost(
            first_by_clip[n].raw["prompt_tokens"], first_by_clip[n].raw["output_tokens"]
        )
        + final_by_clip[n][1]
        for n in exp3.ANCHORS
    )
    assert payload["cost_usd"] == pytest.approx(expected_cost)
    guard = guard_box[0]
    assert guard.checked  # checked BEFORE spending
    assert guard.added == [pytest.approx(expected_cost)]
