import json

import numpy as np
import pytest
from eval.experiments import common
from eval.experiments import exp1_gap_noise as exp1
from eval.experiments.exp1_gap_noise import concat_gaps

from autoace_audio.analyzers.vad import Segment, VadMap
from autoace_audio.config import Settings

SR = 16000


def _vad(gaps, total_s):
    return VadMap(
        speech=[],
        gaps=[Segment(a, b) for a, b in gaps],
        speech_ratio=0.5,
        max_gap_s=max((b - a for a, b in gaps), default=0.0),
        long_silence_present=False,
        total_s=total_s,
    )


def test_concat_gaps_keeps_only_long_gaps_and_caps():
    total = 30.0
    samples = np.arange(int(total * SR), dtype=np.float32)
    vad = _vad([(0.0, 0.5), (2.0, 4.0), (10.0, 12.5)], total)
    out = concat_gaps(samples, SR, vad, min_gap_s=1.0, cap_s=60.0)
    # 0.5s gap dropped; 2.0s + 2.5s kept = 4.5s
    assert out.size == int(4.5 * SR)
    # content really comes from the gap regions (first kept sample = t=2.0s)
    assert out[0] == samples[int(2.0 * SR)]


def test_concat_gaps_caps_total_seconds():
    total = 200.0
    samples = np.zeros(int(total * SR), dtype=np.float32)
    vad = _vad([(0.0, 50.0), (60.0, 130.0)], total)
    out = concat_gaps(samples, SR, vad, cap_s=60.0)
    assert out.size == int(60.0 * SR)


def test_concat_gaps_empty_when_no_qualifying_gap():
    samples = np.zeros(SR, dtype=np.float32)
    vad = _vad([(0.0, 0.4)], 1.0)
    assert concat_gaps(samples, SR, vad).size == 0


# ---------------------------------------------------------------------------
# ask_gemini_gaps: Task 8 amendment -- must also surface its own token usage
# (was cost-only) so combined.py's gap-listening lever can log per-clip
# tokens like every other Task 5-8 module. No network -- Client faked,
# google.genai.types stays real, same convention as test_exp3_advocate.py.
# ---------------------------------------------------------------------------


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


def _patch_ask_gemini_gaps_collaborators(monkeypatch, resp):
    calls: list[dict] = []

    class _FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return resp

    class _FakeGenaiClient:
        def __init__(self, api_key=None):
            self.models = _FakeModels()

    monkeypatch.setattr(exp1, "get_settings", _fake_settings)
    monkeypatch.setattr("google.genai.Client", _FakeGenaiClient)
    return calls


def test_ask_gemini_gaps_returns_data_cost_and_tokens(monkeypatch):
    data = {
        "background_noise_present": True,
        "background_noise_type": "static",
        "character": "constant",
    }
    resp = _FakeResp(data, 1800, 40)
    _patch_ask_gemini_gaps_collaborators(monkeypatch, resp)

    result_data, cost, tokens = exp1.ask_gemini_gaps(b"fake-ogg-bytes")

    assert result_data == data
    assert tokens == {"in": 1800, "out": 40}
    assert cost == pytest.approx(common.gemini_cost(1800, 40))


def test_ask_gemini_gaps_handles_missing_usage_metadata(monkeypatch):
    data = {"background_noise_present": False, "background_noise_type": "", "character": "none"}
    resp = _FakeResp(data, None, None)
    _patch_ask_gemini_gaps_collaborators(monkeypatch, resp)

    _, cost, tokens = exp1.ask_gemini_gaps(b"fake-ogg-bytes")

    assert tokens == {"in": None, "out": None}
    assert cost == 0.0


def test_ask_gemini_gaps_sends_the_configured_model_and_prompt(monkeypatch):
    data = {"background_noise_present": False, "background_noise_type": "", "character": "none"}
    resp = _FakeResp(data, 100, 10)
    calls = _patch_ask_gemini_gaps_collaborators(monkeypatch, resp)

    exp1.ask_gemini_gaps(b"fake-ogg-bytes")

    assert calls[0]["model"] == _fake_settings().gemini_model
    assert calls[0]["contents"][1] == exp1.PROMPT


# ---------------------------------------------------------------------------
# run_once: the widened ask_gemini_gaps tokens must reach the per-clip log
# for unskipped clips (not dropped on the floor).
# ---------------------------------------------------------------------------


class _FakeAudio:
    samples = np.ones(SR * 20, dtype=np.float32)
    sr = SR


def test_run_once_logs_tokens_for_unskipped_clips(monkeypatch, tmp_path):
    truth = {
        "call_001.ogg": {"background_noise_present": False, "background_noise_type": ""},
        "call_002.ogg": {"background_noise_present": True, "background_noise_type": "TV"},
        "call_003.ogg": {"background_noise_present": True, "background_noise_type": "sharp static"},
    }

    def _fake_vad(samples, sr):
        return _vad([(0.0, 10.0)], 20.0)  # 10s gap, well over the 2s floor

    def _fake_ask_gemini_gaps(blob):
        return (
            {
                "background_noise_present": True,
                "background_noise_type": "static",
                "character": "constant",
            },
            0.0015,
            {"in": 900, "out": 25},
        )

    order: list[str] = []

    class _OrderedGuard:
        def check(self, projected_usd):
            order.append("check")

        def add(self, cost_usd):
            order.append("add")

    monkeypatch.setattr(exp1, "load_truth", lambda: truth)
    monkeypatch.setattr(exp1, "_noise_clips", lambda: {})
    monkeypatch.setattr(exp1, "load_audio", lambda path: _FakeAudio())
    monkeypatch.setattr(exp1, "analyze_vad", _fake_vad)
    monkeypatch.setattr(exp1, "encode_opus_ogg", lambda samples, sr: b"fake-ogg-bytes")
    monkeypatch.setattr(exp1, "ask_gemini_gaps", _fake_ask_gemini_gaps)
    monkeypatch.setattr(exp1, "SpendGuard", lambda: _OrderedGuard())
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)

    payload = exp1.run_once(1)

    pc = payload["per_clip"]
    assert pc["call_001.ogg"]["skipped"] is False
    assert pc["call_001.ogg"]["tokens"] == {"in": 900, "out": 25}
    assert pc["call_001.ogg"]["cost_usd"] == pytest.approx(0.0015)

    on_disk = json.loads((tmp_path / "exp1_gap_noise_run1.json").read_text())
    assert on_disk["per_clip"]["call_002.ogg"]["tokens"] == {"in": 900, "out": 25}
    assert order[0] == "check"
    assert order[-1] == "add"
