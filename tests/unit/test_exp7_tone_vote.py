"""Mock-level tests for E7: self-consistency majority-vote harness over the
shipping tone/intensity classification (autoace_audio.analyzers.tone.
gemini_tone). Brief's question: does classic self-consistency (3 samples at
diversity temperature, majority vote) beat the shipping single greedy call
on tone/intensity? Baseline tone at temp 0.1 is run-deterministic
(out/experiments/exp0_baseline_run{1,2,3}.json agree byte-for-byte on every
clip's emotional_tone/emotional_intensity), so voting at 0.1 would trivially
null -- the experiment is voting at temperature 0.7 (pinned, the point of
the test) vs the temp-0.1 single-call baseline.

No network -- every collaborator exp7 touches is monkeypatched at its
imported name, same convention as test_exp4_flash.py / test_exp6_gap_vote.py
(google.genai.types stays real -- pure data construction, no network; only
google.genai.Client is faked)."""

import json

import numpy as np
import pytest
from eval.experiments import common
from eval.experiments import exp7_tone_vote as exp7

from autoace_audio.analyzers.tone import gemini_tone
from autoace_audio.config import Settings

# ---------------------------------------------------------------------------
# Import-not-copy identity: exp7 must reuse the shipping arm's own prompt
# builder + response schema, not copies -- a copy could silently drift from
# the shipping contract. (Brief: "Import build_prompt + GEMINI_RESPONSE_
# SCHEMA from the shipping arm (identity-asserted in tests)".)
# ---------------------------------------------------------------------------


def test_build_prompt_is_the_shipping_function_not_a_copy():
    assert exp7.build_prompt is gemini_tone.build_prompt


def test_response_schema_is_the_shipping_schema_not_a_copy():
    assert exp7.GEMINI_RESPONSE_SCHEMA is gemini_tone.GEMINI_RESPONSE_SCHEMA


# ---------------------------------------------------------------------------
# Pinned constants + the baseline-temperature sanity guard: E7 hardcodes its
# own diversity temperature and must never read (or accidentally mutate the
# meaning of) the shipping default.
# ---------------------------------------------------------------------------


def test_pinned_constants():
    assert exp7.VOTES_PER_CLIP == 3
    assert pytest.approx(0.7) == exp7.VOTE_TEMPERATURE
    assert pytest.approx(0.02) == exp7.EST_COST_PER_RUN
    assert pytest.approx(0.00146) == exp7.BASELINE_PER_MIN


def test_baseline_gemini_temperature_setting_is_unchanged_at_0_1():
    # The shipping default (exp0's baseline config) must still be 0.1 -- E7
    # hardcodes VOTE_TEMPERATURE=0.7 explicitly rather than reading
    # settings.gemini_temperature, specifically SO THAT voting at 0.7 stays
    # a real contrast against the (still 0.1) single-call baseline. If a
    # future edit silently changed the shipping default off 0.1, this
    # module's whole "voting at 0.1 would trivially null" premise would
    # quietly stop being true without this guard.
    s = Settings(_env_file=None, gemini_api_key="fake-key-for-test")
    assert s.gemini_temperature == 0.1
    assert s.gemini_temperature != exp7.VOTE_TEMPERATURE


# ---------------------------------------------------------------------------
# _dispersion_shape: pure agreement-shape classifier ("3-same"/"2-1"/
# "3-way"), independent of confidence -- confidence only ever decides a
# 3-way tie's WINNER, never the dispersion label itself.
# ---------------------------------------------------------------------------


def test_dispersion_shape_all_three_agree():
    assert exp7._dispersion_shape(["upset", "upset", "upset"]) == "3-same"


def test_dispersion_shape_two_agree_one_dissents():
    assert exp7._dispersion_shape(["upset", "frustrated", "upset"]) == "2-1"
    # position of the dissenter must not matter
    assert exp7._dispersion_shape(["frustrated", "upset", "upset"]) == "2-1"
    assert exp7._dispersion_shape(["upset", "upset", "frustrated"]) == "2-1"


def test_dispersion_shape_all_three_disagree():
    assert exp7._dispersion_shape(["upset", "frustrated", "neutral"]) == "3-way"


# ---------------------------------------------------------------------------
# majority_vote: the brief's pinned rule, applied INDEPENDENTLY to
# emotional_tone and emotional_intensity -- winner = value shared by >=2 of
# 3; 3-way split (all different) -> highest tone_confidence; further tie
# (equal top confidence) -> first vote (original order).
# ---------------------------------------------------------------------------


def _vote(tone: str, intensity: str, confidence: float) -> dict:
    return {
        "emotional_tone": tone,
        "emotional_intensity": intensity,
        "tone_confidence": confidence,
    }


def test_majority_vote_all_three_agree_on_both_fields():
    votes = [_vote("upset", "high", 0.9)] * 3
    result = exp7.majority_vote(votes)
    assert result["emotional_tone"] == "upset"
    assert result["emotional_intensity"] == "high"
    assert result["dispersion"] == {
        "emotional_tone": "3-same",
        "emotional_intensity": "3-same",
    }


def test_majority_vote_clean_2_1_split_picks_the_majority_value_not_the_first_vote():
    # the FIRST vote is the dissenter ("frustrated") -- proves the winner
    # comes from counting votes, not from vote order, when there IS a real
    # majority.
    votes = [
        _vote("frustrated", "medium", 0.5),
        _vote("upset", "medium", 0.5),
        _vote("upset", "medium", 0.5),
    ]
    result = exp7.majority_vote(votes)
    assert result["emotional_tone"] == "upset"
    assert result["dispersion"]["emotional_tone"] == "2-1"


def test_majority_vote_3way_tie_breaks_on_highest_tone_confidence():
    votes = [
        _vote("neutral", "low", 0.2),
        _vote("upset", "low", 0.95),  # highest confidence -> wins
        _vote("frustrated", "low", 0.5),
    ]
    result = exp7.majority_vote(votes)
    assert result["emotional_tone"] == "upset"
    assert result["dispersion"]["emotional_tone"] == "3-way"


def test_majority_vote_3way_tie_confidence_tiebreak_is_not_just_first_index():
    # the highest-confidence vote sits at index 1, NOT index 0 -- proves the
    # tiebreak really reads confidence rather than silently defaulting to
    # vote order.
    votes = [
        _vote("neutral", "high", 0.1),
        _vote("upset", "high", 0.99),
        _vote("frustrated", "high", 0.4),
    ]
    result = exp7.majority_vote(votes)
    assert result["emotional_tone"] == "upset"


def test_majority_vote_3way_tie_with_equal_top_confidence_takes_first_vote():
    # votes[0] and votes[1] are TIED at the top confidence (0.9); votes[0]
    # must win because it comes first in original call order. votes[2] has
    # lower confidence so it is never a contender.
    votes = [
        _vote("upset", "high", 0.9),
        _vote("frustrated", "high", 0.9),
        _vote("neutral", "high", 0.3),
    ]
    result = exp7.majority_vote(votes)
    assert result["emotional_tone"] == "upset"


def test_majority_vote_clear_2_1_is_not_treated_as_a_tie_even_with_misleading_confidence():
    # a 2-1 split IS a clear majority winner -- must not fall through to the
    # confidence tiebreak (confidence here would pick "distressed" if the
    # tiebreak wrongly fired; it must not fire).
    votes = [
        _vote("upset", "high", 0.1),
        _vote("upset", "high", 0.2),
        _vote("distressed", "high", 0.99),
    ]
    result = exp7.majority_vote(votes)
    assert result["emotional_tone"] == "upset"
    assert result["dispersion"]["emotional_tone"] == "2-1"


def test_majority_vote_fields_vote_independently():
    # tone is a clean 2-1 (no tiebreak needed); intensity is a 3-way split
    # in the SAME 3 votes (tiebreak needed) -- each field's verdict must
    # come from ITS OWN value distribution, not be coupled to the other
    # field's shape.
    votes = [
        _vote("upset", "low", 0.3),
        _vote("upset", "medium", 0.9),  # highest confidence -> wins intensity
        _vote("frustrated", "high", 0.5),
    ]
    result = exp7.majority_vote(votes)
    assert result["emotional_tone"] == "upset"
    assert result["emotional_intensity"] == "medium"
    assert result["dispersion"] == {
        "emotional_tone": "2-1",
        "emotional_intensity": "3-way",
    }


# ---------------------------------------------------------------------------
# _ask_gemini_tone_vote: what actually gets sent to (a faked) Gemini client,
# and what it returns. Client is faked; google.genai.types stays real (pure
# data construction, no network) -- same convention as test_exp4_flash.py.
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, in_tok, out_tok):
        self.prompt_token_count = in_tok
        self.candidates_token_count = out_tok


class _FakeResp:
    def __init__(self, data, in_tok, out_tok):
        self.text = json.dumps(data)
        self.usage_metadata = _FakeUsage(in_tok, out_tok)


_FULL_DATA = {
    "emotional_tone": "neutral",
    "emotional_intensity": "medium",
    "tone_confidence": 0.75,
}


def _fake_settings(temperature: float = 0.1, model: str = "gemini-3.1-flash-lite") -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key="fake-key-for-test",
        gemini_model=model,
        gemini_temperature=temperature,
    )


def _patch_gemini_client(monkeypatch, resp):
    """Fakes only google.genai.Client (no network); returns the list its
    fake models.generate_content() call will record kwargs into.
    google.genai.types stays real -- GenerateContentConfig/Part construction
    is pure data assembly, matching test_exp4_flash.py's convention."""
    calls: list[dict] = []

    class _FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return resp

    class _FakeGenaiClient:
        def __init__(self, api_key=None):
            self.models = _FakeModels()

    monkeypatch.setattr("google.genai.Client", _FakeGenaiClient)
    return calls


def test_ask_gemini_tone_vote_sends_vote_temperature_not_the_settings_temperature(monkeypatch):
    # settings' OWN gemini_temperature is deliberately the real shipping
    # default (0.1, NOT 0.7) -- proves VOTE_TEMPERATURE is hardcoded into
    # the call, never read from settings.
    resp = _FakeResp(_FULL_DATA, 2000, 120)
    calls = _patch_gemini_client(monkeypatch, resp)
    monkeypatch.setattr(exp7, "get_settings", lambda: _fake_settings(temperature=0.1))

    exp7._ask_gemini_tone_vote(b"fake-blob", "fake-prompt")

    assert calls[0]["config"].temperature == pytest.approx(0.7)


def test_ask_gemini_tone_vote_sends_the_shipping_model_id_from_settings(monkeypatch):
    resp = _FakeResp(_FULL_DATA, 2000, 120)
    calls = _patch_gemini_client(monkeypatch, resp)
    monkeypatch.setattr(exp7, "get_settings", lambda: _fake_settings(model="gemini-3.1-flash-lite"))

    exp7._ask_gemini_tone_vote(b"fake-blob", "fake-prompt")

    assert calls[0]["model"] == "gemini-3.1-flash-lite"


def test_ask_gemini_tone_vote_sends_the_exact_prompt_it_was_given_and_the_shipping_schema(
    monkeypatch,
):
    resp = _FakeResp(_FULL_DATA, 2000, 120)
    calls = _patch_gemini_client(monkeypatch, resp)
    monkeypatch.setattr(exp7, "get_settings", lambda: _fake_settings())

    prompt = gemini_tone.build_prompt(42.0, 12.5, 0.6)
    exp7._ask_gemini_tone_vote(b"fake-blob-bytes", prompt)

    assert calls[0]["contents"][1] == prompt
    assert calls[0]["contents"][0].inline_data.mime_type == "audio/ogg"
    assert calls[0]["contents"][0].inline_data.data == b"fake-blob-bytes"
    # response_schema round-trips through a real pydantic model, so identity
    # is not preserved end to end -- but the shipping schema's own identity
    # is already proven above (test_response_schema_is_the_shipping_schema_
    # not_a_copy); this proves the exact value sent over the wire matches it.
    assert calls[0]["config"].response_schema == gemini_tone.GEMINI_RESPONSE_SCHEMA
    assert calls[0]["config"].response_mime_type == "application/json"


def test_ask_gemini_tone_vote_parses_data_tokens_and_cost(monkeypatch):
    resp = _FakeResp(_FULL_DATA, 2000, 120)
    _patch_gemini_client(monkeypatch, resp)
    monkeypatch.setattr(exp7, "get_settings", lambda: _fake_settings())

    data, cost, tokens = exp7._ask_gemini_tone_vote(b"fake-blob", "fake-prompt")

    assert data == _FULL_DATA
    assert tokens == {"in": 2000, "out": 120}
    assert cost == pytest.approx((2000 * 0.50 + 120 * 1.50) / 1e6)


def test_ask_gemini_tone_vote_handles_missing_usage_metadata(monkeypatch):
    resp = _FakeResp(_FULL_DATA, None, None)
    _patch_gemini_client(monkeypatch, resp)
    monkeypatch.setattr(exp7, "get_settings", lambda: _fake_settings())

    data, cost, tokens = exp7._ask_gemini_tone_vote(b"fake-blob", "fake-prompt")

    assert tokens == {"in": None, "out": None}
    assert cost == 0.0


# ---------------------------------------------------------------------------
# _vote_tone: one clip's full 3-vote round. Audio load/VAD/noise/encode and
# prompt construction happen ONCE (deterministic -- would add cost, not
# signal, if repeated per vote); only the Gemini call itself repeats
# VOTES_PER_CLIP times against that SAME blob + SAME prompt.
# ---------------------------------------------------------------------------


class _FakeAudio:
    def __init__(self, seconds: float, sr: int = 16000):
        self.samples = np.zeros(int(seconds * sr), dtype=np.float32)
        self.sr = sr


class _FakeVad:
    speech_ratio = 0.6


class _FakeNoise:
    snr_db = 12.5


def test_vote_tone_casts_exactly_3_votes_against_the_same_blob_and_prompt(monkeypatch):
    ask_calls: list[tuple] = []

    def _fake_ask(blob, prompt):
        ask_calls.append((blob, prompt))
        return (
            {"emotional_tone": "upset", "emotional_intensity": "high", "tone_confidence": 0.8},
            0.0009,
            {"in": 1500, "out": 100},
        )

    encode_calls: list[int] = []

    def _fake_encode(samples, sr):
        encode_calls.append(samples.size)
        return b"fixed-blob-for-this-clip"

    prompt_calls: list[tuple] = []

    def _fake_build_prompt(duration_s, snr_db, speech_ratio):
        prompt_calls.append((duration_s, snr_db, speech_ratio))
        return "the-one-true-prompt"

    monkeypatch.setattr(exp7, "load_audio", lambda path: _FakeAudio(seconds=20.0))
    monkeypatch.setattr(exp7, "analyze_vad", lambda samples, sr: _FakeVad())
    monkeypatch.setattr(exp7, "analyze_noise", lambda samples, sr, vad: _FakeNoise())
    monkeypatch.setattr(exp7, "encode_opus_ogg", _fake_encode)
    monkeypatch.setattr(exp7, "build_prompt", _fake_build_prompt)
    monkeypatch.setattr(exp7, "_ask_gemini_tone_vote", _fake_ask)

    result = exp7._vote_tone("call_001.ogg")

    assert len(result["votes"]) == 3
    assert len(ask_calls) == 3
    assert len({c[0] for c in ask_calls}) == 1  # every vote hit the SAME blob
    assert len({c[1] for c in ask_calls}) == 1  # every vote used the SAME prompt
    assert len(encode_calls) == 1  # encoded exactly once, not once per vote
    assert len(prompt_calls) == 1  # prompt built exactly once, not once per vote
    assert result["majority"]["emotional_tone"] == "upset"
    assert result["majority"]["emotional_intensity"] == "high"
    assert result["cost_usd"] == pytest.approx(0.0027)
    assert result["tokens"] == {"in": 4500, "out": 300}
    assert result["audio_s"] == pytest.approx(20.0)
    assert all(v["tone_confidence"] == 0.8 for v in result["votes"])
    assert all(v["tokens"] == {"in": 1500, "out": 100} for v in result["votes"])
    assert all(v["cost_usd"] == pytest.approx(0.0009) for v in result["votes"])


# ---------------------------------------------------------------------------
# run_once: aggregation, truth/correct wiring, dispersion in the log,
# log shape, guard ordering. _vote_tone is mocked directly here (same split
# as test_exp6_gap_vote.py's run_once tests mocking _vote_gap_listening).
# ---------------------------------------------------------------------------


class _OrderedGuard:
    def __init__(self):
        self.order: list[tuple[str, float]] = []

    def check(self, projected_usd):
        self.order.append(("check", projected_usd))

    def add(self, cost_usd):
        self.order.append(("add", cost_usd))


def _fake_vote_result(
    tone: str, intensity: str, cost: float, audio_s: float, dispersion=None
) -> dict:
    dispersion = dispersion or {"emotional_tone": "3-same", "emotional_intensity": "3-same"}
    per_vote_cost = cost / 3
    votes = [
        {
            "emotional_tone": tone,
            "emotional_intensity": intensity,
            "tone_confidence": 0.8,
            "tokens": {"in": 1500, "out": 100},
            "cost_usd": per_vote_cost,
        }
        for _ in range(3)
    ]
    return {
        "audio_s": audio_s,
        "votes": votes,
        "majority": {
            "emotional_tone": tone,
            "emotional_intensity": intensity,
            "dispersion": dispersion,
        },
        "tokens": {"in": 4500, "out": 300},
        "cost_usd": cost,
    }


_TRUTH = {
    "call_001.ogg": {"emotional_tone": "upset", "emotional_intensity": "high"},
    "call_002.ogg": {"emotional_tone": "neutral", "emotional_intensity": "medium"},
    "call_003.ogg": {"emotional_tone": "satisfied", "emotional_intensity": "medium"},
}


def test_run_once_builds_pred_correct_dispersion_and_log_shape(monkeypatch, tmp_path):
    results = {
        # known intensity miss (matches exp0's real one-band-low pattern)
        "call_001.ogg": _fake_vote_result("upset", "medium", cost=0.0027, audio_s=16.1),
        # known tone miss (matches exp0's real frustrated-vs-neutral pattern)
        "call_002.ogg": _fake_vote_result(
            "frustrated",
            "medium",
            cost=0.0029,
            audio_s=34.96,
            dispersion={"emotional_tone": "2-1", "emotional_intensity": "3-way"},
        ),
        "call_003.ogg": _fake_vote_result("satisfied", "medium", cost=0.008, audio_s=171.9),
    }
    guard = _OrderedGuard()

    monkeypatch.setattr(exp7, "load_truth", lambda: _TRUTH)
    monkeypatch.setattr(exp7, "_vote_tone", lambda name: results[name])
    monkeypatch.setattr(exp7, "SpendGuard", lambda: guard)
    monkeypatch.setattr(exp7, "get_settings", lambda: _fake_settings())
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)

    payload = exp7.run_once(1)

    pc = payload["per_clip"]
    assert pc["call_001.ogg"]["correct"] == {"emotional_tone": True, "emotional_intensity": False}
    assert pc["call_002.ogg"]["correct"] == {"emotional_tone": False, "emotional_intensity": True}
    assert pc["call_003.ogg"]["correct"] == {"emotional_tone": True, "emotional_intensity": True}
    assert pc["call_002.ogg"]["dispersion"] == {
        "emotional_tone": "2-1",
        "emotional_intensity": "3-way",
    }
    assert pc["call_003.ogg"]["dispersion"] == {
        "emotional_tone": "3-same",
        "emotional_intensity": "3-same",
    }
    assert len(pc["call_001.ogg"]["votes"]) == 3

    assert payload["exp"] == "exp7_tone_vote"
    assert payload["run"] == 1
    assert payload["cost_usd"] == pytest.approx(0.0027 + 0.0029 + 0.008)
    assert payload["votes_per_clip"] == 3
    assert payload["vote_temperature"] == pytest.approx(0.7)
    assert payload["audio_minutes"] == pytest.approx((16.1 + 34.96 + 171.9) / 60.0)
    assert payload["model"] == "gemini-3.1-flash-lite"
    assert payload["pricing"]["in_per_1m"] == common.GEMINI_LITE_IN
    assert payload["pricing"]["out_per_1m"] == common.GEMINI_LITE_OUT
    assert payload["pricing"]["source"]

    # guard consulted BEFORE any spend, charged with the measured total AFTER
    assert guard.order[0] == ("check", exp7.EST_COST_PER_RUN)
    assert guard.order[-1] == ("add", pytest.approx(0.0027 + 0.0029 + 0.008))

    on_disk = json.loads((tmp_path / "exp7_tone_vote_run1.json").read_text())
    assert on_disk["per_clip"]["call_002.ogg"]["correct"]["emotional_tone"] is False
    assert on_disk["per_clip"]["call_002.ogg"]["dispersion"]["emotional_intensity"] == "3-way"
    assert on_disk["cost_usd"] == pytest.approx(0.0027 + 0.0029 + 0.008)


def test_run_once_uses_the_shipping_model_from_settings_not_a_hardcoded_literal(monkeypatch):
    results = {
        name: _fake_vote_result("upset", "high", cost=0.001, audio_s=10.0)
        for name in common.ANCHORS
    }
    monkeypatch.setattr(exp7, "load_truth", lambda: _TRUTH)
    monkeypatch.setattr(exp7, "_vote_tone", lambda name: results[name])
    monkeypatch.setattr(exp7, "SpendGuard", lambda: _OrderedGuard())
    monkeypatch.setattr(exp7, "get_settings", lambda: _fake_settings(model="a-different-model-id"))
    monkeypatch.setattr(common, "OUT_DIR", None)  # must not be touched below
    monkeypatch.setattr(exp7, "log_run", lambda *a, **k: None)

    payload = exp7.run_once(1)

    assert payload["model"] == "a-different-model-id"


# ---------------------------------------------------------------------------
# cost_per_min: E7's own measured $/audio-min for the full voted-tone
# configuration (voting REPLACES the shipping arm's only billed step, so
# unlike E6 there is no additive marginal formula -- the measured mean run
# cost over audio-minutes IS the number).
# ---------------------------------------------------------------------------


def test_cost_per_min_matches_measured_mean_over_audio_minutes():
    runs = [
        {"cost_usd": 0.0090, "audio_minutes": 3.9637},
        {"cost_usd": 0.0092, "audio_minutes": 3.9637},
        {"cost_usd": 0.0088, "audio_minutes": 3.9637},
    ]
    result = exp7.cost_per_min(runs)
    mean_cost = (0.0090 + 0.0092 + 0.0088) / 3
    assert result["mean_run_cost_usd"] == pytest.approx(mean_cost)
    assert result["audio_minutes"] == pytest.approx(3.9637)
    assert result["voted_tone_per_min"] == pytest.approx(mean_cost / 3.9637)
    assert result["baseline_per_min"] == pytest.approx(0.00146)
    assert result["multiple_of_baseline"] == pytest.approx((mean_cost / 3.9637) / 0.00146)


def test_cost_per_min_raises_on_empty_runs():
    with pytest.raises(ValueError, match="no runs"):
        exp7.cost_per_min([])


def test_cost_per_min_raises_when_audio_minutes_disagree_across_runs():
    runs = [
        {"cost_usd": 0.001, "audio_minutes": 3.9637},
        {"cost_usd": 0.001, "audio_minutes": 5.0},
    ]
    with pytest.raises(ValueError, match="audio_minutes"):
        exp7.cost_per_min(runs)


# ---------------------------------------------------------------------------
# dispersion_summary: tallies per-clip dispersion shapes across runs,
# independently per field -- feeds the report's dispersion table.
# ---------------------------------------------------------------------------


def test_dispersion_summary_tallies_shapes_per_field_across_runs():
    runs = [
        {
            "per_clip": {
                "call_001.ogg": {
                    "dispersion": {"emotional_tone": "3-same", "emotional_intensity": "2-1"}
                },
                "call_002.ogg": {
                    "dispersion": {"emotional_tone": "3-way", "emotional_intensity": "3-same"}
                },
                "call_003.ogg": {
                    "dispersion": {"emotional_tone": "2-1", "emotional_intensity": "3-same"}
                },
            }
        },
        {
            "per_clip": {
                "call_001.ogg": {
                    "dispersion": {"emotional_tone": "3-same", "emotional_intensity": "3-same"}
                },
                "call_002.ogg": {
                    "dispersion": {"emotional_tone": "2-1", "emotional_intensity": "2-1"}
                },
                "call_003.ogg": {
                    "dispersion": {"emotional_tone": "3-same", "emotional_intensity": "3-same"}
                },
            }
        },
    ]
    summary = exp7.dispersion_summary(runs)
    assert summary["emotional_tone"]["3-same"] == 3
    assert summary["emotional_tone"]["2-1"] == 2
    assert summary["emotional_tone"]["3-way"] == 1
    assert summary["emotional_intensity"]["3-same"] == 4
    assert summary["emotional_intensity"]["2-1"] == 2
    assert summary["emotional_intensity"]["3-way"] == 0
