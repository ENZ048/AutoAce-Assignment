"""Mock-level tests for E6: majority-vote harness over E1's gap-listening
question. Motivating facts (see exp6_gap_vote.py module docstring): call_003
static confirmed 3/3 runs in exp1_gap_noise's own session
(out/experiments/exp1_gap_noise_run{1,2,3}.json) but only 1/3 runs in
combined.py's session (out/experiments/combined_run{1,2,3}.json's per-clip
"gap_listening" sub-object) -- same clip, same question, same model, same
temperature, different session. This harness asks the SAME question 3x per
clip per run and majority-votes, to test whether within-session voting
recovers cross-session stability.

No network -- every collaborator exp6 touches is monkeypatched at its
imported name, same convention as test_exp1_gaps.py / test_exp4_flash.py /
test_combined.py (which mocks combined._gap_listen directly for run_once-
level tests, and its collaborators directly for _gap_listen-level tests --
this file mirrors that split: _vote_gap_listening tested directly against
mocked collaborators, run_once tested with _vote_gap_listening mocked)."""

import json

import numpy as np
import pytest
from eval.experiments import common, exp1_gap_noise
from eval.experiments import exp6_gap_vote as exp6

from autoace_audio.config import Settings

# ---------------------------------------------------------------------------
# Import-not-copy identity: exp6 must reuse E1's own question/gap-extraction
# collaborators, not copies -- a copy could silently drift from the shipping
# question. (Brief: "the question text and audio-segment logic must be
# byte-identical to E1's".)
# ---------------------------------------------------------------------------


def test_ask_gemini_gaps_is_the_e1_function_not_a_copy():
    assert exp6.ask_gemini_gaps is exp1_gap_noise.ask_gemini_gaps


def test_concat_gaps_is_the_e1_function_not_a_copy():
    assert exp6.concat_gaps is exp1_gap_noise.concat_gaps


# ---------------------------------------------------------------------------
# majority_vote: the brief's pinned rule, verbatim --
#   present = >=2 of 3 votes true.
#   type = modal normalized (lowercase/strip) string among the present-true
#   votes; if no modal winner (tie), take the vote with higher self-reported
#   noise confidence if the response carries one, else the first true vote.
#   All ties toward absent.
# Both tie shapes matter: a 2-true-vote 1-1 split, and a 3-true-vote 1-1-1
# split.
# ---------------------------------------------------------------------------


def _vote(present: bool, type_: str = "", **extra) -> dict:
    return {"background_noise_present": present, "background_noise_type": type_, **extra}


def test_majority_present_true_and_type_agrees_when_3_of_3_vote_true():
    votes = [_vote(True, "static")] * 3
    assert exp6.majority_vote(votes) == {
        "background_noise_present": True,
        "background_noise_type": "static",
    }


def test_majority_present_true_when_exactly_2_of_3_agree_on_type():
    votes = [_vote(True, "TV"), _vote(True, "tv"), _vote(False, "")]
    assert exp6.majority_vote(votes) == {
        "background_noise_present": True,
        "background_noise_type": "tv",
    }


def test_majority_present_false_when_only_1_of_3_true_ties_toward_absent():
    votes = [_vote(True, "static"), _vote(False, ""), _vote(False, "")]
    assert exp6.majority_vote(votes) == {
        "background_noise_present": False,
        "background_noise_type": "",
    }


def test_majority_present_false_when_0_of_3_true():
    votes = [_vote(False, "")] * 3
    assert exp6.majority_vote(votes) == {
        "background_noise_present": False,
        "background_noise_type": "",
    }


def test_majority_normalizes_type_case_and_whitespace_before_counting():
    # "Static" / " static " / "STATIC" must all land in the SAME modal
    # bucket -- proves normalization happens BEFORE the modal count, not
    # after picking a representative string.
    votes = [_vote(True, "Static"), _vote(True, " static "), _vote(True, "STATIC")]
    result = exp6.majority_vote(votes)
    assert result == {"background_noise_present": True, "background_noise_type": "static"}


def test_majority_type_tie_2true_votes_disagree_no_confidence_takes_first_true_vote():
    # tie shape 1: exactly 2 present=true votes, different normalized types
    # (1-1 split), neither carries a confidence field -> first TRUE vote in
    # original order wins (votes[1], since votes[0] is the false one).
    votes = [_vote(False, ""), _vote(True, "TV"), _vote(True, "static")]
    result = exp6.majority_vote(votes)
    assert result == {"background_noise_present": True, "background_noise_type": "tv"}


def test_majority_type_tie_2true_votes_disagree_confidence_breaks_it():
    # same tie shape, but the LOWER-indexed true vote has LOWER confidence --
    # proves the tiebreak really reads confidence, not just vote order.
    votes = [
        _vote(True, "TV", noise_confidence=0.4),
        _vote(True, "static", noise_confidence=0.9),
        _vote(False, ""),
    ]
    result = exp6.majority_vote(votes)
    assert result == {"background_noise_present": True, "background_noise_type": "static"}


def test_majority_type_tie_1_1_1_three_true_votes_all_disagree_no_confidence():
    # tie shape 2: all 3 votes true, all 3 normalized types different (1-1-1
    # split -- presence is unanimous but there's still no modal type winner)
    # -> first true vote wins when no confidence is present.
    votes = [_vote(True, "TV"), _vote(True, "static"), _vote(True, "hum")]
    result = exp6.majority_vote(votes)
    assert result == {"background_noise_present": True, "background_noise_type": "tv"}


def test_majority_type_tie_1_1_1_confidence_breaks_it():
    votes = [
        _vote(True, "TV", noise_confidence=0.2),
        _vote(True, "static", noise_confidence=0.95),
        _vote(True, "hum", noise_confidence=0.5),
    ]
    result = exp6.majority_vote(votes)
    assert result == {"background_noise_present": True, "background_noise_type": "static"}


def test_majority_type_clear_2_1_modal_winner_is_not_treated_as_a_tie():
    # 2-1 split among 3 true votes IS a clear modal winner -- must not fall
    # through to the confidence/first-vote tiebreak (confidence here would
    # pick "static" if the tiebreak wrongly fired; it must not fire).
    votes = [
        _vote(True, "TV", noise_confidence=0.1),
        _vote(True, "tv", noise_confidence=0.1),
        _vote(True, "static", noise_confidence=0.99),
    ]
    result = exp6.majority_vote(votes)
    assert result == {"background_noise_present": True, "background_noise_type": "tv"}


# ---------------------------------------------------------------------------
# Constants pinned by the brief.
# ---------------------------------------------------------------------------


def test_pinned_constants():
    assert exp6.VOTES_PER_CLIP == 3
    assert pytest.approx(0.02) == exp6.EST_COST_PER_RUN
    assert pytest.approx(0.00146) == exp6.BASELINE_PER_MIN


# ---------------------------------------------------------------------------
# _vote_gap_listening: one clip's full 3-vote round.
# ---------------------------------------------------------------------------


class _FakeAudio:
    def __init__(self, seconds: float, sr: int = 16000):
        self.samples = np.zeros(int(seconds * sr), dtype=np.float32)
        self.sr = sr


def test_vote_gap_listening_casts_exactly_3_votes_against_the_same_encoded_blob(monkeypatch):
    ask_calls: list[bytes] = []

    def _fake_ask(blob):
        ask_calls.append(blob)
        return (
            {
                "background_noise_present": True,
                "background_noise_type": "static",
                "character": "constant",
            },
            0.0002,
            {"in": 300, "out": 25},
        )

    encode_calls: list[int] = []

    def _fake_encode(samples, sr):
        encode_calls.append(samples.size)
        return b"fixed-blob-for-this-clip"

    monkeypatch.setattr(exp6, "load_audio", lambda path: _FakeAudio(seconds=20.0))
    monkeypatch.setattr(exp6, "analyze_vad", lambda samples, sr: object())
    monkeypatch.setattr(
        exp6, "concat_gaps", lambda samples, sr, vad: np.zeros(16000 * 10, dtype=np.float32)
    )
    monkeypatch.setattr(exp6, "encode_opus_ogg", _fake_encode)
    monkeypatch.setattr(exp6, "ask_gemini_gaps", _fake_ask)

    result = exp6._vote_gap_listening("call_003.ogg")

    assert result["skipped"] is False
    assert len(result["votes"]) == 3
    assert len(ask_calls) == 3
    assert len(set(ask_calls)) == 1  # all 3 votes hit the SAME encoded blob
    assert len(encode_calls) == 1  # encoded exactly once, not once per vote
    assert result["majority"] == {
        "background_noise_present": True,
        "background_noise_type": "static",
    }
    assert result["cost_usd"] == pytest.approx(0.0006)
    assert result["tokens"] == {"in": 900, "out": 75}
    assert all(v["tokens"] == {"in": 300, "out": 25} for v in result["votes"])
    assert all(v["cost_usd"] == pytest.approx(0.0002) for v in result["votes"])
    assert result["gap_seconds"] == pytest.approx(10.0)
    assert result["audio_s"] == pytest.approx(20.0)


def test_vote_gap_listening_skips_when_gap_audio_too_short(monkeypatch):
    calls = []

    def _fake_ask(blob):
        calls.append(blob)
        raise AssertionError("must not be called when gap audio is too short")

    monkeypatch.setattr(exp6, "load_audio", lambda path: _FakeAudio(seconds=5.0))
    monkeypatch.setattr(exp6, "analyze_vad", lambda samples, sr: object())
    monkeypatch.setattr(
        exp6, "concat_gaps", lambda samples, sr, vad: np.zeros(1000, dtype=np.float32)
    )
    monkeypatch.setattr(exp6, "ask_gemini_gaps", _fake_ask)

    result = exp6._vote_gap_listening("call_001.ogg")

    assert result["skipped"] is True
    assert result["votes"] == []
    assert result["majority"] is None
    assert result["cost_usd"] == 0.0
    assert result["tokens"] == {"in": 0, "out": 0}
    assert calls == []


# ---------------------------------------------------------------------------
# run_once: aggregation, truth/present_correct wiring, log shape, guard
# ordering. _vote_gap_listening is mocked directly here (same split as
# test_combined.py's run_once tests mocking combined._gap_listen directly).
# ---------------------------------------------------------------------------


class _OrderedGuard:
    def __init__(self):
        self.order: list[tuple[str, float]] = []

    def check(self, projected_usd):
        self.order.append(("check", projected_usd))

    def add(self, cost_usd):
        self.order.append(("add", cost_usd))


def _fake_vote_result(present: bool, type_: str, cost: float, audio_s: float) -> dict:
    per_vote_cost = cost / 3
    votes = [
        {
            "data": {
                "background_noise_present": present,
                "background_noise_type": type_,
                "character": "constant" if present else "none",
            },
            "tokens": {"in": 300, "out": 25},
            "cost_usd": per_vote_cost,
        }
        for _ in range(3)
    ]
    return {
        "skipped": False,
        "gap_seconds": 15.0,
        "audio_s": audio_s,
        "votes": votes,
        "majority": {"background_noise_present": present, "background_noise_type": type_},
        "tokens": {"in": 900, "out": 75},
        "cost_usd": cost,
    }


_TRUTH = {
    "call_001.ogg": {"background_noise_present": False, "background_noise_type": ""},
    "call_002.ogg": {"background_noise_present": True, "background_noise_type": "TV"},
    "call_003.ogg": {"background_noise_present": True, "background_noise_type": "sharp static"},
}


def _fake_settings() -> Settings:
    return Settings(_env_file=None, gemini_api_key="fake-key-for-test")


def test_run_once_builds_truth_present_correct_and_log_shape(monkeypatch, tmp_path):
    results = {
        "call_001.ogg": _fake_vote_result(False, "", cost=0.0003, audio_s=16.1),
        "call_002.ogg": _fake_vote_result(False, "", cost=0.0003, audio_s=34.96),  # known miss
        "call_003.ogg": _fake_vote_result(True, "static", cost=0.0006, audio_s=171.9),
    }
    guard = _OrderedGuard()

    monkeypatch.setattr(exp6, "load_truth", lambda: _TRUTH)
    monkeypatch.setattr(exp6, "_vote_gap_listening", lambda name: results[name])
    monkeypatch.setattr(exp6, "SpendGuard", lambda: guard)
    monkeypatch.setattr(exp6, "get_settings", _fake_settings)
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)

    payload = exp6.run_once(1)

    pc = payload["per_clip"]
    assert pc["call_001.ogg"]["present_correct"] is True  # correctly stays absent
    assert pc["call_001.ogg"]["truth"] == _TRUTH["call_001.ogg"]
    assert pc["call_002.ogg"]["present_correct"] is False  # known miss, unchanged
    assert pc["call_003.ogg"]["present_correct"] is True  # static confirmed by majority
    assert len(pc["call_003.ogg"]["votes"]) == 3

    assert payload["exp"] == "exp6_gap_vote"
    assert payload["run"] == 1
    assert payload["cost_usd"] == pytest.approx(0.0003 + 0.0003 + 0.0006)
    assert payload["votes_per_clip"] == 3
    assert payload["audio_minutes"] == pytest.approx((16.1 + 34.96 + 171.9) / 60.0)
    assert payload["model"] == "gemini-3.1-flash-lite"
    assert payload["pricing"]["in_per_1m"] == common.GEMINI_LITE_IN
    assert payload["pricing"]["out_per_1m"] == common.GEMINI_LITE_OUT
    assert payload["pricing"]["source"]

    # guard consulted BEFORE any spend, charged with the measured total AFTER
    assert guard.order[0] == ("check", exp6.EST_COST_PER_RUN)
    assert guard.order[-1] == ("add", pytest.approx(0.0012))

    on_disk = json.loads((tmp_path / "exp6_gap_vote_run1.json").read_text())
    assert on_disk["per_clip"]["call_003.ogg"]["majority"] == {
        "background_noise_present": True,
        "background_noise_type": "static",
    }
    assert on_disk["cost_usd"] == pytest.approx(0.0012)


def test_run_once_skipped_clip_has_no_present_correct_key(monkeypatch, tmp_path):
    results = {
        "call_001.ogg": {
            "skipped": True,
            "gap_seconds": 1.0,
            "audio_s": 5.0,
            "votes": [],
            "majority": None,
            "tokens": {"in": 0, "out": 0},
            "cost_usd": 0.0,
        },
        "call_002.ogg": _fake_vote_result(False, "", cost=0.0003, audio_s=34.96),
        "call_003.ogg": _fake_vote_result(True, "static", cost=0.0006, audio_s=171.9),
    }

    monkeypatch.setattr(exp6, "load_truth", lambda: _TRUTH)
    monkeypatch.setattr(exp6, "_vote_gap_listening", lambda name: results[name])
    monkeypatch.setattr(exp6, "SpendGuard", lambda: _OrderedGuard())
    monkeypatch.setattr(exp6, "get_settings", _fake_settings)
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)

    payload = exp6.run_once(1)

    assert payload["per_clip"]["call_001.ogg"]["skipped"] is True
    assert "present_correct" not in payload["per_clip"]["call_001.ogg"]
    assert payload["cost_usd"] == pytest.approx(0.0003 + 0.0006)


# ---------------------------------------------------------------------------
# operating_point: the brief's formula -- baseline + 3x measured single-vote
# marginal, computed from this module's own run logs.
# ---------------------------------------------------------------------------


def test_operating_point_matches_briefs_formula():
    runs = [
        {"cost_usd": 0.0011, "audio_minutes": 3.964018},
        {"cost_usd": 0.0012, "audio_minutes": 3.964018},
        {"cost_usd": 0.0010, "audio_minutes": 3.964018},
    ]
    op = exp6.operating_point(runs)

    mean_cost = (0.0011 + 0.0012 + 0.0010) / 3
    expected_voting_marginal = mean_cost / 3.964018
    expected_single_vote = expected_voting_marginal / 3

    assert op["voting_marginal_per_min"] == pytest.approx(expected_voting_marginal)
    assert op["single_vote_marginal_per_min"] == pytest.approx(expected_single_vote)
    assert op["baseline_per_min"] == pytest.approx(0.00146)
    # brief's literal formula: baseline + 3x the single-vote marginal
    assert op["operating_point_per_min"] == pytest.approx(0.00146 + 3 * expected_single_vote)
    # identity: 3x the single-vote marginal == the full measured voting marginal
    assert op["operating_point_per_min"] == pytest.approx(0.00146 + expected_voting_marginal)


def test_operating_point_raises_on_empty_runs():
    with pytest.raises(ValueError, match="no runs"):
        exp6.operating_point([])


def test_operating_point_raises_when_audio_minutes_disagree_across_runs():
    runs = [
        {"cost_usd": 0.001, "audio_minutes": 3.964018},
        {"cost_usd": 0.001, "audio_minutes": 5.0},
    ]
    with pytest.raises(ValueError, match="audio_minutes"):
        exp6.operating_point(runs)
