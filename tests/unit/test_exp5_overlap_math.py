"""Mock-level and pure-math tests for exp5: Deepgram diarization overlap +
customer-only dimensional bonus.

First section is brief-verbatim (study-task-7-brief.md Step 1): pure overlap
math from synthetic word/turn fixtures -- turns_from_words +
overlap_from_turns, exactly as given.

Everything below that is additional TDD coverage required by Task 7's Job
item 1 (cumulative-threshold + back-channel-exclusion edge cases, the
customer-speaker selection heuristic, and run_once's log shape), plus tests
for a diagnostic overlap_spans() helper added for the report's "derived
overlap spans" requirement -- kept as a SEPARATE function so the brief's
pinned overlap_from_turns is never touched by it. No network, no local
data/ dependency, no model loads -- every collaborator run_once touches is
monkeypatched at its imported name (same convention as test_exp4_flash.py).
"""

import json

import numpy as np
import pytest
from eval.experiments import common
from eval.experiments import exp5_overlap as exp5

from autoace_audio.analyzers.tone import dimensional
from autoace_audio.analyzers.tone.base import ToneResult
from autoace_audio.schema import EmotionalIntensity, EmotionalTone

# ---------------------------------------------------------------------------
# Step 1 (brief-verbatim): pure overlap math
# ---------------------------------------------------------------------------


def _w(word, start, end, speaker):
    return {"word": word, "start": start, "end": end, "speaker": speaker}


def test_turns_merge_consecutive_same_speaker_words():
    words = [_w("hi", 0.0, 0.3, 0), _w("there", 0.35, 0.6, 0), _w("yes", 2.0, 2.2, 1)]
    turns = exp5.turns_from_words(words)
    assert len(turns) == 2
    assert turns[0] == {"speaker": 0, "start": 0.0, "end": 0.6, "words": 2}
    assert turns[1]["speaker"] == 1


def test_turns_split_on_long_gap_same_speaker():
    words = [_w("a", 0.0, 0.2, 0), _w("b", 3.0, 3.2, 0)]
    assert len(exp5.turns_from_words(words)) == 2


def test_overlap_detects_real_crosstalk():
    turns = [
        {"speaker": 0, "start": 0.0, "end": 5.0, "words": 12},
        {"speaker": 1, "start": 4.0, "end": 8.0, "words": 9},  # 1.0s intersection
    ]
    assert exp5.overlap_from_turns(turns) is True


def test_overlap_ignores_backchannel():
    turns = [
        {"speaker": 0, "start": 0.0, "end": 5.0, "words": 12},
        {"speaker": 1, "start": 4.2, "end": 4.9, "words": 1},  # short "uh-huh"
    ]
    assert exp5.overlap_from_turns(turns) is False


def test_overlap_ignores_sub_threshold_intersection():
    turns = [
        {"speaker": 0, "start": 0.0, "end": 5.0, "words": 12},
        {"speaker": 1, "start": 4.8, "end": 8.0, "words": 10},  # 0.2s graze
    ]
    assert exp5.overlap_from_turns(turns) is False


# ---------------------------------------------------------------------------
# Additional pure-math edge cases (Task 7 Job item 1: "cover the
# cumulative-threshold and back-channel-exclusion rules exactly as the
# brief defines them" -- boundary values + multi-pair scanning the brief's
# own 5 tests don't exercise).
# ---------------------------------------------------------------------------


def test_turns_from_words_merges_on_exact_gap_boundary():
    # gap == max_intra_gap_s (0.5s default) must still merge: rule is "<=".
    words = [_w("a", 0.0, 0.2, 0), _w("b", 0.7, 0.9, 0)]
    turns = exp5.turns_from_words(words)
    assert turns == [{"speaker": 0, "start": 0.0, "end": 0.9, "words": 2}]


def test_turns_from_words_honors_custom_gap_param():
    words = [_w("a", 0.0, 0.2, 0), _w("b", 0.5, 0.7, 0)]
    assert len(exp5.turns_from_words(words, max_intra_gap_s=0.2)) == 2
    assert len(exp5.turns_from_words(words, max_intra_gap_s=0.5)) == 1


def test_overlap_exact_threshold_intersection_counts():
    # inter == min_overlap_s (0.5s default): exclusion rule is "<", so
    # exactly-at-threshold must still count as overlap.
    turns = [
        {"speaker": 0, "start": 0.0, "end": 5.0, "words": 12},
        {"speaker": 1, "start": 4.5, "end": 8.0, "words": 9},  # exactly 0.5s
    ]
    assert exp5.overlap_from_turns(turns) is True


def test_overlap_same_speaker_never_counts_regardless_of_intersection():
    turns = [
        {"speaker": 0, "start": 0.0, "end": 5.0, "words": 12},
        {"speaker": 0, "start": 3.0, "end": 8.0, "words": 10},
    ]
    assert exp5.overlap_from_turns(turns) is False


def test_overlap_excluded_when_either_side_is_a_backchannel_not_just_both():
    # short duration AND low word count on just ONE side excludes the pair
    # (brief's for/else: any backchannel side breaks before returning True).
    turns = [
        {"speaker": 0, "start": 4.0, "end": 9.0, "words": 20},  # long, real
        {"speaker": 1, "start": 4.3, "end": 5.0, "words": 2},  # 0.7s/2w backchannel
    ]
    assert exp5.overlap_from_turns(turns) is False


def test_overlap_not_excluded_when_short_turn_has_too_many_words():
    # short duration alone is NOT sufficient for backchannel status -- the
    # AND also requires <= backchannel_max_words; 3 words in 0.7s is fast
    # real speech, not a bare "uh-huh".
    turns = [
        {"speaker": 0, "start": 0.0, "end": 5.0, "words": 12},
        {"speaker": 1, "start": 4.3, "end": 5.0, "words": 3},
    ]
    assert exp5.overlap_from_turns(turns) is True


def test_overlap_scans_past_an_excluded_backchannel_to_a_later_real_pair():
    # A backchannel pair earlier in the scan must not short-circuit the
    # whole function -- a later qualifying pair against the SAME reference
    # turn must still be found.
    turns = [
        {"speaker": 0, "start": 0.0, "end": 10.0, "words": 30},
        {"speaker": 1, "start": 2.0, "end": 2.6, "words": 1},  # backchannel, excluded
        {"speaker": 1, "start": 6.0, "end": 8.0, "words": 15},  # real crosstalk
    ]
    assert exp5.overlap_from_turns(turns) is True


# ---------------------------------------------------------------------------
# overlap_spans: diagnostic-only helper (feeds the run log + report's
# "derived overlap spans"). Independent implementation from
# overlap_from_turns (that function is brief-pinned and untouched here) --
# cross-checked for agreement on every fixture above so the two can never
# silently disagree.
# ---------------------------------------------------------------------------

_OVERLAP_FIXTURES = [
    (
        [
            {"speaker": 0, "start": 0.0, "end": 5.0, "words": 12},
            {"speaker": 1, "start": 4.0, "end": 8.0, "words": 9},
        ],
        True,
    ),
    (
        [
            {"speaker": 0, "start": 0.0, "end": 5.0, "words": 12},
            {"speaker": 1, "start": 4.2, "end": 4.9, "words": 1},
        ],
        False,
    ),
    (
        [
            {"speaker": 0, "start": 0.0, "end": 5.0, "words": 12},
            {"speaker": 1, "start": 4.8, "end": 8.0, "words": 10},
        ],
        False,
    ),
    (
        [
            {"speaker": 0, "start": 0.0, "end": 10.0, "words": 30},
            {"speaker": 1, "start": 2.0, "end": 2.6, "words": 1},
            {"speaker": 1, "start": 6.0, "end": 8.0, "words": 15},
        ],
        True,
    ),
]


@pytest.mark.parametrize("turns,expected_bool", _OVERLAP_FIXTURES)
def test_overlap_spans_agrees_with_overlap_from_turns(turns, expected_bool):
    spans = exp5.overlap_spans(turns)
    assert bool(spans) == expected_bool == exp5.overlap_from_turns(turns)


def test_overlap_spans_reports_the_intersecting_window():
    turns = [
        {"speaker": 0, "start": 0.0, "end": 5.0, "words": 12},
        {"speaker": 1, "start": 4.0, "end": 8.0, "words": 9},
    ]
    spans = exp5.overlap_spans(turns)
    assert spans == [{"speakers": [0, 1], "start": 4.0, "end": 5.0, "intersection_s": 1.0}]


# ---------------------------------------------------------------------------
# customer_only_audio: speaker-selection heuristic (pure function, no
# mocking -- samples are np.arange so slices are exact-value-checkable).
# ---------------------------------------------------------------------------


def test_customer_only_audio_slices_the_non_agent_speakers_turns():
    sr = 16000
    samples = np.arange(3 * sr, dtype=np.float32)  # 3s of distinct sample values
    turns = [
        {"speaker": 0, "start": 0.0, "end": 1.0, "words": 5},  # agent (speaks first)
        {"speaker": 1, "start": 1.2, "end": 2.0, "words": 4},  # customer
    ]
    cust, cust_id, note = exp5.customer_only_audio(samples, sr, turns)
    assert cust_id == 1
    assert note == "first-turn=agent rule"
    expected = samples[int(1.2 * sr) : int(2.0 * sr)]
    assert np.array_equal(cust, expected)


def test_customer_only_audio_agent_is_whoever_speaks_first_not_hardcoded_id0():
    sr = 16000
    samples = np.arange(2 * sr, dtype=np.float32)
    turns = [
        {"speaker": 1, "start": 0.0, "end": 1.0, "words": 5},  # agent == speaker 1 here
        {"speaker": 0, "start": 1.2, "end": 1.8, "words": 4},  # customer == speaker 0
    ]
    cust, cust_id, note = exp5.customer_only_audio(samples, sr, turns)
    assert cust_id == 0
    expected = samples[int(1.2 * sr) : int(1.8 * sr)]
    assert np.array_equal(cust, expected)


def test_customer_only_audio_concatenates_multiple_customer_turns_in_order():
    sr = 16000
    samples = np.arange(6 * sr, dtype=np.float32)
    turns = [
        {"speaker": 0, "start": 0.0, "end": 1.0, "words": 5},
        {"speaker": 1, "start": 1.0, "end": 2.0, "words": 4},
        {"speaker": 0, "start": 2.0, "end": 3.0, "words": 5},
        {"speaker": 1, "start": 3.0, "end": 4.0, "words": 4},
    ]
    cust, cust_id, _ = exp5.customer_only_audio(samples, sr, turns)
    expected = np.concatenate([samples[1 * sr : 2 * sr], samples[3 * sr : 4 * sr]])
    assert cust_id == 1
    assert np.array_equal(cust, expected)


def test_customer_only_audio_falls_back_to_full_audio_when_no_turns():
    samples = np.arange(1000, dtype=np.float32)
    cust, cust_id, note = exp5.customer_only_audio(samples, 16000, [])
    assert cust_id == -1
    assert "no turns" in note
    assert np.array_equal(cust, samples)


def test_customer_only_audio_falls_back_to_full_audio_when_single_speaker():
    samples = np.arange(1000, dtype=np.float32)
    turns = [
        {"speaker": 0, "start": 0.0, "end": 0.5, "words": 3},
        {"speaker": 0, "start": 0.6, "end": 1.0, "words": 3},
    ]
    cust, cust_id, note = exp5.customer_only_audio(samples, 16000, turns)
    assert cust_id == -1
    assert "single speaker" in note
    assert np.array_equal(cust, samples)


# ---------------------------------------------------------------------------
# Pricing + model-id constants: verified-at-run-time, named, sourced, dated
# (mirrors exp4's pricing-constant test convention).
# ---------------------------------------------------------------------------


def test_deepgram_model_id_is_embedded_in_the_request_url_not_duplicated():
    assert exp5.DG_MODEL_ID == "nova-2"
    assert exp5.DG_MODEL_ID in exp5.DG_URL
    assert "diarize=true" in exp5.DG_URL


def test_pricing_constants_are_named_sourced_and_dated():
    assert pytest.approx(0.0043) == exp5.DG_RATE_PER_MIN
    assert pytest.approx(0.0020) == exp5.DG_DIARIZE_ADDON_PER_MIN
    assert exp5.PRICING_SOURCE.startswith("https://deepgram.com/pricing")
    assert "2026-07-17" in exp5.PRICING_SOURCE


def test_dimensional_model_id_is_imported_not_a_duplicated_literal():
    # same object as the shipping dimensional arm's constant -- a copy
    # could silently drift from the model actually loaded.
    assert exp5.DIMENSIONAL_MODEL_ID is dimensional.MODEL_ID


# ---------------------------------------------------------------------------
# run_once: log shape (model/dimensional_model at run level; audio_s +
# cost_usd + overlap_spans per clip, per the standing amendments), the
# customer-only audio actually reaching the dimensional bonus classifier
# (not the full mixed audio), and the guard check-before/add-after order
# (same convention as test_exp4_flash.py's _OrderedGuard).
# ---------------------------------------------------------------------------


class _FakeAudio:
    def __init__(self, n_samples, sr=16000):
        self.samples = np.arange(n_samples, dtype=np.float32)
        self.sr = sr


class _FakeVad:
    speech = []
    speech_ratio = 1.0


_WORDS_BY_CLIP = {
    # 2 turns, no overlap (turn1 starts after turn0 ends).
    "call_001.ogg": [
        _w("hi", 0.0, 0.3, 0),
        _w("there", 0.35, 0.6, 0),
        _w("yes", 1.2, 1.4, 1),
        _w("okay", 1.45, 1.7, 1),
    ],
    # 2 turns, genuine >=0.5s crosstalk (4 words / 1.1s -- not a backchannel).
    # "actually" is given an unrealistically long end (1.1s) purely to push
    # the synthetic cross-speaker intersection past the 0.5s threshold.
    "call_002.ogg": [
        _w("well", 0.0, 0.4, 0),
        _w("actually", 0.42, 1.1, 0),
        _w("no", 0.5, 0.7, 1),
        _w("wait", 0.72, 0.95, 1),
        _w("listen", 0.97, 1.3, 1),
        _w("please", 1.32, 1.6, 1),
    ],
    # single speaker only -> customer_only_audio ambiguous-fallback branch.
    "call_003.ogg": [_w("hello", 0.0, 0.3, 0), _w("world", 0.35, 0.6, 0)],
}
_TRUTH = {
    "call_001.ogg": {"speaker_overlap_present": False},  # matches prediction
    "call_002.ogg": {"speaker_overlap_present": True},  # matches prediction
    "call_003.ogg": {"speaker_overlap_present": True},  # prediction is False -> miss
}


def _patch_run_once_collaborators(monkeypatch, tmp_path, durations_s):
    tone_calls: list[dict] = []
    vad_calls: list[np.ndarray] = []

    def _fake_load_audio(path):
        return _FakeAudio(int(durations_s[path.name] * 16000))

    def _fake_deepgram_words(path):
        return _WORDS_BY_CLIP[path.name]

    def _fake_analyze_vad(samples, sr):
        vad_calls.append(samples)
        return _FakeVad()

    def _fake_classify_tone(arm, samples, sr, vad, snr_db):
        tone_calls.append({"arm": arm, "samples": samples, "sr": sr, "snr_db": snr_db})
        return ToneResult(
            tone=EmotionalTone.NEUTRAL,
            intensity=EmotionalIntensity.MEDIUM,
            confidence=0.6,
            raw={"valence": 0.5, "arousal": 0.4, "dominance": 0.5},
        )

    order: list[tuple[str, float]] = []

    class _OrderedGuard:
        def check(self, projected_usd):
            order.append(("check", projected_usd))

        def add(self, cost_usd):
            order.append(("add", cost_usd))

    monkeypatch.setattr(exp5, "load_truth", lambda: _TRUTH)
    monkeypatch.setattr(exp5, "load_audio", _fake_load_audio)
    monkeypatch.setattr(exp5, "deepgram_words", _fake_deepgram_words)
    monkeypatch.setattr(exp5, "analyze_vad", _fake_analyze_vad)
    monkeypatch.setattr(exp5, "classify_tone", _fake_classify_tone)
    monkeypatch.setattr(exp5, "SpendGuard", lambda: _OrderedGuard())
    # log_run (called for real) writes via common.OUT_DIR -- redirect it so
    # this mock-level test never touches the real out/experiments/ directory.
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    return tone_calls, vad_calls, order


def test_run_once_logs_run_level_models_and_per_clip_audio_s_cost_and_spans(monkeypatch, tmp_path):
    durations_s = {"call_001.ogg": 2.0, "call_002.ogg": 2.0, "call_003.ogg": 1.0}
    tone_calls, vad_calls, order = _patch_run_once_collaborators(monkeypatch, tmp_path, durations_s)

    payload = exp5.run_once(1)

    assert payload["exp"] == "exp5_overlap"
    assert payload["run"] == 1
    assert payload["model"] == exp5.DG_MODEL_ID == "nova-2"
    assert payload["dimensional_model"] == dimensional.MODEL_ID
    assert payload["pricing"]["source"] == exp5.PRICING_SOURCE
    assert payload["pricing"]["rate_per_min"] == exp5.DG_RATE_PER_MIN
    assert payload["pricing"]["diarize_addon_per_min"] == exp5.DG_DIARIZE_ADDON_PER_MIN
    assert payload["audio_minutes"] == pytest.approx(sum(durations_s.values()) / 60.0, abs=1e-4)

    pc = payload["per_clip"]
    effective_rate = exp5.DG_RATE_PER_MIN + exp5.DG_DIARIZE_ADDON_PER_MIN

    # call_001.ogg: no overlap, matches fake truth -> correct
    assert pc["call_001.ogg"]["pred"] == {"speaker_overlap_present": False}
    assert pc["call_001.ogg"]["correct"] == {"speaker_overlap_present": True}
    assert pc["call_001.ogg"]["audio_s"] == pytest.approx(2.0)
    assert pc["call_001.ogg"]["cost_usd"] == pytest.approx(2.0 / 60.0 * effective_rate)
    assert pc["call_001.ogg"]["overlap_spans"] == []
    assert pc["call_001.ogg"]["n_words"] == 4
    assert pc["call_001.ogg"]["n_turns"] == 2

    # call_002.ogg: real crosstalk, matches fake truth -> correct
    assert pc["call_002.ogg"]["pred"] == {"speaker_overlap_present": True}
    assert pc["call_002.ogg"]["correct"] == {"speaker_overlap_present": True}
    assert len(pc["call_002.ogg"]["overlap_spans"]) == 1

    # call_003.ogg: single-speaker fallback, prediction False vs truth True -> miss
    assert pc["call_003.ogg"]["pred"] == {"speaker_overlap_present": False}
    assert pc["call_003.ogg"]["correct"] == {"speaker_overlap_present": False}
    assert "single speaker" in pc["call_003.ogg"]["attribution"]

    # dimensional bonus block present with tone/intensity/valence/arousal
    for name in exp5.ANCHORS:
        dco = pc[name]["dimensional_customer_only"]
        assert dco == {"tone": "neutral", "intensity": "medium", "valence": 0.5, "arousal": 0.4}

    # customer-only audio actually reached the bonus classifier (not full
    # audio) for call_001.ogg: turn1 (speaker 1, customer) is [1.2, 1.7)s.
    sr = 16000
    expected_customer_slice = np.arange(2 * sr, dtype=np.float32)[int(1.2 * sr) : int(1.7 * sr)]
    assert np.array_equal(tone_calls[0]["samples"], expected_customer_slice)
    assert np.array_equal(vad_calls[0], expected_customer_slice)
    assert tone_calls[0]["arm"] == "dimensional"
    assert tone_calls[0]["snr_db"] is None

    # call_003.ogg (index 2): single-speaker fallback -> FULL audio reaches
    # the bonus classifier, not a slice.
    assert np.array_equal(tone_calls[2]["samples"], np.arange(1 * sr, dtype=np.float32))

    total_expected_cost = sum(durations_s[n] / 60.0 * effective_rate for n in exp5.ANCHORS)
    assert payload["cost_usd"] == pytest.approx(total_expected_cost)

    # SpendGuard consulted BEFORE any spend, charged with the measured total AFTER
    assert order[0] == ("check", exp5.EST_COST)
    assert order[-1] == ("add", pytest.approx(total_expected_cost))

    on_disk = json.loads((tmp_path / "exp5_overlap_run1.json").read_text())
    assert on_disk["model"] == "nova-2"
    assert on_disk["per_clip"]["call_002.ogg"]["cost_usd"] == pytest.approx(
        2.0 / 60.0 * effective_rate
    )


def test_run_once_nests_pricing_metadata_per_exp4_convention(monkeypatch, tmp_path):
    """Pricing metadata must be nested under 'pricing' object with 'source' key,
    not flat top-level keys, so Task 9's doc generator parses all experiment
    logs uniformly."""
    durations_s = {"call_001.ogg": 2.0, "call_002.ogg": 2.0, "call_003.ogg": 1.0}
    _patch_run_once_collaborators(monkeypatch, tmp_path, durations_s)

    payload = exp5.run_once(1)

    # Pricing must be nested
    assert "pricing" in payload
    assert isinstance(payload["pricing"], dict)
    assert payload["pricing"]["rate_per_min"] == exp5.DG_RATE_PER_MIN
    assert payload["pricing"]["diarize_addon_per_min"] == exp5.DG_DIARIZE_ADDON_PER_MIN
    assert payload["pricing"]["source"] == exp5.PRICING_SOURCE

    # Flat keys must NOT exist at top level
    assert "rate_per_min" not in payload
    assert "diarize_addon_per_min" not in payload
    assert "pricing_source" not in payload
