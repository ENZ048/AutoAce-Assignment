"""Mock-level tests for combined.py: the net-wins-vs-regressions inclusion
rule (standing amendment -- wins AND regressions get equal prominence, so a
lever must beat baseline on NET, not just clear a single mechanical
wins_field hit), the E1/E5 log-shape reshaping those two levers need, the
binding-determination safety net, and run_once's stack-conditional wiring
(gap_noise wired for the real binding stack; flash/fewshot/advocate/
deepgram_overlap proven wireable but never invoked for it). No network, no
local data/ dependency beyond the already-committed out/experiments/ run
logs used by the "matches real data" tests below -- every live collaborator
is monkeypatched at its imported name (same convention as test_exp4_flash.py
/ test_exp5_overlap_math.py)."""

import json

import numpy as np
import pytest
from eval.experiments import combined, common
from eval.experiments import exp2_fewshot as exp2
from eval.experiments import exp3_advocate as exp3
from eval.experiments import exp4_flash as exp4

# ---------------------------------------------------------------------------
# _lever_nets_positive: pure function, the core of the inclusion rule.
# ---------------------------------------------------------------------------


def _runs(correct_flags: list[bool], clip="call_002.ogg", field="emotional_tone"):
    return [{"per_clip": {clip: {"pred": {}, "correct": {field: flag}}}} for flag in correct_flags]


def test_lever_nets_positive_true_for_a_clean_flip_no_regression():
    base = _runs([False, False, False])
    lever = _runs([True, True, False])
    assert combined._lever_nets_positive(base, lever, ["emotional_tone"], ["call_002.ogg"]) is True


def test_lever_nets_positive_false_for_clean_null():
    base = _runs([False, False, False])
    lever = _runs([False, False, False])
    assert combined._lever_nets_positive(base, lever, ["emotional_tone"], ["call_002.ogg"]) is False


def test_lever_nets_positive_false_when_regressions_outnumber_wins():
    """The exact E3/E4 shape: a mechanical wins_field hit on one clip, but
    outnumbered by regressions elsewhere -- must NOT be counted as included,
    even though a naive "any win" check would say True."""
    win_clip = _runs([True, True, True], clip="call_002.ogg")
    win_base = _runs([False, False, False], clip="call_002.ogg")
    loss_clip_a_base = _runs([True, True, True], clip="call_001.ogg")
    loss_clip_a_lever = _runs([False, False, False], clip="call_001.ogg")
    loss_clip_b_base = _runs([True, True, True], clip="call_003.ogg")
    loss_clip_b_lever = _runs([False, False, False], clip="call_003.ogg")

    def _merge(*run_lists):
        merged = [{"per_clip": {}} for _ in range(3)]
        for runs in run_lists:
            for i, r in enumerate(runs):
                merged[i]["per_clip"].update(r["per_clip"])
        return merged

    base = _merge(win_base, loss_clip_a_base, loss_clip_b_base)
    lever = _merge(win_clip, loss_clip_a_lever, loss_clip_b_lever)

    fields, clips = ["emotional_tone"], ["call_001.ogg", "call_002.ogg", "call_003.ogg"]
    # sanity: the mechanical win alone is real (this is what a naive rule would use)
    assert common.wins_field(base, lever, "emotional_tone", "call_002.ogg") is True
    # but net across the full scope is negative -> excluded
    assert combined._lever_nets_positive(base, lever, fields, clips) is False


def test_lever_nets_positive_false_on_empty_runs():
    assert combined._lever_nets_positive(
        [], _runs([True, True, True]), ["emotional_tone"], ["c"]
    ) is (False)
    assert combined._lever_nets_positive(
        _runs([True, True, True]), [], ["emotional_tone"], ["c"]
    ) is (False)


# ---------------------------------------------------------------------------
# _gap_noise_included / _overlap_included: E1 and E5 need reshaping (their
# log format differs from the standard {"correct": {field: bool}} shape) --
# tested against realistic fixtures written to a tmp OUT_DIR.
# ---------------------------------------------------------------------------

_TRUTH = {
    "call_001.ogg": {"background_noise_present": False, "background_noise_type": ""},
    "call_002.ogg": {"background_noise_present": True, "background_noise_type": "TV"},
    "call_003.ogg": {"background_noise_present": True, "background_noise_type": "sharp static"},
}


def _write_baseline_noise_opinion(tmp_path, monkeypatch, per_clip_present_by_run):
    """per_clip_present_by_run: list of 3 dicts {clip: present_bool} mimicking
    exp0_baseline's pred.noise_opinion.present across 3 runs."""
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    for i, present_by_clip in enumerate(per_clip_present_by_run, start=1):
        per_clip = {
            clip: {"pred": {"noise_opinion": {"present": present, "type": ""}}}
            for clip, present in present_by_clip.items()
        }
        common.log_run("exp0_baseline", i, {"cost_usd": 0.0, "per_clip": per_clip})


def _write_gap_noise(tmp_path, monkeypatch, present_correct_by_run):
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    for i, correct_by_clip in enumerate(present_correct_by_run, start=1):
        per_clip = {
            clip: {"skipped": False, "present_correct": correct}
            for clip, correct in correct_by_clip.items()
        }
        common.log_run("exp1_gap_noise", i, {"cost_usd": 0.0, "per_clip": per_clip})


def test_gap_noise_included_true_when_a_noisy_anchor_flips_baseline_blind(monkeypatch, tmp_path):
    monkeypatch.setattr(combined, "load_truth", lambda: _TRUTH)
    # baseline: never confirms noise on either noisy anchor (0/3 both)
    _write_baseline_noise_opinion(
        tmp_path,
        monkeypatch,
        [
            {"call_001.ogg": False, "call_002.ogg": False, "call_003.ogg": False},
            {"call_001.ogg": False, "call_002.ogg": False, "call_003.ogg": False},
            {"call_001.ogg": False, "call_002.ogg": False, "call_003.ogg": False},
        ],
    )
    # E1: call_003 flips to 3/3 correct; call_002 still misses (0/3, matches real data)
    _write_gap_noise(
        tmp_path,
        monkeypatch,
        [
            {"call_001.ogg": True, "call_002.ogg": False, "call_003.ogg": True},
            {"call_001.ogg": True, "call_002.ogg": False, "call_003.ogg": True},
            {"call_001.ogg": True, "call_002.ogg": False, "call_003.ogg": True},
        ],
    )
    assert combined._gap_noise_included() is True


def test_gap_noise_included_false_when_neither_noisy_anchor_flips(monkeypatch, tmp_path):
    monkeypatch.setattr(combined, "load_truth", lambda: _TRUTH)
    _write_baseline_noise_opinion(
        tmp_path,
        monkeypatch,
        [{c: False for c in _TRUTH}] * 3,
    )
    # E1 also never confirms noise anywhere -- no flip, no win
    _write_gap_noise(tmp_path, monkeypatch, [{c: False for c in _TRUTH}] * 3)
    assert combined._gap_noise_included() is False


def test_gap_noise_included_false_when_no_logs_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    monkeypatch.setattr(combined, "load_truth", lambda: _TRUTH)
    assert combined._gap_noise_included() is False


def _write_overlap_correct(tmp_path, monkeypatch, exp, run_correct_by_clip):
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    for i, correct_by_clip in enumerate(run_correct_by_clip, start=1):
        per_clip = {
            clip: {"pred": {}, "correct": {"speaker_overlap_present": correct}}
            for clip, correct in correct_by_clip.items()
        }
        common.log_run(exp, i, {"cost_usd": 0.0, "per_clip": per_clip})


def test_overlap_included_false_when_lever_regresses_a_solid_baseline_clip(monkeypatch, tmp_path):
    _write_overlap_correct(
        tmp_path,
        monkeypatch,
        "exp0_baseline",
        [
            {"call_001.ogg": True, "call_002.ogg": False, "call_003.ogg": True},
            {"call_001.ogg": True, "call_002.ogg": False, "call_003.ogg": True},
            {"call_001.ogg": True, "call_002.ogg": False, "call_003.ogg": True},
        ],
    )
    # single deterministic run: call_003 (solid baseline) flips wrong -> regression, 0 wins
    _write_overlap_correct(
        tmp_path,
        monkeypatch,
        "exp5_overlap",
        [{"call_001.ogg": True, "call_002.ogg": False, "call_003.ogg": False}],
    )
    assert combined._overlap_included() is False


def test_overlap_included_true_when_lever_cleanly_flips_a_blind_clip(monkeypatch, tmp_path):
    _write_overlap_correct(
        tmp_path,
        monkeypatch,
        "exp0_baseline",
        [
            {"call_001.ogg": True, "call_002.ogg": False, "call_003.ogg": True},
            {"call_001.ogg": True, "call_002.ogg": False, "call_003.ogg": True},
            {"call_001.ogg": True, "call_002.ogg": False, "call_003.ogg": True},
        ],
    )
    # single run: call_002 (baseline-blind) now correct, nothing regresses
    _write_overlap_correct(
        tmp_path,
        monkeypatch,
        "exp5_overlap",
        [{"call_001.ogg": True, "call_002.ogg": True, "call_003.ogg": True}],
    )
    assert combined._overlap_included() is True


# ---------------------------------------------------------------------------
# decide_stack: wiring (does it call the right building block for the right
# key?) and the real on-disk acceptance check (does it reproduce the
# controller's binding determination from Study Tasks 2-7's actual logs?).
# ---------------------------------------------------------------------------


def test_decide_stack_wires_each_key_to_the_right_building_block(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    common.log_run("exp0_baseline", 1, {"cost_usd": 0.0, "per_clip": {}})

    seen = {}

    def _fake_net(base, lever_runs, fields, clips):
        seen[tuple(fields)] = (base, lever_runs, clips)
        return fields == ["emotional_tone", "emotional_intensity", "speaker_overlap_present"]

    monkeypatch.setattr(combined, "_lever_nets_positive", _fake_net)
    monkeypatch.setattr(combined, "_gap_noise_included", lambda: True)
    monkeypatch.setattr(combined, "_overlap_included", lambda: False)

    stack = combined.decide_stack()

    assert stack == {
        "flash": True,  # the only fields tuple our fake returns True for
        "fewshot": False,
        "advocate": False,
        "gap_noise": True,
        "deepgram_overlap": False,
    }
    assert ("emotional_intensity",) in seen  # fewshot scope
    assert ("emotional_tone",) in seen  # advocate scope


def test_decide_stack_matches_the_binding_determination_on_real_run_logs():
    """Acceptance check against the actual, already-committed
    out/experiments/*.json logs from Study Tasks 2-7 (no monkeypatch --
    reads the real files, but decide_stack() is read-only). This is the
    controller's binding determination from study-task-8-brief.md's
    dispatch: only E1 (gap_noise) nets positive."""
    stack = combined.decide_stack()
    assert stack == {
        "flash": False,
        "fewshot": False,
        "advocate": False,
        "gap_noise": True,
        "deepgram_overlap": False,
    }


# ---------------------------------------------------------------------------
# verify_stack_matches_determination: safety net before spending live money.
# ---------------------------------------------------------------------------


def test_verify_stack_matches_determination_passes_for_the_binding_stack():
    stack = {
        "flash": False,
        "fewshot": False,
        "advocate": False,
        "gap_noise": True,
        "deepgram_overlap": False,
    }
    combined.verify_stack_matches_determination(stack)  # must not raise


def test_verify_stack_matches_determination_raises_on_mismatch():
    stack = {
        "flash": False,
        "fewshot": False,
        "advocate": True,  # disagrees with the binding E1-only determination
        "gap_noise": True,
        "deepgram_overlap": False,
    }
    with pytest.raises(RuntimeError, match="E1"):
        combined.verify_stack_matches_determination(stack)


# ---------------------------------------------------------------------------
# _shipping_classify: the shipping arm, no lever -- same code path as
# exp0_baseline.run_once's per-clip classification.
# ---------------------------------------------------------------------------


class _FakeAudio:
    samples = np.zeros(16000 * 20, dtype=np.float32)
    sr = 16000


class _FakeVad:
    speech_ratio = 0.6


class _FakeNoise:
    snr_db = 12.5


class _FakeToneResult:
    def __init__(self, data, in_tok, out_tok):
        from autoace_audio.schema import EmotionalIntensity, EmotionalTone

        self.tone = EmotionalTone(data["emotional_tone"])
        self.intensity = EmotionalIntensity(data["emotional_intensity"])
        self.overlap_opinion = data["speaker_overlap_present"]
        self.raw = {"response": data, "prompt_tokens": in_tok, "output_tokens": out_tok}


_SHIPPING_DATA = {
    "emotional_tone": "upset",
    "emotional_intensity": "medium",
    "speaker_overlap_present": False,
    "background_noise_present": False,
    "background_noise_type": "",
    "rationale": "raised voice",
}


def test_shipping_classify_uses_the_shared_gemini_arm_and_returns_full_response(monkeypatch):
    calls = []

    def _fake_classify_tone(arm, samples, sr, vad, snr_db):
        calls.append(arm)
        return _FakeToneResult(_SHIPPING_DATA, 1500, 100)

    monkeypatch.setattr(combined, "load_audio", lambda path: _FakeAudio())
    monkeypatch.setattr(combined, "analyze_vad", lambda samples, sr: _FakeVad())
    monkeypatch.setattr(combined, "analyze_noise", lambda samples, sr, vad: _FakeNoise())
    monkeypatch.setattr(combined, "classify_tone", _fake_classify_tone)

    data, cost, tokens = combined._shipping_classify("call_001.ogg")

    assert calls == ["gemini"]  # the shipping arm, not a copy of its logic
    assert data == _SHIPPING_DATA
    assert tokens == {"in": 1500, "out": 100}
    assert cost == pytest.approx(common.gemini_cost(1500, 100))


# ---------------------------------------------------------------------------
# _gap_listen: E1's lever applied to one clip.
# ---------------------------------------------------------------------------


def test_gap_listen_not_skipped_calls_ask_gemini_gaps_and_returns_its_result(monkeypatch):
    class _LongGapAudio:
        samples = np.zeros(16000 * 20, dtype=np.float32)
        sr = 16000

    class _LongGapVad:
        pass

    def _fake_concat_gaps(samples, sr, vad):
        return np.zeros(16000 * 10, dtype=np.float32)  # 10s of gap audio

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

    monkeypatch.setattr(combined, "load_audio", lambda path: _LongGapAudio())
    monkeypatch.setattr(combined, "analyze_vad", lambda samples, sr: _LongGapVad())
    monkeypatch.setattr(combined, "encode_opus_ogg", lambda samples, sr: b"fake-ogg")
    monkeypatch.setattr(combined.exp1_gap_noise, "concat_gaps", _fake_concat_gaps)
    monkeypatch.setattr(combined.exp1_gap_noise, "ask_gemini_gaps", _fake_ask_gemini_gaps)

    result = combined._gap_listen("call_003.ogg")

    assert result["skipped"] is False
    assert result["pred"]["background_noise_type"] == "static"
    assert result["tokens"] == {"in": 900, "out": 25}
    assert result["cost_usd"] == pytest.approx(0.0015)


def test_gap_listen_skips_when_gap_audio_too_short(monkeypatch):
    class _ShortGapAudio:
        samples = np.zeros(16000 * 5, dtype=np.float32)
        sr = 16000

    calls = []

    def _fake_ask_gemini_gaps(blob):
        calls.append(blob)
        raise AssertionError("must not be called when gap audio is too short")

    monkeypatch.setattr(combined, "load_audio", lambda path: _ShortGapAudio())
    monkeypatch.setattr(combined, "analyze_vad", lambda samples, sr: object())
    monkeypatch.setattr(combined.exp1_gap_noise, "concat_gaps", lambda s, sr, v: np.zeros(1000))
    monkeypatch.setattr(combined.exp1_gap_noise, "ask_gemini_gaps", _fake_ask_gemini_gaps)

    result = combined._gap_listen("call_001.ogg")

    assert result["skipped"] is True
    assert result["pred"] is None
    assert result["cost_usd"] == 0.0
    assert calls == []


# ---------------------------------------------------------------------------
# run_once: stack-conditional wiring + log shape. First with the REAL
# binding stack (gap_noise wired, everything else absent), then with each
# hypothetical flag flipped to prove the mechanism actually generalizes
# (not hardcoded to only ever do E1).
# ---------------------------------------------------------------------------

_TRUTH_ANCHORS = {
    "call_001.ogg": {
        "emotional_tone": "upset",
        "emotional_intensity": "high",
        "speaker_overlap_present": False,
        "background_noise_present": False,
        "background_noise_type": "",
    },
    "call_002.ogg": {
        "emotional_tone": "neutral",
        "emotional_intensity": "medium",
        "speaker_overlap_present": True,
        "background_noise_present": True,
        "background_noise_type": "TV",
    },
    "call_003.ogg": {
        "emotional_tone": "satisfied",
        "emotional_intensity": "medium",
        "speaker_overlap_present": True,
        "background_noise_present": True,
        "background_noise_type": "sharp static",
    },
}

_BINDING_STACK = {
    "flash": False,
    "fewshot": False,
    "advocate": False,
    "gap_noise": True,
    "deepgram_overlap": False,
}

_ALL_FALSE_STACK = {
    "flash": False,
    "fewshot": False,
    "advocate": False,
    "gap_noise": False,
    "deepgram_overlap": False,
}


def _shipping_fixture():
    return {
        "call_001.ogg": (
            {
                "emotional_tone": "upset",
                "emotional_intensity": "high",
                "speaker_overlap_present": False,
                "background_noise_present": False,
                "background_noise_type": "",
            },
            0.0015,
            {"in": 1500, "out": 100},
        ),
        "call_002.ogg": (
            {
                "emotional_tone": "frustrated",  # baseline's known call_002 miss
                "emotional_intensity": "medium",
                "speaker_overlap_present": False,
                "background_noise_present": False,
                "background_noise_type": "",
            },
            0.0016,
            {"in": 1600, "out": 110},
        ),
        "call_003.ogg": (
            {
                "emotional_tone": "satisfied",
                "emotional_intensity": "low",
                "speaker_overlap_present": True,
                "background_noise_present": False,
                "background_noise_type": "",
            },
            0.0050,
            {"in": 5000, "out": 105},
        ),
    }


def _gap_fixture():
    """call_003 is E1's live win (type fixed to 'static'); call_001/002
    mirror the real measured behavior (correct absence / still-missed TV)."""
    return {
        "call_001.ogg": {
            "skipped": False,
            "gap_seconds": 16.1,
            "pred": {
                "background_noise_present": False,
                "background_noise_type": "",
                "character": "none",
            },
            "tokens": {"in": 300, "out": 15},
            "cost_usd": 0.0001,
        },
        "call_002.ogg": {
            "skipped": False,
            "gap_seconds": 10.0,
            "pred": {
                "background_noise_present": False,
                "background_noise_type": "",
                "character": "none",
            },
            "tokens": {"in": 250, "out": 15},
            "cost_usd": 0.0001,
        },
        "call_003.ogg": {
            "skipped": False,
            "gap_seconds": 40.6,
            "pred": {
                "background_noise_present": True,
                "background_noise_type": "static",
                "character": "constant",
            },
            "tokens": {"in": 700, "out": 20},
            "cost_usd": 0.0003,
        },
    }


class _OrderedGuard:
    def __init__(self):
        self.order: list[str] = []

    def check(self, projected_usd):
        self.order.append("check")

    def add(self, cost_usd):
        self.order.append("add")


def test_run_once_binding_stack_wires_shipping_and_gap_listening_only(monkeypatch, tmp_path):
    shipping = _shipping_fixture()
    gaps = _gap_fixture()
    guard = _OrderedGuard()

    shipping_calls, gap_calls = [], []
    unexpected_calls = []

    def _fake_shipping(name):
        shipping_calls.append(name)
        return shipping[name]

    def _fake_gap(name):
        gap_calls.append(name)
        return gaps[name]

    monkeypatch.setattr(combined, "load_truth", lambda: _TRUTH_ANCHORS)
    monkeypatch.setattr(combined, "_shipping_classify", _fake_shipping)
    monkeypatch.setattr(combined, "_gap_listen", _fake_gap)
    monkeypatch.setattr(combined, "SpendGuard", lambda: guard)
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)

    monkeypatch.setattr(
        exp2, "classify_with_exemplars", lambda name: unexpected_calls.append("fewshot")
    )
    monkeypatch.setattr(exp4, "classify_flash", lambda name: unexpected_calls.append("flash"))
    monkeypatch.setattr(exp3, "advocate_pass", lambda *a, **kw: unexpected_calls.append("advocate"))

    payload = combined.run_once(1, _BINDING_STACK)

    assert unexpected_calls == []  # flash/fewshot/advocate never invoked
    assert sorted(shipping_calls) == sorted(_TRUTH_ANCHORS)
    assert sorted(gap_calls) == sorted(_TRUTH_ANCHORS)

    pc = payload["per_clip"]
    # tone/intensity/overlap: untouched by gap_noise, straight from shipping
    assert pc["call_002.ogg"]["pred"]["emotional_tone"] == "frustrated"
    assert (
        pc["call_002.ogg"]["correct"]["emotional_tone"] is False
    )  # known baseline miss, preserved

    # noise fields: overridden by E1 gap-listening -- the call_003 type fix
    assert pc["call_003.ogg"]["pred"]["background_noise_present"] is True
    assert pc["call_003.ogg"]["pred"]["background_noise_type"] == "static"
    assert pc["call_003.ogg"]["correct"]["background_noise_present"] is True
    assert pc["call_003.ogg"]["background_noise_type_truth"] == "sharp static"

    # call_001/002 noise fields also come from gap-listening (both False/""),
    # matching shipping's own values here -- no visible change, but sourced
    # from E1, not silently left as shipping's
    assert pc["call_001.ogg"]["pred"]["background_noise_present"] is False
    assert (
        pc["call_002.ogg"]["pred"]["background_noise_present"] is False
    )  # still-missed TV, disclosed

    # per-clip tokens/cost_usd always present, aggregating shipping + gap call
    assert pc["call_003.ogg"]["tokens"] == {"in": 5000 + 700, "out": 105 + 20}
    assert pc["call_003.ogg"]["cost_usd"] == pytest.approx(0.0050 + 0.0003)
    assert pc["call_003.ogg"]["gap_listening"] == gaps["call_003.ogg"]

    # run-level shape
    assert payload["exp"] == "combined"
    assert payload["run"] == 1
    assert payload["stack"] == _BINDING_STACK
    assert payload["included_levers"] == ["E1"]
    assert set(payload["exclusions"]) == {"E2", "E3", "E4", "E5", "E5_bonus"}
    assert "model" in payload
    assert payload["pricing"]["in_per_1m"] == common.GEMINI_LITE_IN
    assert payload["cost_usd"] == pytest.approx(sum(v["cost_usd"] for v in pc.values()))

    # SpendGuard consulted before any spend, charged with the measured total after
    assert guard.order[0] == "check"
    assert guard.order[-1] == "add"

    on_disk = json.loads((tmp_path / "combined_run1.json").read_text())
    assert on_disk["included_levers"] == ["E1"]
    assert on_disk["per_clip"]["call_003.ogg"]["pred"]["background_noise_type"] == "static"


def test_run_once_gap_noise_off_leaves_shipping_noise_fields_untouched(monkeypatch, tmp_path):
    shipping = _shipping_fixture()
    gap_calls = []

    monkeypatch.setattr(combined, "load_truth", lambda: _TRUTH_ANCHORS)
    monkeypatch.setattr(combined, "_shipping_classify", lambda name: shipping[name])
    monkeypatch.setattr(combined, "_gap_listen", lambda name: gap_calls.append(name) or {})
    monkeypatch.setattr(combined, "SpendGuard", lambda: _OrderedGuard())
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)

    payload = combined.run_once(1, _ALL_FALSE_STACK)

    assert gap_calls == []  # gap_noise off -> _gap_listen never invoked
    pc = payload["per_clip"]
    assert (
        pc["call_003.ogg"]["pred"]["background_noise_present"] is False
    )  # shipping's own, unmodified
    assert pc["call_003.ogg"]["gap_listening"] is None


def test_run_once_flash_flag_dispatches_to_exp4_not_shipping(monkeypatch, tmp_path):
    stack = dict(_ALL_FALSE_STACK, flash=True)
    flash_calls = []
    shipping_calls = []

    def _fake_flash(name):
        flash_calls.append(name)
        return (
            {
                "emotional_tone": "neutral",
                "emotional_intensity": "low",
                "speaker_overlap_present": False,
                "background_noise_present": False,
                "background_noise_type": "",
            },
            0.002,
            {"in": 2000, "out": 90},
        )

    monkeypatch.setattr(combined, "load_truth", lambda: _TRUTH_ANCHORS)
    monkeypatch.setattr(exp4, "classify_flash", _fake_flash)
    monkeypatch.setattr(combined, "_shipping_classify", lambda name: shipping_calls.append(name))
    monkeypatch.setattr(combined, "SpendGuard", lambda: _OrderedGuard())
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)

    payload = combined.run_once(1, stack)

    assert shipping_calls == []
    assert sorted(flash_calls) == sorted(_TRUTH_ANCHORS)
    assert payload["per_clip"]["call_001.ogg"]["pred"]["emotional_tone"] == "neutral"


def test_run_once_fewshot_flag_dispatches_to_exp2(monkeypatch, tmp_path):
    stack = dict(_ALL_FALSE_STACK, fewshot=True)
    fewshot_calls = []

    def _fake_fewshot(name):
        fewshot_calls.append(name)
        return (
            {
                "emotional_tone": "upset",
                "emotional_intensity": "high",
                "speaker_overlap_present": False,
                "background_noise_present": False,
                "background_noise_type": "",
            },
            0.003,
        )

    monkeypatch.setattr(combined, "load_truth", lambda: _TRUTH_ANCHORS)
    monkeypatch.setattr(exp2, "classify_with_exemplars", _fake_fewshot)
    monkeypatch.setattr(combined, "SpendGuard", lambda: _OrderedGuard())
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)

    payload = combined.run_once(1, stack)

    assert sorted(fewshot_calls) == sorted(_TRUTH_ANCHORS)
    assert payload["per_clip"]["call_001.ogg"]["tokens"] == {
        "in": 0,
        "out": 0,
    }  # exp2 predates tokens


def test_run_once_advocate_flag_dispatches_to_exp3_and_overrides_tone(monkeypatch, tmp_path):
    stack = dict(_ALL_FALSE_STACK, advocate=True)
    shipping = _shipping_fixture()
    advocate_calls = []

    def _fake_advocate(name, first_tone, first_intensity, rationale):
        advocate_calls.append((name, first_tone, first_intensity))
        # advocate_pass sends the SAME full GEMINI_RESPONSE_SCHEMA as the
        # shipping arm (required: tone/intensity/noise/overlap), so a
        # realistic final verdict always carries every field, not just tone.
        return (
            {
                "emotional_tone": "neutral",
                "emotional_intensity": first_intensity,
                "speaker_overlap_present": False,
                "background_noise_present": False,
                "background_noise_type": "",
            },
            0.001,
            {"in": 900, "out": 60},
        )

    monkeypatch.setattr(combined, "load_truth", lambda: _TRUTH_ANCHORS)
    monkeypatch.setattr(combined, "_shipping_classify", lambda name: shipping[name])
    monkeypatch.setattr(exp3, "advocate_pass", _fake_advocate)
    monkeypatch.setattr(combined, "SpendGuard", lambda: _OrderedGuard())
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)

    payload = combined.run_once(1, stack)

    assert ("call_002.ogg", "frustrated", "medium") in advocate_calls
    # advocate's final verdict overrides the shipping first pass
    assert payload["per_clip"]["call_002.ogg"]["pred"]["emotional_tone"] == "neutral"


def test_run_once_deepgram_overlap_flag_overrides_from_exp5_run_log(monkeypatch, tmp_path):
    stack = dict(_ALL_FALSE_STACK, deepgram_overlap=True)
    shipping = _shipping_fixture()

    monkeypatch.setattr(combined, "load_truth", lambda: _TRUTH_ANCHORS)
    monkeypatch.setattr(combined, "_shipping_classify", lambda name: shipping[name])
    monkeypatch.setattr(combined, "SpendGuard", lambda: _OrderedGuard())
    monkeypatch.setattr(common, "OUT_DIR", tmp_path)
    common.log_run(
        "exp5_overlap",
        1,
        {
            "cost_usd": 0.0,
            "per_clip": {
                "call_001.ogg": {"pred": {"speaker_overlap_present": True}},
                "call_002.ogg": {"pred": {"speaker_overlap_present": True}},
                "call_003.ogg": {"pred": {"speaker_overlap_present": False}},
            },
        },
    )

    payload = combined.run_once(1, stack)

    # shipping said call_001 overlap=False; deepgram overrides it to True
    assert payload["per_clip"]["call_001.ogg"]["pred"]["speaker_overlap_present"] is True
    assert payload["per_clip"]["call_003.ogg"]["pred"]["speaker_overlap_present"] is False
