import pytest

from autoace_audio.analyzers.noise import analyze_noise
from autoace_audio.analyzers.vad import analyze_vad
from autoace_audio.audio_io import load_audio


@pytest.mark.slow
def test_call_002_has_noise_call_001_does_not(sample_calls_dir):
    a1 = load_audio(sample_calls_dir / "call_001.ogg")
    n1 = analyze_noise(a1.samples, a1.sr, analyze_vad(a1.samples, a1.sr))
    a2 = load_audio(sample_calls_dir / "call_002.ogg")
    n2 = analyze_noise(a2.samples, a2.sr, analyze_vad(a2.samples, a2.sr))
    assert not n1.present, f"001 labeled no-noise; got {n1.top_events[:3]}"
    assert n2.present, f"002 labeled TV/medium; got {n2.top_events[:3]}"


@pytest.mark.slow
@pytest.mark.xfail(
    reason="CNN14 lacks a static/crackle response for this noise; fusion's audio-LLM "
    "noise-opinion backstop (Rule B) is designed to cover this case but has zero observed "
    "live triggers -- Gemini denies background_noise_present outright for this noise family "
    "(call_003 and all 9 synthetic static/hum/TV-bleed clips tested); see docs/decisions.md "
    "Task 9 and Task 11 section 2",
    strict=False,
)
def test_call_003_static_detected(sample_calls_dir):
    """call_003 is labeled "sharp static"/medium. present + severity happen to match
    the label (via a sustained "Radio" read, not a static-family class — see
    task-6-report.md), so this checks the meaningful outcome (a static-family
    type_label) rather than the presence bit alone, which would pass for the wrong
    reason."""
    a3 = load_audio(sample_calls_dir / "call_003.ogg")
    n3 = analyze_noise(a3.samples, a3.sr, analyze_vad(a3.samples, a3.sr))
    assert n3.present and n3.type_label == "static", (
        f"003 labeled sharp static/medium; got {n3.top_events[:3]}"
    )
