from autoace_audio.analyzers.vad import Segment, build_vad_map


def test_gaps_include_leading_and_trailing():
    m = build_vad_map([Segment(5.0, 10.0)], total_s=20.0, long_silence_s=10.0)
    assert m.gaps == [Segment(0.0, 5.0), Segment(10.0, 20.0)]
    assert m.max_gap_s == 10.0
    assert m.long_silence_present  # trailing 10s gap hits threshold


def test_seven_second_gap_is_not_long_silence():
    """AutoAce labeled a 7.4s mid-call gap as long_silence=false."""
    m = build_vad_map(
        [Segment(0.0, 100.0), Segment(107.4, 170.0)], total_s=170.0, long_silence_s=10.0
    )
    assert not m.long_silence_present
    assert abs(m.max_gap_s - 7.4) < 1e-6


def test_no_speech_at_all_is_one_giant_gap():
    m = build_vad_map([], total_s=30.0, long_silence_s=10.0)
    assert m.speech_ratio == 0.0
    assert m.long_silence_present
