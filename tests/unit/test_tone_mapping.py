from autoace_audio.analyzers.tone.dimensional import confidence_from_va, map_va
from autoace_audio.schema import EmotionalIntensity, EmotionalTone


def test_high_valence_is_satisfied_any_arousal():
    assert map_va(arousal=0.2, valence=0.8)[0] == EmotionalTone.SATISFIED
    assert map_va(arousal=0.9, valence=0.8)[0] == EmotionalTone.SATISFIED


def test_low_valence_high_arousal_is_upset():
    assert map_va(arousal=0.7, valence=0.35)[0] == EmotionalTone.UPSET


def test_extreme_corner_is_distressed():
    assert map_va(arousal=0.8, valence=0.2)[0] == EmotionalTone.DISTRESSED


def test_mild_negative_is_frustrated():
    assert map_va(arousal=0.5, valence=0.42)[0] == EmotionalTone.FRUSTRATED


def test_middle_is_neutral_and_intensity_bands():
    tone, intensity = map_va(arousal=0.3, valence=0.5)
    assert tone == EmotionalTone.NEUTRAL and intensity == EmotionalIntensity.LOW
    assert map_va(arousal=0.5, valence=0.5)[1] == EmotionalIntensity.MEDIUM
    assert map_va(arousal=0.9, valence=0.5)[1] == EmotionalIntensity.HIGH


def test_confidence_floor_is_reachable_exactly_on_a_boundary():
    # valence == va_satisfied_v (0.60) puts boundary_dist at exactly 0 regardless
    # of arousal -- the floor must be reachable, not just an unreachable clip bound.
    assert confidence_from_va(arousal=0.5, valence=0.60) == 0.35


def test_confidence_ceiling_far_from_every_boundary():
    # arousal=0.0 (far from va_upset_a=0.60) and valence=1.0 (far from
    # va_satisfied_v/va_upset_v/va_frustrated_v) -> boundary_dist=0.4, well past
    # the ceiling's saturation point.
    assert confidence_from_va(arousal=0.0, valence=1.0) == 0.85
