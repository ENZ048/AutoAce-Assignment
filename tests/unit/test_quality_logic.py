from autoace_audio.analyzers.quality import rate_quality
from autoace_audio.schema import AudioQuality


def test_high_pesq_good_stoi_is_clear():
    rating, override = rate_quality(pesq=3.5, stoi=0.9, clipping_ratio=0.0)
    assert rating == AudioQuality.CLEAR and not override


def test_mid_pesq_is_slightly_impaired():
    rating, _ = rate_quality(pesq=2.4, stoi=0.9, clipping_ratio=0.0)
    assert rating == AudioQuality.SLIGHTLY_IMPAIRED


def test_low_stoi_degrades_one_level():
    rating, _ = rate_quality(pesq=3.5, stoi=0.6, clipping_ratio=0.0)
    assert rating == AudioQuality.SLIGHTLY_IMPAIRED


def test_clipping_overrides_everything():
    rating, override = rate_quality(pesq=4.0, stoi=0.95, clipping_ratio=0.08)
    assert rating == AudioQuality.SEVERELY_IMPAIRED and override


def test_missing_scores_default_clear_no_override():
    rating, override = rate_quality(pesq=None, stoi=None, clipping_ratio=0.0)
    assert rating == AudioQuality.CLEAR and not override
