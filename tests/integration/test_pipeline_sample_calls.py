import json

import pytest

from autoace_audio.pipeline import analyze

FIELDS_SCORED = [
    "emotional_tone",
    "background_noise_present",
    "audio_quality",
    "speaker_overlap_present",
    "long_silence_present",
]


@pytest.mark.network
def test_end_to_end_against_labels(sample_calls_dir):
    import csv

    labels = {}
    with open(sample_calls_dir / "labels.csv", newline="") as f:
        for row in csv.DictReader(f):
            labels[row["name"]] = json.loads(row["result_json"])
    hits, total = 0, 0
    for name, expected in labels.items():
        output = analyze(sample_calls_dir / name)
        out = output.result.model_dump(mode="json")
        for field in FIELDS_SCORED:
            total += 1
            match = out[field] == expected[field]
            hits += int(match)
            if not match:
                print(f"MISS {name}.{field}: got {out[field]!r} expected {expected[field]!r}")
        print(name, out, output.diagnostics)
    accuracy = hits / total
    # Controller amendment E: call_002's tone is a known/adjudicated miss (gemini
    # returns frustrated vs the neutral label -- see task-8-report.md and
    # test_tone_sample_calls.py's documented xfail); do NOT try to fix it here.
    # 5 fields x 3 calls = 15 comparisons, 1 expected miss = 93% -> still clears 80%.
    assert accuracy >= 0.8, f"sample-call field accuracy {accuracy:.0%} below 80%"
