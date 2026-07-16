import csv
import json
from pathlib import Path

from autoace_audio.audio_io import DecodeError
from autoace_audio.batch import run_batch, validate_batch
from autoace_audio.pipeline import PipelineOutput
from autoace_audio.schema import AnalysisResult

GOOD = AnalysisResult(
    emotional_tone="neutral",
    emotional_intensity="low",
    background_noise_present=False,
    background_noise_type="",
    background_noise_severity="none",
    audio_quality="clear",
    speaker_overlap_present=False,
    long_silence_present=False,
    confidence=0.8,
)


def _fake_analyze(path, tone_arm=None):
    if "corrupt" in Path(path).name:
        raise DecodeError("bad file")
    return PipelineOutput(result=GOOD, diagnostics={})


def _mkbatch(tmp_path, names, manifest_rows):
    d = tmp_path / "batch"
    d.mkdir()
    for n in names:
        (d / n).write_bytes(b"RIFFxxxxWAVE")
    with open(d / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "result_json"])
        w.writerows(manifest_rows)
    return d


def test_validate_reports_mismatches_both_ways(tmp_path):
    d = _mkbatch(tmp_path, ["a.wav"], [["a.wav", ""], ["missing.wav", ""]])
    (d / "extra.wav").write_bytes(b"RIFF")
    files, warnings = validate_batch(d)
    assert {f.name for f in files} == {"a.wav", "extra.wav"}
    joined = " ".join(warnings)
    assert "missing.wav" in joined and "extra.wav" in joined


def test_one_corrupt_file_does_not_kill_batch(tmp_path):
    d = _mkbatch(tmp_path, ["ok.wav", "corrupt.wav"], [["ok.wav", ""], ["corrupt.wav", ""]])
    report = run_batch(d, tmp_path / "out", analyze_fn=_fake_analyze)
    assert set(report.results) == {"ok.wav"}
    assert len(report.errors) == 1 and report.errors[0].name == "corrupt.wav"
    with open(tmp_path / "out" / "results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["name"] == "ok.wav"
    assert json.loads(rows[0]["result_json"])["emotional_tone"] == "neutral"


def test_results_json_preserves_filenames(tmp_path):
    d = _mkbatch(tmp_path, ["x.wav"], [["x.wav", ""]])
    run_batch(d, tmp_path / "out", analyze_fn=_fake_analyze)
    data = json.loads((tmp_path / "out" / "results.json").read_text())
    assert list(data.keys()) == ["x.wav"]
