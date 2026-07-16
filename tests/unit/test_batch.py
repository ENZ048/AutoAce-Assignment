import csv
import json
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

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


def test_flat_zip_processes_files(tmp_path):
    """Flat zip: audio + csv at root."""
    # Create files
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFFxxxxWAVE")
    csv_file = tmp_path / "manifest.csv"
    with open(csv_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "result_json"])
        w.writerow(["audio.wav", ""])
    # Create zip
    zip_path = tmp_path / "flat.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(audio_file, arcname="audio.wav")
        z.write(csv_file, arcname="manifest.csv")
    # Process
    report = run_batch(zip_path, tmp_path / "out", analyze_fn=_fake_analyze)
    assert set(report.results) == {"audio.wav"}
    assert len(report.errors) == 0


def test_csv_root_with_audio_subdir(tmp_path):
    """CSV at root, audio in single subdir."""
    # Create manifest at root
    manifest = tmp_path / "labels.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "result_json"])
        w.writerow(["sound.wav", ""])
    # Create audio in subdir
    audio_dir = tmp_path / "calls"
    audio_dir.mkdir()
    (audio_dir / "sound.wav").write_bytes(b"RIFFxxxxWAVE")
    # Create zip
    zip_path = tmp_path / "with_subdir.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(manifest, arcname="labels.csv")
        z.write(audio_dir / "sound.wav", arcname="calls/sound.wav")
    # Process — CSV should be moved into calls/ dir
    report = run_batch(zip_path, tmp_path / "out", analyze_fn=_fake_analyze)
    assert set(report.results) == {"sound.wav"}
    assert len(report.errors) == 0


def test_temp_dir_cleanup_after_run_batch(tmp_path):
    """Verify that temp directory is cleaned up after run_batch returns."""
    # Create a zip
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFFxxxxWAVE")
    csv_file = tmp_path / "manifest.csv"
    with open(csv_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "result_json"])
        w.writerow(["audio.wav", ""])
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(audio_file, arcname="audio.wav")
        z.write(csv_file, arcname="manifest.csv")
    # Capture mkdtemp calls to track temp dir creation
    created_dirs = []
    original_mkdtemp = tempfile.mkdtemp

    def capture_mkdtemp(*args, **kwargs):
        d = original_mkdtemp(*args, **kwargs)
        created_dirs.append(d)
        return d

    with patch("tempfile.mkdtemp", side_effect=capture_mkdtemp):
        report = run_batch(zip_path, tmp_path / "out", analyze_fn=_fake_analyze)
    # Verify processing worked
    assert set(report.results) == {"audio.wav"}
    # Verify temp dir was cleaned up
    assert len(created_dirs) == 1
    assert not Path(created_dirs[0]).exists()
