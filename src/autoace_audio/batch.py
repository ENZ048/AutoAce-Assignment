"""Batch processing with per-file failure isolation: one bad file never kills the run.
Manifest contract per brief: CSV with `name` (exact filename) and `result_json`."""

import csv
import json
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from autoace_audio.audio_io import DecodeError
from autoace_audio.pipeline import PipelineOutput, analyze
from autoace_audio.schema import AnalysisResult, FileError

AUDIO_SUFFIXES = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".webm"}


@dataclass
class BatchReport:
    results: dict[str, AnalysisResult] = field(default_factory=dict)
    errors: list[FileError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _find_manifest(input_dir: Path) -> Path | None:
    csvs = sorted(input_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def validate_batch(input_dir: Path) -> tuple[list[Path], list[str]]:
    """Cross-check manifest rows vs files on disk, both directions."""
    files = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
    )
    warnings: list[str] = []
    manifest = _find_manifest(input_dir)
    if manifest is None:
        warnings.append("no CSV manifest found — processing every audio file")
        return files, warnings
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("name")]
    manifest_names = {r["name"].strip() for r in rows}
    disk_names = {p.name for p in files}
    for name in sorted(manifest_names - disk_names):
        warnings.append(f"manifest row has no file on disk: {name}")
    for name in sorted(disk_names - manifest_names):
        warnings.append(f"file not listed in manifest (processed anyway): {name}")
    return files, warnings


def _unzip_if_needed(input_path: Path) -> tuple[Path, Path | None]:
    """Extract ZIP if needed, return (working_dir, temp_dir_to_cleanup)."""
    if input_path.suffix.lower() == ".zip":
        target = Path(tempfile.mkdtemp(prefix="autoace_batch_"))
        with zipfile.ZipFile(input_path) as z:
            z.extractall(target)
        # Determine which directory to process: if root has audio files, use it;
        # elif one subdir exists and root has CSVs, move CSVs into subdir and use it;
        # else use root.
        non_csv_files = [p for p in target.iterdir() if p.is_file() and p.suffix.lower() != ".csv"]
        if non_csv_files:
            return target, target
        subdirs = [d for d in target.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            csv_files = list(target.glob("*.csv"))
            if csv_files:
                # Move CSVs into the single subdirectory
                for csv_file in csv_files:
                    shutil.move(str(csv_file), str(subdirs[0] / csv_file.name))
            return subdirs[0], target
        return target, target
    return input_path, None


def run_batch(
    input_path: Path,
    out_dir: Path,
    tone_arm: str | None = None,
    analyze_fn: Callable[[Path, str | None], PipelineOutput] = analyze,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> BatchReport:
    input_dir, temp_dir = _unzip_if_needed(Path(input_path))
    try:
        files, warnings = validate_batch(input_dir)
        report = BatchReport(warnings=warnings)
        for i, path in enumerate(files):
            try:
                out = analyze_fn(path, tone_arm=tone_arm)
                report.results[path.name] = out.result
            except DecodeError as e:
                report.errors.append(FileError(name=path.name, error=f"decode: {e}"))
            except Exception as e:  # noqa: BLE001 — isolation is the contract
                report.errors.append(FileError(name=path.name, error=f"{type(e).__name__}: {e}"))
            if progress_cb:
                progress_cb(i + 1, len(files), path.name)
        _write_outputs(report, Path(out_dir))
        return report
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _write_outputs(report: BatchReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "result_json"])
        for name, result in report.results.items():
            w.writerow([name, result.to_result_json()])
    (out_dir / "results.json").write_text(
        json.dumps({n: json.loads(r.to_result_json()) for n, r in report.results.items()}, indent=2)
    )
    with open(out_dir / "errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "error"])
        for e in report.errors:
            w.writerow([e.name, e.error])
