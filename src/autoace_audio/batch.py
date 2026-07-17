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
    all_by_name = {p.name: p for p in input_dir.iterdir() if p.is_file()}
    warnings: list[str] = []
    manifest = _find_manifest(input_dir)
    if manifest is None:
        files = sorted(p for p in all_by_name.values() if p.suffix.lower() in AUDIO_SUFFIXES)
        warnings.append("no CSV manifest found — processing every audio file")
        return files, warnings
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("name")]
    manifest_names = {r["name"].strip() for r in rows}
    suffix_names = {name for name, p in all_by_name.items() if p.suffix.lower() in AUDIO_SUFFIXES}
    # A manifest-listed file that exists on disk is processed regardless of its
    # extension -- decode is ffprobe content-sniffing, never extension-based (see
    # audio_io.py's module docstring); an unrecognized suffix must never silently
    # drop a real, manifest-referenced file.
    keep_names = suffix_names | (manifest_names & all_by_name.keys())
    files = sorted(all_by_name[name] for name in keep_names)
    for name in sorted(manifest_names - all_by_name.keys()):
        warnings.append(f"manifest row has no file on disk: {name}")
    for name in sorted(suffix_names - manifest_names):
        warnings.append(f"file not listed in manifest (processed anyway): {name}")
    return files, warnings


def _is_archive_junk(p: Path) -> bool:
    """Archiver metadata that must not influence batch-root resolution:
    macOS Finder's __MACOSX/ mirror, .DS_Store, and AppleDouble ._* files."""
    return p.name in ("__MACOSX", ".DS_Store") or p.name.startswith("._")


def resolve_batch_root(root: Path) -> Path:
    """Pick the directory to process after extraction: root wins if it has
    non-CSV files; else a single subdir becomes the batch root and root-level
    CSVs move into it. Shared by the CLI and the dashboard so both resolve the
    same root for the same ZIP."""
    entries = [p for p in root.iterdir() if not _is_archive_junk(p)]
    non_csv_files = [p for p in entries if p.is_file() and p.suffix.lower() != ".csv"]
    if non_csv_files:
        return root
    subdirs = [d for d in entries if d.is_dir()]
    if len(subdirs) == 1:
        csv_files = [p for p in entries if p.is_file() and p.suffix.lower() == ".csv"]
        for csv_file in csv_files:
            shutil.move(str(csv_file), str(subdirs[0] / csv_file.name))
        return subdirs[0]
    return root


def _unzip_if_needed(input_path: Path) -> tuple[Path, Path | None]:
    """Extract ZIP if needed, return (working_dir, temp_dir_to_cleanup)."""
    if input_path.suffix.lower() == ".zip":
        target = Path(tempfile.mkdtemp(prefix="autoace_batch_"))
        with zipfile.ZipFile(input_path) as z:
            z.extractall(target)
        return resolve_batch_root(target), target
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
        # Per-file diagnostics never leave this function -- BatchReport's public
        # shape stays exactly what it was; we only need a count out of it below.
        tone_fallback_count = 0
        for i, path in enumerate(files):
            try:
                out = analyze_fn(path, tone_arm=tone_arm)
                report.results[path.name] = out.result
                if out.diagnostics.get("tone_error"):
                    tone_fallback_count += 1
            except DecodeError as e:
                report.errors.append(FileError(name=path.name, error=f"decode: {e}"))
            except Exception as e:  # noqa: BLE001 — isolation is the contract
                report.errors.append(FileError(name=path.name, error=f"{type(e).__name__}: {e}"))
            if progress_cb:
                progress_cb(i + 1, len(files), path.name)
        if tone_fallback_count > 0:
            # A silent whole-batch tone downgrade (every fallback is a per-file
            # tone-arm miss) must be visible at the report level, not just buried
            # in per-file diagnostics that callers may never inspect.
            report.warnings.append(
                f"{tone_fallback_count}/{len(files)} files fell back from the "
                "requested tone arm (see tone_error)"
            )
        _write_outputs(report, Path(out_dir))
        return report
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _write_outputs(report: BatchReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Explicit utf-8 everywhere: background_noise_type (and error text) can carry
    # non-ASCII, and locale-default writes are a real deliverable risk on any box
    # whose locale isn't UTF-8. ensure_ascii=False keeps results.json human-readable
    # (real characters) instead of \uXXXX-escaping every non-ASCII value.
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "result_json"])
        for name, result in report.results.items():
            w.writerow([name, result.to_result_json()])
    (out_dir / "results.json").write_text(
        json.dumps(
            {n: json.loads(r.to_result_json()) for n, r in report.results.items()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with open(out_dir / "errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "error"])
        for e in report.errors:
            w.writerow([e.name, e.error])
