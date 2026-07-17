"""Zip-slip-safe extraction + batch-root resolution.

Mirrors autoace_audio.batch._unzip_if_needed's root-resolution semantics.
Exists because ZipFile.extractall lacks member-path safety and we extract into
the job directory, not a tempdir. Backend code is never modified."""

import shutil
import zipfile
from pathlib import Path


class UnsafeZipError(ValueError):
    """A zip member path would escape the extraction directory."""


def _validate_members(zf: zipfile.ZipFile, dest: Path) -> None:
    base = dest.resolve()
    for info in zf.infolist():
        name = info.filename
        if name.startswith(("/", "\\")) or ".." in Path(name).parts:
            raise UnsafeZipError(f"unsafe path in zip: {name!r}")
        if not (dest / name).resolve().is_relative_to(base):
            raise UnsafeZipError(f"unsafe path in zip: {name!r}")


def extract_zip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        _validate_members(zf, dest)  # all-or-nothing: validate before extracting anything
        for info in zf.infolist():
            if info.is_dir():
                (dest / info.filename).mkdir(parents=True, exist_ok=True)
                continue
            target = dest / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    return _resolve_batch_root(dest)


def _resolve_batch_root(dest: Path) -> Path:
    """Same rules as the backend: root wins if it has non-CSV files; else a single
    subdir becomes the batch root and root-level CSVs move into it."""
    non_csv = [p for p in dest.iterdir() if p.is_file() and p.suffix.lower() != ".csv"]
    if non_csv:
        return dest
    subdirs = [d for d in dest.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        root_csvs = [p for p in dest.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
        for csv_file in root_csvs:
            shutil.move(str(csv_file), str(subdirs[0] / csv_file.name))
        return subdirs[0]
    return dest
