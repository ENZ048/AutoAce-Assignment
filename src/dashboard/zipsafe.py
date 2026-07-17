"""Zip-slip-safe extraction.

Exists because ZipFile.extractall lacks member-path safety and we extract into
the job directory, not a tempdir. Batch-root resolution is shared with the CLI
(autoace_audio.batch.resolve_batch_root) so both surfaces agree on the root."""

import shutil
import zipfile
from pathlib import Path

from autoace_audio.batch import resolve_batch_root


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
    return resolve_batch_root(dest)
