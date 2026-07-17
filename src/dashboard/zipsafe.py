"""Zip-slip-safe extraction.

Exists because ZipFile.extractall lacks member-path safety and we extract into
the job directory, not a tempdir. Batch-root resolution is shared with the CLI
(autoace_audio.batch.resolve_batch_root) so both surfaces agree on the root."""

import zipfile
from pathlib import Path

from autoace_audio.batch import resolve_batch_root


class UnsafeZipError(ValueError):
    """A zip member path would escape the extraction directory, or the
    archive's decompressed size exceeds the extraction budget."""


def _validate_members(zf: zipfile.ZipFile, dest: Path, max_extracted_bytes: int | None) -> None:
    base = dest.resolve()
    declared_total = 0
    for info in zf.infolist():
        name = info.filename
        if name.startswith(("/", "\\")) or ".." in Path(name).parts:
            raise UnsafeZipError(f"unsafe path in zip: {name!r}")
        if not (dest / name).resolve().is_relative_to(base):
            raise UnsafeZipError(f"unsafe path in zip: {name!r}")
        declared_total += info.file_size
    if max_extracted_bytes is not None and declared_total > max_extracted_bytes:
        raise UnsafeZipError(
            f"archive decompresses to {declared_total} bytes, over the "
            f"{max_extracted_bytes}-byte extraction limit"
        )


def _copy_limited(src, out, remaining: int) -> int:
    """Copy src to out, raising once more than `remaining` bytes are read.
    Guards against zip metadata that under-declares file_size."""
    copied = 0
    while True:
        chunk = src.read(64 * 1024)
        if not chunk:
            return copied
        copied += len(chunk)
        if copied > remaining:
            raise UnsafeZipError("zip member exceeds its declared size and the extraction limit")
        out.write(chunk)


def extract_zip(zip_path: Path, dest: Path, max_extracted_bytes: int | None = None) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        # all-or-nothing: validate paths + declared sizes before extracting anything
        _validate_members(zf, dest, max_extracted_bytes)
        budget = max_extracted_bytes if max_extracted_bytes is not None else float("inf")
        for info in zf.infolist():
            if info.is_dir():
                (dest / info.filename).mkdir(parents=True, exist_ok=True)
                continue
            target = dest / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                budget -= _copy_limited(src, out, budget)
    return resolve_batch_root(dest)
