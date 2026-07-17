import io
import zipfile

import pytest

from dashboard.zipsafe import UnsafeZipError, _copy_limited, extract_zip


def make_zip(path, entries: dict[str, bytes]):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def test_flat_zip_extracts_to_dest_root(tmp_path):
    z = make_zip(tmp_path / "b.zip", {"call_001.wav": b"RIFF", "labels.csv": b"name,result_json\n"})
    root = extract_zip(z, tmp_path / "x")
    assert root == tmp_path / "x"
    assert (root / "call_001.wav").read_bytes() == b"RIFF"
    assert (root / "labels.csv").exists()


def test_single_subdir_becomes_root_and_csv_moves_in(tmp_path):
    z = make_zip(
        tmp_path / "b.zip",
        {"batch/call_001.wav": b"RIFF", "labels.csv": b"name,result_json\n"},
    )
    root = extract_zip(z, tmp_path / "x")
    assert root == tmp_path / "x" / "batch"
    assert (root / "call_001.wav").exists()
    assert (root / "labels.csv").exists()  # moved into the subdir
    assert not (tmp_path / "x" / "labels.csv").exists()


def test_uppercase_csv_also_moves_into_subdir_root(tmp_path):
    z = make_zip(
        tmp_path / "b.zip",
        {"batch/call_001.wav": b"RIFF", "LABELS.CSV": b"name,result_json\n"},
    )
    root = extract_zip(z, tmp_path / "x")
    assert root == tmp_path / "x" / "batch"
    assert (root / "LABELS.CSV").exists()
    assert not (tmp_path / "x" / "LABELS.CSV").exists()


def test_finder_zip_macosx_junk_ignored_for_root_resolution(tmp_path):
    """A macOS Finder zip (payload dir + __MACOSX/ + .DS_Store) must still
    resolve the payload dir as the batch root, not the extraction root."""
    z = make_zip(
        tmp_path / "b.zip",
        {
            "batch/call_001.wav": b"RIFF",
            "batch/labels.csv": b"name,result_json\n",
            "__MACOSX/._batch": b"\x00\x05\x16\x07",
            "__MACOSX/batch/._call_001.wav": b"\x00\x05\x16\x07",
            ".DS_Store": b"\x00",
        },
    )
    root = extract_zip(z, tmp_path / "x")
    assert root == tmp_path / "x" / "batch"
    assert (root / "call_001.wav").exists()


def test_zip_bomb_rejected_by_decompressed_size_cap(tmp_path):
    """A zip whose declared decompressed size exceeds the cap is rejected
    before anything is extracted (compressed size is irrelevant)."""
    z = make_zip(tmp_path / "b.zip", {"big.wav": b"\x00" * 65536, "ok.wav": b"RIFF"})
    dest = tmp_path / "x"
    with pytest.raises(UnsafeZipError):
        extract_zip(z, dest, max_extracted_bytes=16384)
    assert not (dest / "ok.wav").exists()  # all-or-nothing


def test_copy_limited_stops_streams_that_exceed_the_declared_size():
    """Defense in depth: even if zip metadata lies about file_size, the
    streaming copy itself enforces the byte budget."""
    src = io.BytesIO(b"\x00" * 1000)
    out = io.BytesIO()
    with pytest.raises(UnsafeZipError):
        _copy_limited(src, out, remaining=100)


@pytest.mark.parametrize("evil", ["../evil.txt", "a/../../evil.txt", "/abs.txt"])
def test_hostile_members_rejected_before_extraction(tmp_path, evil):
    z = make_zip(tmp_path / "b.zip", {"ok.wav": b"RIFF", evil: b"x"})
    dest = tmp_path / "x"
    with pytest.raises(UnsafeZipError):
        extract_zip(z, dest)
    assert not (dest / "ok.wav").exists()  # nothing extracted at all
    assert not (tmp_path / "evil.txt").exists()
