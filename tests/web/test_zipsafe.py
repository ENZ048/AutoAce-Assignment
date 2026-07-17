import zipfile

import pytest

from dashboard.zipsafe import UnsafeZipError, extract_zip


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


@pytest.mark.parametrize("evil", ["../evil.txt", "a/../../evil.txt", "/abs.txt"])
def test_hostile_members_rejected_before_extraction(tmp_path, evil):
    z = make_zip(tmp_path / "b.zip", {"ok.wav": b"RIFF", evil: b"x"})
    dest = tmp_path / "x"
    with pytest.raises(UnsafeZipError):
        extract_zip(z, dest)
    assert not (dest / "ok.wav").exists()  # nothing extracted at all
    assert not (tmp_path / "evil.txt").exists()
