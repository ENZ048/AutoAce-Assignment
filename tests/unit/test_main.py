"""CLI entry-point tests: argument validation and the friendly-error contract for
a bad input path. No real analysis ever runs -- failures happen before analyze_fn
would be reached."""

import sys

import pytest

from autoace_audio.__main__ import main


def test_nonexistent_input_path_exits_2_with_friendly_stderr_message(tmp_path, monkeypatch, capsys):
    """Reviewer finding: a missing/invalid input path must print a clear one-line
    error to stderr and exit(2) -- not dump a raw traceback."""
    missing = tmp_path / "does_not_exist"
    monkeypatch.setattr(
        sys,
        "argv",
        ["autoace_audio", "analyze", str(missing), "--out", str(tmp_path / "out")],
    )

    rc = main()

    assert rc == 2
    captured = capsys.readouterr()
    assert "does_not_exist" in captured.err
    assert "Traceback" not in captured.err


def test_arm_choices_rejects_unknown_arm(tmp_path, monkeypatch):
    """--arm must be validated against the 3 real arms, not silently accepted and
    only discovered to be bogus deep inside the pipeline (base.py's ToneClassifierError)."""
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["autoace_audio", "analyze", str(tmp_path), "--out", str(out_dir), "--arm", "bogus"],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
