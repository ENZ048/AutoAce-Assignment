import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="autoace_audio")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyze", help="analyze a folder or ZIP of calls (+ optional CSV manifest)")
    p.add_argument("input", type=Path)
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--arm", default=None, help="tone arm: gemini | dimensional | transcript")
    args = parser.parse_args()

    from autoace_audio.batch import run_batch

    def progress(done: int, total: int, name: str) -> None:
        print(f"[{done}/{total}] {name}", flush=True)

    report = run_batch(args.input, args.out, tone_arm=args.arm, progress_cb=progress)
    for w in report.warnings:
        print(f"WARN: {w}", file=sys.stderr)
    for e in report.errors:
        print(f"ERROR: {e.name}: {e.error}", file=sys.stderr)
    print(f"done: {len(report.results)} ok, {len(report.errors)} failed -> {args.out}/")
    return 0 if report.results or not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
