"""Score pipeline predictions against ground truth (labels.csv format or
validation_manifest.csv partial-truth format) and emit the markdown report."""

import argparse
import csv
import json
from pathlib import Path

from eval.metrics import field_report


def load_labels(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            blob = row.get("result_json") or row.get("truth") or ""
            if row.get("name") and blob.strip():
                out[row["name"].strip()] = json.loads(blob)
    return out


def main(pred_path: Path, labels_path: Path, out_path: Path | None) -> None:
    preds = json.loads(Path(pred_path).read_text())
    labels = load_labels(labels_path)
    report = field_report(labels, preds)
    print(report)
    if out_path:
        Path(out_path).write_text(report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True, help="out/results.json from the batch CLI")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    main(a.pred, a.labels, a.out)
