"""Shared harness for the budget-accuracy study (docs/superpowers/specs/
2026-07-17-budget-accuracy-study-design.md). Run logging, cost accounting,
the $10 spend guard, and the 2-of-3 improvement rule. Experiments import
shipping code and pass overrides as arguments -- nothing here mutates
shipping config or fusion."""

import csv
import json
import os
import tempfile
from pathlib import Path

ANCHORS = ["call_001.ogg", "call_002.ogg", "call_003.ogg"]
DATA_DIR = Path("data")
OUT_DIR = Path("out/experiments")

# Live Gemini flash-lite audio rates, $ per 1M tokens (verified 2026-07-16).
GEMINI_LITE_IN = 0.50
GEMINI_LITE_OUT = 1.50


class BudgetExceeded(RuntimeError):
    pass


class SpendGuard:
    """File-backed cumulative spend tracker. check() BEFORE a run with the
    projected cost; add() AFTER with the measured cost."""

    def __init__(
        self,
        state_path: Path = OUT_DIR / "spend.json",
        cap_usd: float = 10.0,
        warn_usd: float = 7.0,
    ) -> None:
        self.state_path = state_path
        self.cap_usd = cap_usd
        self.warn_usd = warn_usd

    def total(self) -> float:
        if self.state_path.exists():
            return float(json.loads(self.state_path.read_text())["total_usd"])
        return 0.0

    def add(self, cost_usd: float) -> None:
        total = self.total() + float(cost_usd)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=self.state_path.parent, delete=False) as f:
            f.write(json.dumps({"total_usd": total}))
            temp_path = f.name
        os.replace(temp_path, self.state_path)

    def check(self, projected_usd: float) -> None:
        projected_total = self.total() + float(projected_usd)
        if projected_total > self.cap_usd:
            raise BudgetExceeded(
                f"projected total ${projected_total:.2f} exceeds cap ${self.cap_usd:.2f}"
            )
        if projected_total > self.warn_usd:
            print(
                f"WARNING: projected study spend ${projected_total:.2f} (cap ${self.cap_usd:.2f})"
            )


def gemini_cost(
    prompt_tokens: int | None,
    output_tokens: int | None,
    in_rate: float = GEMINI_LITE_IN,
    out_rate: float = GEMINI_LITE_OUT,
) -> float:
    return ((prompt_tokens or 0) * in_rate + (output_tokens or 0) * out_rate) / 1e6


def load_truth(labels_path: Path = DATA_DIR / "labels.csv") -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(labels_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("name") and row.get("result_json", "").strip():
                out[row["name"].strip()] = json.loads(row["result_json"])
    return out


def log_run(exp: str, run_idx: int, payload: dict) -> Path:
    if "cost_usd" not in payload:
        raise ValueError("every run log must carry its measured cost")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{exp}_run{run_idx}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def read_runs(exp: str) -> list[dict]:
    def run_index(p: Path) -> int:
        return int(p.stem.split("_run")[1])

    return [
        json.loads(p.read_text()) for p in sorted(OUT_DIR.glob(f"{exp}_run*.json"), key=run_index)
    ]


def field_compare(pred: dict, truth: dict, fields: list[str]) -> dict[str, bool]:
    return {f: pred.get(f) == truth.get(f) for f in fields}


def _correct_count(runs: list[dict], field: str, clip: str) -> tuple[int, int]:
    flags = [
        r["per_clip"][clip]["correct"][field]
        for r in runs
        if clip in r.get("per_clip", {}) and field in r["per_clip"][clip].get("correct", {})
    ]
    return sum(flags), len(flags)


def wins_field(baseline_runs: list[dict], lever_runs: list[dict], field: str, clip: str) -> bool:
    """Spec 2-of-3 rule: lever right >=2/3 where baseline right <=1/3."""
    base_ok, base_n = _correct_count(baseline_runs, field, clip)
    lever_ok, lever_n = _correct_count(lever_runs, field, clip)
    if base_n == 0 or lever_n == 0:
        return False
    return base_ok <= base_n // 3 and lever_ok >= (2 * lever_n + 2) // 3
