PY := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest

setup:
	python3.12 -m venv .venv || python3 -m venv .venv
	$(PIP) install -q -U pip
	$(PIP) install -q -e ".[dev]"
	@command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found on PATH — required at runtime"

test:
	$(PYTEST) -q -m "not slow and not network"

test-all:
	$(PYTEST) -q

lint:
	.venv/bin/ruff check src tests eval
	.venv/bin/ruff format --check src tests eval

analyze:
	$(PY) -m autoace_audio analyze $(DIR) --out out/

evaluate:
	$(PY) -m eval.evaluate --pred out/results.json --labels data/labels.csv --out out/validation_report.md

bakeoff:
	$(PY) -m eval.bakeoff --data data/ --out out/bakeoff.md
