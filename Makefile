PY := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest

setup:
	python3.12 -m venv .venv || python3 -m venv .venv
	$(PIP) install -q -U pip
	$(PIP) install -q -e ".[web,dev]"
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

webapp-build:  ## build the React SPA into webapp/dist
	cd webapp && npm install && npm run build

web: webapp-build  ## serve API + built SPA on :8000
	.venv/bin/uvicorn --factory dashboard.app:create_app --host 0.0.0.0 --port 8000

web-dev:  ## API with reload; pair with `cd webapp && npm run dev` for HMR
	.venv/bin/uvicorn --factory dashboard.app:create_app --reload --port 8000
