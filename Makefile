# Reproduce everything: `make all` (downloads ~0.3 GB; CUPAC and HTE model fits are cached in data/processed).
# `make quick` re-checks code, tests, readouts, docs and acceptance from the cached results in a few minutes.
PY      := .venv/bin/python
EXPKIT  := .venv/bin/expkit
BUILD   := build

.PHONY: all quick setup data cupac lint test validate reports figures docs check-docs acceptance clean

all:
	@mkdir -p $(BUILD); date +%s > $(BUILD)/.make_start
	$(MAKE) setup data cupac lint test validate reports figures docs check-docs
	@$(PY) -c "import json,time; s=int(open('$(BUILD)/.make_start').read()); json.dump({'exit':0,'seconds':int(time.time())-s}, open('$(BUILD)/make_status.json','w'))"
	$(MAKE) acceptance

quick: lint test reports docs check-docs acceptance

setup:
	@test -d .venv || uv venv --python 3.12 .venv
	uv pip install --python $(PY) -q -e ".[dev]" gsdesign==0.1.5

data:            ## download Criteo (HF Hub), ASOS (OSF API), Hillstrom; write Parquet + row counts
	$(EXPKIT) data

cupac:           ## cross-fitted LightGBM CUPAC covariates + predicted exposure (cached)
	$(EXPKIT) cupac

lint:            ## ruff + mypy; exit codes recorded for acceptance E4
	@mkdir -p $(BUILD)
	-.venv/bin/ruff check src tests; echo $$? > $(BUILD)/.ruff
	-.venv/bin/ruff format --check src tests; echo $$? > $(BUILD)/.fmt
	-.venv/bin/mypy src; echo $$? > $(BUILD)/.mypy
	@$(PY) -c "import json; r={k:int(open('$(BUILD)/.'+k).read()) for k in ('ruff','fmt','mypy')}; json.dump({'ruff check':r['ruff'],'ruff format --check':r['fmt'],'mypy src':r['mypy']}, open('$(BUILD)/lint.json','w')); print(r)"

test:            ## unit + property + cross-check tests (+ real-data scorecard tests)
	@mkdir -p $(BUILD)
	.venv/bin/pytest --deselect tests/test_docs_drift.py --cov=expkit --cov-report=json:$(BUILD)/coverage.json \
		--cov-report=term:skip-covered --junitxml=$(BUILD)/junit.xml

validate:        ## real-data validation suites -> results/*.json
	$(EXPKIT) validate all

reports:         ## HTML experiment readouts
	$(EXPKIT) analyze --config experiments/criteo.yaml
	$(EXPKIT) analyze --config experiments/criteo_srm_bug.yaml
	$(EXPKIT) analyze --config experiments/asos_d53f0e.yaml

figures:
	$(PY) -m expkit.analyses.figures

docs:            ## render README and docs/*.md from docs/templates + results/*.json
	$(EXPKIT) docs

check-docs:      ## fail if any rendered doc drifted from results/*.json
	.venv/bin/pytest -q tests/test_docs_drift.py --junitxml=$(BUILD)/junit_docs.xml

acceptance:      ## PASS/FAIL table from results/*.json -> docs/ACCEPTANCE.md
	$(EXPKIT) acceptance

clean:
	rm -rf reports/test_*.html .pytest_cache .mypy_cache .ruff_cache
