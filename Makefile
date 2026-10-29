# Reproduce everything: `make all` (data download ~0.3 GB, ~30-45 min on a 10-core workstation).
PY      := .venv/bin/python
EXPKIT  := .venv/bin/expkit
RESULTS := results

.PHONY: all setup data cupac lint typecheck test validate reports figures acceptance clean

all:
	@mkdir -p $(RESULTS); date +%s > $(RESULTS)/.make_start
	$(MAKE) setup data cupac lint test validate reports figures
	@$(PY) -c "import json,time; s=int(open('$(RESULTS)/.make_start').read()); json.dump({'exit':0,'seconds':int(time.time())-s}, open('$(RESULTS)/make_status.json','w'))"
	$(MAKE) acceptance

setup:
	@test -d .venv || uv venv --python 3.12 .venv
	uv pip install --python $(PY) -q -e ".[dev]" gsdesign==0.1.5

data:            ## download Criteo (HF Hub), ASOS (OSF API), Hillstrom; write Parquet + row counts
	$(EXPKIT) data

cupac:           ## cross-fitted LightGBM CUPAC covariates + predicted exposure (~15 min)
	$(EXPKIT) cupac

lint:            ## ruff + mypy; exit codes recorded for acceptance E4
	@mkdir -p $(RESULTS)
	-.venv/bin/ruff check src tests; echo $$? > $(RESULTS)/.ruff
	-.venv/bin/ruff format --check src tests; echo $$? > $(RESULTS)/.fmt
	-.venv/bin/mypy src; echo $$? > $(RESULTS)/.mypy
	@$(PY) -c "import json; r={k:int(open('$(RESULTS)/.'+k).read()) for k in ('ruff','fmt','mypy')}; json.dump({'ruff check':r['ruff'],'ruff format --check':r['fmt'],'mypy src':r['mypy']}, open('$(RESULTS)/lint.json','w')); print(r)"

typecheck:
	.venv/bin/mypy src

test:            ## unit + property + cross-check tests (+ real-data scorecard tests)
	.venv/bin/pytest --cov=expkit --cov-report=json:$(RESULTS)/coverage.json --cov-report=term:skip-covered \
		--junitxml=$(RESULTS)/junit.xml

validate:        ## real-data validation suites -> results/*.json
	$(EXPKIT) validate all

reports:         ## HTML experiment readouts
	$(EXPKIT) analyze --config experiments/criteo.yaml
	$(EXPKIT) analyze --config experiments/criteo_srm_bug.yaml

figures:
	$(PY) -m expkit.analyses.figures

acceptance:      ## PASS/FAIL table from results/*.json
	$(EXPKIT) acceptance

clean:
	rm -rf reports/test_*.html .pytest_cache .mypy_cache .ruff_cache
