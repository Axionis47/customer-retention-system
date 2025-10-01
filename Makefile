PY=python3.11
VENV=.venv
ACTIVATE=. $(VENV)/bin/activate

.PHONY: setup test lint docker-build e2e train-risk train-ppo train-sft train-rm train-ppo-text serve deploy clean \
        data.telco data.bank data.sft data.prefs data.all data.catalog

setup:
	$(PY) -m venv $(VENV)
	$(ACTIVATE) && pip install -U pip setuptools wheel
	$(ACTIVATE) && pip install -e ".[dev]"
	$(ACTIVATE) && pre-commit install

test:
	$(ACTIVATE) && pytest -q --cov=env --cov=agents --cov=serve --cov=models --cov=rlhf --cov=eval --cov-report=term-missing

test-unit:
	$(ACTIVATE) && pytest -q tests/unit/ -v

test-integration:
	$(ACTIVATE) && pytest -q tests/integration/ -v

test-contract:
	$(ACTIVATE) && pytest -q tests/contract/ -v

lint:
	$(ACTIVATE) && ruff check .
	$(ACTIVATE) && mypy .

format:
	$(ACTIVATE) && black .
	$(ACTIVATE) && ruff check --fix .

# Data pipeline targets
data.telco:
	$(ACTIVATE) && python data/processors/telco_processor.py

data.bank:
	$(ACTIVATE) && python data/processors/bank_processor.py

data.sft:
	$(ACTIVATE) && python data/processors/oasst1_processor.py

data.prefs:
	$(ACTIVATE) && python data/processors/preferences_processor.py

data.all: data.telco data.bank data.sft data.prefs
	@echo "✓ All datasets processed"
	$(ACTIVATE) && python -c "from data.catalog_manager import DataCatalog; print(DataCatalog().summary())"

data.catalog:
	$(ACTIVATE) && python -c "from data.catalog_manager import DataCatalog; print(DataCatalog().summary())"

docker-build:
	docker build -f ops/docker/Dockerfile.app -t churn-saver-app:local .
	docker build -f ops/docker/Dockerfile.trainer -t churn-saver-trainer:local .

e2e:
	$(ACTIVATE) && pytest -q tests/e2e/test_docker_smoke.py -v

train-risk:
	$(ACTIVATE) && python models/risk_accept/train_churn.py
	$(ACTIVATE) && python models/risk_accept/train_accept.py
	$(ACTIVATE) && python models/risk_accept/calibrate.py

train-ppo:
	$(ACTIVATE) && python agents/ppo_policy.py --config ops/configs/ppo.yaml

train-sft:
	$(ACTIVATE) && python rlhf/sft_train.py --config ops/configs/sft.yaml

train-rm:
	$(ACTIVATE) && python rlhf/rm_train.py --config ops/configs/rm.yaml

train-ppo-text:
	$(ACTIVATE) && python rlhf/ppo_text.py --config ops/configs/ppo_text.yaml

serve:
	$(ACTIVATE) && uvicorn serve.app:app --host 0.0.0.0 --port 8080 --reload

deploy:
	gcloud builds submit --config ops/cloudbuild.yaml \
		--substitutions=_ENV=dev,_REGION=$$GCP_REGION,_SERVICE_NAME=$$SERVICE_NAME

clean:
	rm -rf $(VENV)
	rm -rf .pytest_cache
	rm -rf .coverage htmlcov
	rm -rf .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

