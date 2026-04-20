.PHONY: install lint format test run train clean

install:
	pip install --upgrade pip
	pip install -e .[dev]

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:
	pytest

run:
	uvicorn churn_predictor.api.app:app --reload --host 0.0.0.0 --port 8000

train:
	python -m churn_predictor.pipelines.baselines
	python -m churn_predictor.pipelines.neural_network

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/
