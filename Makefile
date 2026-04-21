.PHONY: install lint format test run train clean reset docker-build docker-run

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
	@python -c "import shutil, glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True) + ['.pytest_cache', '.ruff_cache', 'build', 'dist'] + glob.glob('**/*.egg-info', recursive=True)]"

reset: clean
	@python -c "import shutil, os; shutil.rmtree('mlruns', ignore_errors=True); shutil.rmtree('models', ignore_errors=True); os.makedirs('models', exist_ok=True); open('models/.gitkeep', 'w').close()"

docker-build:
	docker build -t churn_predictor_api:latest .

docker-run:
	docker run -p 8000:8000 churn_predictor_api:latest
