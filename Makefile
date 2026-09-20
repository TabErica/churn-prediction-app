IMAGE ?= $(shell echo $${DOCKERHUB_USERNAME:-ziqimeng0128})/churn-radar
TAG   ?= latest
DATA  ?= data/sample      # override to run on the full logs: make train DATA=path/to/logs

.PHONY: install lint format test train submission run docker-build docker-run lock

install:            ## Create the dev environment
	pip install -r requirements-dev.txt && pip install -e . && pre-commit install

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

test:
	pytest --cov=churn --cov-report=term-missing

train:              ## Train LR + KNN voting on $(DATA) -> models/model.joblib
	python -m churn.train --data $(DATA)/train.parquet --model "LR + KNN voting" --out models/model.joblib

submission:         ## Score $(DATA)/test.parquet, aligned to its example_submission.csv
	python -m churn.predict --model models/model.joblib --data $(DATA)/test.parquet \
		--template $(DATA)/example_submission.csv --pos-rate 0.5 --out submission.csv

run:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t $(IMAGE):$(TAG) .

docker-run:
	docker run --rm -p 8501:8501 $(IMAGE):$(TAG)

lock:               ## Re-pin runtime dependencies from a clean venv
	python -m venv .lockenv && .lockenv/bin/pip install -e . && .lockenv/bin/pip freeze --exclude-editable > requirements.txt && rm -rf .lockenv
