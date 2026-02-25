.PHONY: format lint ruff-fix ruff-check mypy \
	test test-coverage \
	docker-build docker-run docker-build-aws docker-run-aws

format: ruff-fix

ruff-fix:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

lint: ruff-check mypy

ruff-check:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

mypy:
	uv run mypy src/ tests/

test:
	uv run pytest tests/unit/ -svv

test-coverage:
	uv run pytest tests/unit/ --cov=src/ --cov=src --cov-report=term-missing

test-integration:
	uv run pytest tests/integration/ -svv

GIT_HASH := $(shell git rev-parse HEAD 2> /dev/null)
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2> /dev/null)
GIT_URL := $(shell git config --get remote.origin.url 2> /dev/null)
GIT_TAG := $(shell git describe --tags --abbrev=0 2> /dev/null || echo "no-tag") # latest tag
docker-build:
	cd docker && docker compose build hsr_data_converter \
		--build-arg GIT_HASH=${GIT_HASH} --build-arg GIT_BRANCH=${GIT_BRANCH} \
		--build-arg GIT_URL=${GIT_URL} --build-arg GIT_TAG=${GIT_TAG}

docker-run:
	cd docker && docker compose run hsr_data_converter

docker-build-aws:
	docker build -f docker/Dockerfile.aws -t robot-data-pipeline:${GIT_HASH} \
		--build-arg GIT_HASH=${GIT_HASH} --build-arg GIT_BRANCH=${GIT_BRANCH} \
		--build-arg GIT_URL=${GIT_URL} --build-arg GIT_TAG=${GIT_TAG} .

docker-run-aws:
	docker run -it --rm robot-data-pipeline:${GIT_HASH} /bin/bash
