SHELL := /bin/bash
UV := $(shell command -v uv 2> /dev/null)
DEBUG ?= false

.PHONY: all data train export benchmark register test clean \
        run build up down logs smoke test-serve test-pipeline \
        lint format typecheck quality pre-commit-install \
		install install-train

.PHONY: help install quality test quality pipeline-full clean serve docker-up docker-down

help: ## Helper Message 
	@grep -E '^[a-zA-Z_-]+:.*?## .**$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' 

# ── Install ───────────────────────────────────────────────────────────────────
install:
	@if [ -z "$(UV)" ]; then echo "uv not found. Install uv firstly."; exit 1; fi
	uv sync --all-groups

install-train:
	${PIP} install -r requirements.txt -r requirements-train.txt

install-debug:
	${PIP} install -r requirements-debug.txt -r requirements.txt
# ── Phase 1: MLOps pipeline ───────────────────────────────────────────────────
all: data train export benchmark register

debug:
	DEBUG_MODE=true uv run python -m pipeline.stage1_data
	DEBUG_MODE=true uv run python -m pipeline.stage2_data

debug-train:
	DEBUG_MODE=true uv run python -m pipeline.stage2_data

.data_ready: pipeline/stage1_data.py
	DEBUG_MODE=$(DEBUG) uv run python -m pipeline.stage1_data.py
	@touch .data_ready

.model_trained: .data_ready pipeline/stage2_train.py
	DEBUG_MODE=$(DEBUG) uv run python -m pipeline.stage2_data.py
	@touch .model_trained

.benchmark_passed: .model_trained pipeline/stage4_benchmark.py
	DEBUG_MODE=$(DEBUG) uv run python -m pipeline.stage4_benchmark.py
	@touch .benchmark_passed

.model_exported: .benchmark_passed pipeline/stage3_export.py
	DEBUG_MODE=$(DEBUG) uv run python -m pipeline.stage3_export.py
	@touch .model_exported

pipeline-full: quality .model_exported # Executed from model_export phase
	uv run python -m pipeline.stage5_register

# ── Phase 2: Serving layer ────────────────────────────────────────────────────
run:
	uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

build:
	docker build -t rescuevision-serve .

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

smoke:
	bash scripts/smoke_test.sh

# ── Tests ─────────────────────────────────────────────────────────────────────
test-pipeline:
	uv run pytest tests/ -v --ignore=tests/serve --cov=pipeline --cov=core --cov-report=term-missing

test-serve:
	uv run pytest tests/serve/ -v --cov=app --cov-report=term-missing

test:
	uv run pytest tests/ -v --cov=pipeline --cov=core --cov=app --cov-report=term-missing

# ── Code quality ─────────────────────────────────────────────────────────────

quality:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy app/ core/ pipeline/  --ignore-missing-imports

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf artifacts/ runs/ __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
