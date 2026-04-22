SHELL := /bin/bash
UV := $(shell command -v uv 2> /dev/null)
DEBUG ?= false

.PHONY: help install quality test quality pipeline-full clean serve docker-up docker-down

help: ## Helper Message 
	@grep -E '^[a-zA-Z_-]+:.*?## .**$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' 

# ── Install ───────────────────────────────────────────────────────────────────
install-dev:
	@if [ -z "$(UV)" ]; then echo "uv not found. Install uv firstly."; exit 1; fi
	uv sync --only-group dev

install-train:
	uv sync --all-groups

install-debug:
	uv sync --group dev --group debug

# ── Phase 1: MLOps pipeline ───────────────────────────────────────────────────
all: data train export benchmark register

debug:
	DEBUG_MODE=true uv run python -m pipeline.stage1_data
	DEBUG_MODE=true uv run python -m pipeline.stage2_train
	DEBUG_MODE=true uv run python -m pipeline.stage3_export
	DEBUG_MODE=true uv run python -m pipeline.stage4_benchmark
	DEBUG_MODE=true uv run python -m pipeline.stage5_register

debug-train:
	DEBUG_MODE=true uv run python -m pipeline.stage2_train

# ── Subset mode (Mac Mini / low-resource training) ────────────────────────────
subset: ## Run full pipeline with subset dataset (USE_SUBSET=true)
	USE_SUBSET=true uv run python -m pipeline.stage1_data
	USE_SUBSET=true uv run python -m pipeline.stage2_train
	USE_SUBSET=true uv run python -m pipeline.stage3_export
	USE_SUBSET=true uv run python -m pipeline.stage4_benchmark
	USE_SUBSET=true uv run python -m pipeline.stage5_register

subset-train: ## Train only with subset dataset
	USE_SUBSET=true uv run python -m pipeline.stage2_train

create-subset: ## Create subset on PC Lab (run dvc push after this)
	uv run python scripts/create_subset.py --n_train 300 --n_val 80

.data_ready: pipeline/stage1_data.py
	DEBUG_MODE=$(DEBUG) uv run python -m pipeline.stage1_data
	@touch .data_ready

.model_trained: .data_ready pipeline/stage2_train.py
	DEBUG_MODE=$(DEBUG) uv run python -m pipeline.stage2_train
	@touch .model_trained

.model_exported: .model_trained pipeline/stage3_export.py
	DEBUG_MODE=$(DEBUG) uv run python -m pipeline.stage3_export
	@touch .model_exported

.benchmark_passed: .model_exported pipeline/stage4_benchmark.py
	DEBUG_MODE=$(DEBUG) uv run python -m pipeline.stage4_benchmark
	@touch .benchmark_passed

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

load-test: ## Run Locust load test (requires: make run in another terminal)
	uv run locust -f scripts/locustfile.py \
		--host http://localhost:8080 \
		--users 50 --spawn-rate 5 \
		--run-time 60s --headless \
		--only-summary

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
