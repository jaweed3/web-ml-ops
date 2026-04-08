.PHONY: all data train export benchmark register test clean \
        run build up down logs smoke test-serve test-pipeline \
        lint format typecheck quality pre-commit-install

# ── Phase 1: MLOps pipeline ───────────────────────────────────────────────────
all: data train export benchmark register

data:
	python -m pipeline.stage1_data

train:
	python -m pipeline.stage2_train

export:
	python -m pipeline.stage3_export

benchmark:
	python -m pipeline.stage4_benchmark

register:
	python -m pipeline.stage5_register

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
	pytest tests/ -v --ignore=tests/serve --cov=pipeline --cov=core --cov-report=term-missing

test-serve:
	pytest tests/serve/ -v --cov=app --cov-report=term-missing

test:
	pytest tests/ -v --cov=pipeline --cov=core --cov=app --cov-report=term-missing

# ── Code quality ─────────────────────────────────────────────────────────────
lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy app/ core/ pipeline/ --ignore-missing-imports

quality: lint typecheck

pre-commit-install:
	pre-commit install

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf artifacts/ runs/ __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
