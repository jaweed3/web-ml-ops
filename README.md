# RescueVision MLOps

End-to-end ML pipeline for SAR victim detection — from versioned dataset to edge-ready model artifacts and a production inference server.

```
dataset (DVC)
  → train YOLOv8n (MLflow)
  → export ONNX FP32 / INT8 + TFLite INT8
  → benchmark + metric gate
  → MLflow Model Registry (Staging → Production)
  → FastAPI inference server
  → Prometheus + Grafana
```

> **Testing status:** All stages verified end-to-end via `make debug`. Unit tests cover the serving layer (74% coverage, enforced), metric gate logic, export shapes, and benchmark SLOs.

---

## Artifacts produced

| File | Format | Target |
|---|---|---|
| `artifacts/model.onnx` | ONNX FP32 | Laptop / Raspberry Pi |
| `artifacts/model_int8.onnx` | ONNX INT8 | Edge CPU |
| `artifacts/model_int8.tflite` | TFLite INT8 | Raspberry Pi / mobile |
| `artifacts/benchmark_report.json` | JSON | CI gate + MLflow |
| `artifacts/training_metrics.json` | JSON | mAP50 for metric gate |
| `artifacts/dataset_hash.txt` | text | Dataset lineage tag |

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <your-repo-url>
cd rescuevision-mlops

# Install base deps (no torch, no ultralytics)
make install-dev

# Create .env and fill in your DagsHub credentials
cp .env .env.local   # or create .env from scratch — see docs/setup.md

# Run the full ML pipeline via DVC
dvc repro

# Start the inference server + monitoring stack
make up
```

Server → http://localhost:8080/docs  
Grafana → http://localhost:3000 (admin / rescuevision)

---

## Running individual pipeline stages

```bash
# Stage 1 — data pull + validation
uv run python -m pipeline.stage1_data

# Stage 2 — training (requires make install-train)
uv run python -m pipeline.stage2_train

# Stage 3 — export & quantization (verified)
uv run python -m pipeline.stage3_export

# Stage 4 — benchmark
uv run python -m pipeline.stage4_benchmark

# Stage 5 — register to MLflow (Staging)
uv run python -m pipeline.stage5_register --stage Staging

# Promote Staging → Production (with mAP50 regression guard)
uv run python -m pipeline.stage_promote

# Debug mode — runs stages 1-3 with dummy data, no GPU required
make debug
```

---

## Documentation

| Doc | What's in it |
|---|---|
| [docs/setup.md](docs/setup.md) | Prerequisites, install, env vars, DVC remote |
| [docs/pipeline.md](docs/pipeline.md) | Phase 1 — stages 1-5 reference |
| [docs/serving.md](docs/serving.md) | Phase 2 — API reference, Docker, architecture |
| [docs/monitoring.md](docs/monitoring.md) | Prometheus metrics, Grafana dashboard |
| [docs/testing.md](docs/testing.md) | How to test everything, start to finish |
| [docs/ci-cd.md](docs/ci-cd.md) | CI/CD jobs, environments, secrets |
| [docs/architecture.md](docs/architecture.md) | Architecture decisions and trade-offs |

---

## Repo layout

```
rescuevision-mlops/
├── pipeline/          # Stage 1-5 scripts
├── core/              # Logger, config loader, MLflow helpers
├── app/               # FastAPI serving layer
│   ├── components/    # ModelLoader, Preprocessor, Postprocessor, Runner, Metrics
│   ├── config/        # ConfigurationManager
│   ├── constant/      # All constants
│   ├── entity/        # Dataclass config entities
│   ├── pipeline/      # PredictionPipeline
│   ├── router/        # FastAPI routes
│   ├── schema/        # Pydantic request/response models
│   ├── middleware/     # Request logger + metrics hook
│   └── utils/         # Shared utilities
├── configs/           # train_config.yaml, serve_config.yaml
├── monitoring/        # Prometheus scrape config + Grafana provisioning
├── tests/
│   ├── serve/             # API + component unit tests (mocked, 74% coverage)
│   ├── test_metric_gate.py  # Pure logic tests for metric gate
│   ├── test_data.py
│   ├── test_export.py
│   ├── test_benchmark.py
│   └── test_register.py   # MLflow registry smoke tests (needs DAGSHUB_* env)
├── research/          # Jupyter notebooks
├── scripts/           # smoke_test.sh
├── docs/              # All documentation
├── dvc.yaml           # Pipeline DAG
├── pyproject.toml     # Ruff, mypy, pytest config
└── .pre-commit-config.yaml
```
