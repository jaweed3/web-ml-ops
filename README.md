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

---

## Artifacts produced

| File | Format | Target |
|---|---|---|
| `artifacts/model.onnx` | ONNX FP32 | Laptop / Raspberry Pi |
| `artifacts/model_int8.onnx` | ONNX INT8 | Edge CPU |
| `artifacts/model_int8.tflite` | TFLite INT8 | Raspberry Pi / mobile |
| `artifacts/benchmark_report.json` | JSON | CI gate + MLflow |

---

## Quickstart

```bash
git clone https://github.com/your-username/rescuevision-mlops
cd rescuevision-mlops
pip install -r requirements.txt
cp .env.example .env   # fill in your DagsHub credentials

# Run the full ML pipeline
make all

# Start the inference server + monitoring stack
make up
```

Server → http://localhost:8080/docs  
Grafana → http://localhost:3000 (admin / rescuevision)

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
│   ├── middleware/    # Request logger + metrics hook
│   └── utils/         # Shared utilities
├── configs/           # train_config.yaml, serve_config.yaml
├── monitoring/        # Prometheus scrape config + Grafana provisioning
├── tests/
│   ├── serve/         # API + component unit tests
│   ├── test_data.py
│   ├── test_export.py
│   └── test_benchmark.py
├── research/          # Jupyter notebooks
├── scripts/           # smoke_test.sh
├── docs/              # All documentation
├── dvc.yaml           # Pipeline DAG
├── pyproject.toml     # Ruff, mypy, pytest config
└── .pre-commit-config.yaml
```
