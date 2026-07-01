# RescueVision MLOps — Project Specification

> **Tagline:** *From drone feed to inference graph — an end-to-end MLOps platform for search-and-rescue victim detection, built for production.*

---

## 1. Product Overview

**RescueVision** is a production-grade MLOps platform for real-time **person detection from drone footage** in search-and-rescue (SAR) operations. It is not a one-off notebook — it is a **repeatable, auditable, observable ML pipeline** that trains a YOLOv8n model, quantizes it for edge deployment, benchmarks it against quality gates, registers it in a model registry, and serves it behind a hardened FastAPI inference server with full observability.

The platform is designed for **3 machine profiles**: laptop (CPU), Mac Mini M4 (MPS), and RTX 4060 lab (CUDA) — making it viable for field deployments where hardware varies.

**Version:** 2.0.0  
**Tracking:** DagsHub (DVC remote + MLflow tracking server)  
**License:** Proprietary

---

## 2. The Pipeline — 5 Stages + 1 Promotion Gate

The entire ML lifecycle is codified as a **DVC pipeline DAG** — each stage is deterministic, cacheable, and reproducible.

```
data → train → export → benchmark → register → promote
```

### Stage 1 — Data Ingest & Validation (`pipeline/stage1_data.py`)
- Pulls versioned COCO person subset from DagsHub via DVC
- Validates dataset integrity: structure, image/label count parity, YOLO label format (class ID ≥ 0, coordinates ∈ [0,1]), 10 random image corruption checks via PIL `verify()`
- Debug mode generates synthetic dummy data (no DVC pull) so the full pipeline runs on any machine
- Subset mode uses a smaller dataset slice for laptops / resource-constrained environments
- Writes a deterministic `dataset_hash.txt` for downstream traceability

### Stage 2 — Training (`pipeline/stage2_train.py`)
- Trains YOLOv8n via Ultralytics API on the validated dataset
- Logs every hyperparameter + metric to MLflow (mAP50, mAP50-95, precision, recall, training time)
- Device-adaptive: CUDA (RTX 4060), MPS (Mac Mini), or CPU fallback
- Checkpoint saved to `artifacts/` and registered as an MLflow artifact

### Stage 3 — Export & Quantization (`pipeline/stage3_export.py`)
- Converts `best.pt` into **3 edge-optimized formats**:
  | Format | Size | Quantization |
  |--------|------|-------------|
  | ONNX FP32 | ~12 MB | None |
  | ONNX INT8 | ~3.2 MB | Dynamic quantization via `onnxruntime.quantization` |
  | TFLite INT8 | ~3 MB | Full integer quantization via Ultralytics export |
- TFLite export is graceful — skipped transparently if TensorFlow is absent
- All artifacts are logged to MLflow for the current run

### Stage 4 — Benchmark (`pipeline/stage4_benchmark.py`)
- Runs ONNX Runtime inference on 100 validation images per format (FP32, INT8)
- Measures: mean latency, p95 latency, model size on disk
- Computes actual mAP50 via YOLO validation backend
- Writes a structured `benchmark_report.json` consumed by Stage 5

### Stage 5 — Registration & Metric Gate (`pipeline/stage5_register.py`)
- **Quality gate:** `mAP50(INT8) ≥ mAP50(FP32) × 0.97` — if the quantized model degrades more than 3%, the pipeline **fails**. No degraded model reaches production.
- On pass: registers all 3 formats to MLflow Model Registry with tags (`git_commit`, `dataset_hash`, `pipeline`, `stage`)
- On fail: exits 1, sends Telegram alert
- Registered model names: `rescuevision-onnx-fp32`, `rescuevision-onnx-int8`, `rescuevision-tflite-int8`

### Stage Promote — Staging → Production (`pipeline/stage_promote.py`)
- Compares Staging vs Production mAP50; promotes if regression ≤ 1%
- **Auto-rollback:** optionally polls live `/health` endpoint — if error rate crosses 5%, automatically re-promotes the previous Production version
- Sends Telegram notification on success/failure/rollback

---

## 3. Inference Server Architecture

A **hardened FastAPI server** (port 8080, 1 worker — ONNX sessions are not fork-safe) with layered architecture:

```
Request (image bytes)
  → ImagePreprocessor (decode, letterbox, normalize → NCHW float32)
    → ONNXRunner (thread-safe InferenceSession with ORT_ENABLE_ALL)
      → DetectionPostprocessor (NMS, threshold, xyxy → structured dicts)
        → PredictionLogger (JSONL append)
          → MetricsRecorder (Prometheus histograms)
            → Response
```

### Internal Layering (Dependency-Inversion)
```
constants/ → entity/ → config/ → components/ → pipeline/ → router/ → main.py
```
Each layer depends only on layers below it. Every layer is independently testable.

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Upload image → receive detections (class, confidence, bbox) |
| `/health` | GET | Liveness probe (always 200) |
| `/ready` | GET | Readiness probe (200 only after model loaded) |
| `/model/info` | GET | Full model metadata (name, version, format, input shape) |
| `/model/version` | GET | Current model version string |
| `/metrics` | GET | Prometheus scrape target |

### Security Hardening
- **Rate limiting:** 10 req/min per IP via `slowapi`
- **API key auth:** Optional `X-API-Key` header validation
- **Content-length check:** Rejects >10 MB before reading bytes
- **MIME validation:** Only `image/jpeg`, `image/png`, `image/webp`
- **One worker:** ONNX session thread-safety enforced at architecture level

---

## 4. Infrastructure & Deployment

### Containerization
```
Dockerfile           → python:3.12-slim, non-root appuser, healthcheck every 10s
docker-compose.yml   → 3 services:
  ├── rescuevision-serve   (FastAPI, port 8080)
  ├── prometheus            (v2.51.0, 7-day retention, port 9090)
  └── grafana               (10.4.0, auto-provisioned dashboard, port 3000)
```

### Kubernetes (`k8s/`)
- Deployment manifest with resource limits
- Horizontal Pod Autoscaler (HPA) config
- ClusterIP Service for internal routing

### Monitoring Stack

**Prometheus** scrapes `/metrics` every 15s with 5 alert rules:

| Alert | Condition | What it catches |
|-------|-----------|----------------|
| `InferenceLatencyP95High` | p95 > 150ms for 2m | Model degradation or resource pressure |
| `RequestLatencyP95High` | p95 > 500ms for 2m | Network or preprocessing bottleneck |
| `HighErrorRate` | 5xx rate > 5% | Serving failures |
| `NoIncomingTraffic` | 0 requests for 5m | Downstream pipeline or routing issue |
| `DetectionRateDrop` | Avg detections/request ↓ >70% vs yesterday | **Model drift proxy** — detects data drift without ground truth |

**Grafana dashboard** — auto-provisioned with panels:
- Request Rate (RPS)
- Error Rate (5xx %)
- p95 Latency
- Average Detections per Request
- Latency Percentiles (p50/p95/p99)
- End-to-End Request Duration
- Status Code Distribution
- Detections Histogram

### Metrics Instrumented

| Metric | Type | Labels |
|--------|------|--------|
| `rescuevision_requests_total` | Counter | method, path, status_code |
| `rescuevision_request_duration_seconds` | Histogram | method, path |
| `rescuevision_inference_latency_seconds` | Histogram | — |
| `rescuevision_detections_per_request` | Histogram | — |
| `rescuevision_model_info` | Info | version, format |
| `rescuevision_startup_timestamp_seconds` | Gauge | — |

---

## 5. CI/CD — 4 Jobs, 2 Environments

The entire pipeline is codified in `.github/workflows/pipeline.yml`.

```
[push]               quality (ruff + mypy)
                       ↓
[push]               test (pytest --cov-fail-under=70)
                       ↓
[push → main]        pipeline-staging (DVC pull → Stage 1-5 → MLflow Staging)
                       ↓
[manual dispatch]    promote-production (Stage Promote → Production)
```

**Branch strategy:**
- `feat/*`, `fix/*` → lint + test only
- `main` → full pipeline → register to Staging
- Manual trigger → promote to Production

**Pre-commit hooks:** Ruff lint+format (auto-fix), mypy, trailing whitespace, EOF fixer, YAML/TOML validation, no-commit-to-branch (blocks direct `main` commits), detect-secrets.

---

## 6. Testing — 5-Level Pyramid

| Level | Scope | Tools | Dependencies |
|-------|-------|-------|-------------|
| **Unit** | Preprocessor, postprocessor, metric gate (11 cases), label validation (11 cases), prediction logger | pytest | None |
| **API** | Health, predict (valid/invalid/corrupt), security (auth, rate limit, content length) | FastAPI TestClient + mock ONNXRunner | None |
| **Pipeline** | Data integrity, training checkpoint, export artifacts, benchmark schema, registry tags | pytest + DVC + MLflow | Dataset + trained artifacts |
| **Smoke** | 9 curl-based endpoint checks against running server | bash, curl | Running server |
| **Load** | 50 users, 60s, SLO: p95 < 200ms, error rate < 1% | Locust | Running server |

**Coverage floor:** 70% (`fail_under = 70` in `pyproject.toml`)

---

## 7. Key Technical Decisions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| **ONNX Runtime** over raw PyTorch serving | 3-5× faster inference, smaller memory footprint, no PyTorch in production container |
| **DVC** over plain S3/GS | Pipeline DAG with dependency tracking, automatic incremental re-execution on `dvc repro` |
| **Two config files** (`params.yaml` + `train_config.yaml`) | DVC tracks only the 3 values it needs for change detection; full readable config lives separately |
| **1 worker** (not `uvicorn --workers N`) | ONNX `InferenceSession` is not fork-safe — multi-worker requires per-process sessions or inter-process locking |
| **Stop-the-world NMS** in Python | Keeps dependency footprint minimal; swaps to vectorized NMS if throughput becomes a bottleneck |
| **JSONL prediction log** | Append-only, crash-safe, no DB needed; amenable to log shippers (Fluentd, Vector) |
| **Model drift proxy via detection count** | No ground truth available in production — average detections per request serves as a surprisingly effective canary |

---

## 8. Design Language & Brand Pillars

For the landing page design (to be handed off to Kimi/designer):

### Visual Identity
- **Hero metaphor:** A drone silhouette / top-down thermal view transitioning into a clean data pipeline graph. The story is: *from aerial feed to production inference in 5 stages.*
- **Color palette:** Deep navy / slate (`#0F172A`) as primary → suggests reliability, depth, field operations. Accent of amber or electric blue for data flow highlights.
- **Typography:** Inter or SF Pro-style — clean, technical, modern.
- **Imagery:** Split between field photography (drone shots, SAR teams) and clean UI mockups (Grafana dashboard, pipeline DAG, model registry). The landing page should alternate between *human context* and *technical clarity*.

### Key Numbers to Feature (Hero Section)
- **5 stages** — from data to deployment
- **3 model formats** — ONNX FP32, ONNX INT8, TFLite INT8
- **97% accuracy retention** — quantization quality gate
- **<200ms p99 inference** — on CPU
- **5 alert rules** — proactive monitoring
- **70% test coverage** — enforced
- **3 machine profiles** — laptop, Mac Mini, GPU server

### Layout Sections (Suggested)
1. **Hero** — Tagline + platform diagram (the DVC pipeline DAG as a visual)
2. **The Pipeline** — 5 stages, horizontally scrollable or stepped, each with icon + metric
3. **Inference** — Architecture diagram showing request flow through preprocessor → ONNX → postprocessor → response
4. **Observability** — Grafana dashboard mockup + Prometheus alert grid
5. **Deployment** — Docker Compose → Kubernetes → multi-machine
6. **Quality** — The testing pyramid + metric gate + 70% coverage badge
7. **Open-source / Get Started** — `docker compose up`, links to docs

### Tone
- **Confident but not arrogant** — "Built for search-and-rescue teams who need their model to work in the field, not just in the notebook."
- **Technical but accessible** — Explain the quantization gate in one sentence: *"We guarantee the compressed model stays within 3% of the original's accuracy."*
- **Human-first** — Every technical section should connect back to the mission: finding people faster.

---

## 9. File Structure (Landing Page Relevant)

```
├── pipeline/            # 5 + 1 stages — the core DAG
├── app/                 # FastAPI inference server
│   ├── components/      # preprocessor, runner, postprocessor, metrics, prediction_logger
│   ├── router/          # predict, health, model info, metrics
│   ├── pipeline/        # PredictionPipeline (wires components)
│   ├── schema/          # Pydantic request/response models
│   └── config/          # ConfigurationManager
├── core/                # Shared: config, mlflow_client, logger
├── monitoring/          # Prometheus config + alerts, Grafana dashboard JSON
├── k8s/                 # Deployment + HPA + Service manifests
├── docs/                # 7 documentation files (architecture, pipeline, serving, etc.)
├── tests/               # 5-level test suite
├── scripts/             # smoke_test.sh, locustfile.py, create_subset.py
├── .github/workflows/   # CI/CD: 4 jobs, 2 environments
├── dvc.yaml             # Pipeline DAG definition
├── params.yaml          # DVC-tracked parameters
├── docker-compose.yml   # 3 services: server + Prometheus + Grafana
└── Dockerfile           # python:3.12-slim, non-root, healthcheck
```

---

## 10. Quick Reference — Key Commands

| Action | Command |
|--------|---------|
| Start server | `docker compose up rescuevision-serve` |
| Full stack | `docker compose up` |
| Train | `dvc repro` (or `python pipeline/stage2_train.py`) |
| Run tests | `pytest tests/serve/ --cov=app --cov-fail-under=70` |
| Smoke test | `bash scripts/smoke_test.sh` |
| Load test | `locust -f scripts/locustfile.py --headless -u 50 -r 5 -t 60s` |
| Promote model | Manual GitHub Actions dispatch → `promote-production` |
| Lint | `ruff check .` |
| Type check | `mypy .` |

---

> *"We don't ship notebooks. We ship pipelines."*
