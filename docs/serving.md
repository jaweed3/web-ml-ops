# Serving Reference — Phase 2

The inference server is a FastAPI app under `app/`. It pulls the registered model from MLflow at startup and serves predictions via REST API.

---

## Start the server

**Local (development):**

```bash
export $(cat .env | xargs)
make run
# → http://localhost:8080
```

**Docker (production):**

```bash
make up
# inference server → http://localhost:8080
# Prometheus        → http://localhost:9090
# Grafana           → http://localhost:3000
```

---

## API endpoints

Interactive docs: **http://localhost:8080/docs**

### `POST /predict`

Run victim-detection inference on an uploaded image.

**Request:**
```
Content-Type: multipart/form-data
Body: file=<image>   (JPEG / PNG / WebP, max 10 MB)
```

**Response `200`:**
```json
{
  "model_version": "3",
  "model_format": "onnx_int8",
  "inference_time_ms": 21.4,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.87,
      "bbox": {
        "x1": 142, "y1": 88, "x2": 310, "y2": 445,
        "cx": 226, "cy": 266, "width": 168, "height": 357
      }
    }
  ],
  "image_shape": [640, 640],
  "request_id": "req_a3f92b1c"
}
```

**Error responses:**

| Code | When |
|---|---|
| `422` | Wrong MIME type, corrupt image, empty file, or file > 10 MB |
| `503` | Model not loaded yet (during startup) |

---

### `GET /health`

Liveness check. Returns 200 if the process is alive.

```json
{"status": "ok", "timestamp": "2026-04-08T14:23:01+00:00"}
```

---

### `GET /ready`

Readiness check. Returns 200 only when the model is loaded and ready to serve.

```json
{
  "status": "ready",
  "model_version": "3",
  "loaded_at": "2026-04-08T14:22:55+00:00"
}
```

Returns `503` with `{"status": "loading"}` during model download.

---

### `GET /model/info`

Metadata about the currently loaded model.

```json
{
  "name": "rescuevision-onnx-int8",
  "version": "3",
  "format": "onnx_int8",
  "input_shape": [1, 3, 640, 640],
  "loaded_at": 1712584975.0
}
```

---

### `GET /model/version`

```json
{"version": "3"}
```

---

### `GET /metrics`

Prometheus scrape endpoint. Not shown in Swagger docs. See [monitoring.md](monitoring.md).

---

## Startup sequence

```
docker-compose up
  ↓
FastAPI lifespan → startup
  ↓
ConfigurationManager loads configs/serve_config.yaml + env vars
  ↓
ModelLoader authenticates to DagsHub MLflow
  ↓
ModelLoader resolves "latest" → version N
  ↓
ModelLoader downloads artifact to /tmp/rescuevision_model/
  ↓
ONNXRunner initializes session (ORT_ENABLE_ALL optimization)
  ↓
PredictionPipeline wired with InferenceConfig
  ↓
record_model_ready() → Prometheus model info registered
  ↓
GET /ready returns 200 — accepting requests
```

During model download (typically 5–30s), `/health` returns 200 but `/ready` returns 503.

---

## App structure

```
app/
├── constant/__init__.py       all constants — paths, thresholds, class names
├── entity/config_entity.py    frozen dataclasses for all config sections
├── config/configuration.py    ConfigurationManager — single entry point
├── components/
│   ├── model_loader.py        pull artifact from MLflow registry
│   ├── preprocessor.py        bytes → float32 tensor (letterbox resize)
│   ├── postprocessor.py       raw ONNX output → detections + NMS
│   ├── runner.py              ONNX session singleton, thread-safe
│   └── metrics.py             Prometheus metric definitions
├── pipeline/
│   └── prediction_pipeline.py orchestrates preprocessor → runner → postprocessor
├── schema/
│   ├── predict.py             PredictResponse, DetectionSchema, BBoxSchema
│   ├── health.py              HealthResponse, ReadyResponse
│   └── meta.py                ModelInfoResponse, ModelVersionResponse
├── router/
│   ├── predict.py             POST /predict
│   ├── health.py              GET /health, GET /ready
│   ├── meta.py                GET /model/info, GET /model/version
│   └── metrics.py             GET /metrics
├── middleware/
│   └── request_logger.py      structured logging + Prometheus counter/histogram
├── utils/common.py            file I/O, timing, ID helpers
└── main.py                    FastAPI app + lifespan
```

---

## Serving config

`configs/serve_config.yaml` — all defaults. Override any field with environment variables.

```yaml
dagshub:
  username: ""     # override: DAGSHUB_USERNAME
  repo: ""         # override: DAGSHUB_REPO
  token: ""        # override: DAGSHUB_TOKEN

model:
  name: rescuevision-onnx-int8
  version: latest  # or a specific version number e.g. "3"
  format: onnx_int8

inference:
  imgsz: 640
  conf_threshold: 0.25
  iou_threshold: 0.45
  max_detections: 100
  n_threads: 2

server:
  host: 0.0.0.0
  port: 8080
  workers: 1       # keep at 1 — ONNX session is not fork-safe
  max_file_size_mb: 10
```

---

## Request logs

Every request produces structured JSON log lines:

```json
{"timestamp": "...", "level": "INFO", "logger": "middleware.request", "event": "request_received", "request_id": "a3f92b1c", "method": "POST", "path": "/predict"}
{"timestamp": "...", "level": "INFO", "logger": "components.runner", "event": "inference_complete", "request_id": "req_a3f92b1c", "latency_ms": 21.4}
{"timestamp": "...", "level": "INFO", "logger": "pipeline.prediction", "event": "prediction_complete", "request_id": "req_a3f92b1c", "n_detections": 2, "latency_ms": 21.4}
{"timestamp": "...", "level": "INFO", "logger": "middleware.request", "event": "request_complete", "request_id": "a3f92b1c", "status_code": 200, "duration_ms": 23.1}
```
