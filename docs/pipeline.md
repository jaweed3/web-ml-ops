# Pipeline Reference — Phase 1

The ML pipeline has five stages, each in its own file under `pipeline/`. Every stage can be run independently or through DVC.

```
Stage 1: Data       →  Stage 2: Train  →  Stage 3: Export
                                              ↓
Stage 5: Register   ←  Stage 4: Benchmark
```

---

## Running the pipeline

```bash
# Full pipeline
make all

# Individual stages
make data       # Stage 1
make train      # Stage 2
make export     # Stage 3
make benchmark  # Stage 4
make register   # Stage 5

# DVC — reruns only stages whose inputs changed
dvc repro
```

---

## Stage 1 — Data versioning

**File:** `pipeline/stage1_data.py`

Pulls the versioned dataset from DagsHub via DVC, then validates it before any downstream stage runs.

**What it validates:**
- Required directory structure exists (`images/train`, `images/val`, `labels/train`, `labels/val`)
- Image count matches label count (no orphaned annotations)
- Spot-checks 10 random images for corruption via PIL verify

**Inputs:** `data/coco_person.dvc` (DVC pointer)  
**Outputs:** `data/coco_person/` (populated dataset directory)

**Fails fast if:**
- DVC pull fails
- Any required directory is missing
- Image/label count mismatch
- A sampled image is corrupt

---

## Stage 2 — Training

**File:** `pipeline/stage2_train.py`

Trains YOLOv8n via the Ultralytics API. Every run is tracked in MLflow on DagsHub.

**Logged to MLflow:**
- Params: model name, epochs, imgsz, batch, lr0, optimizer
- Metrics: mAP50, mAP50-95, precision, recall, train_time_seconds
- Artifact: `runs/train/weights/best.pt`

**Inputs:** `data/coco_person/`, `configs/train_config.yaml`, `params.yaml`  
**Outputs:** `runs/train/weights/best.pt`

---

## Stage 3 — Export & quantization

**File:** `pipeline/stage3_export.py`

Converts `best.pt` into three edge-ready formats.

| Format | Method | Output |
|---|---|---|
| ONNX FP32 | Ultralytics export API | `artifacts/model.onnx` |
| ONNX INT8 | `onnxruntime.quantization.quantize_dynamic` | `artifacts/model_int8.onnx` |
| TFLite INT8 | Ultralytics export with `int8=True` | `artifacts/model_int8.tflite` |

**Inputs:** `runs/train/weights/best.pt`  
**Outputs:** all three files in `artifacts/`

---

## Stage 4 — Benchmark

**File:** `pipeline/stage4_benchmark.py`

Runs inference on 100 val images per ONNX format and records objective metrics.

**Metrics recorded per format:**
- `mean_latency_ms`
- `p95_latency_ms`
- `model_size_mb`

**Output:** `artifacts/benchmark_report.json` + all metrics logged to MLflow with prefix (`onnx_fp32_`, `onnx_int8_`)

---

## Stage 5 — Model registry

**File:** `pipeline/stage5_register.py`

Applies a metric gate before registering anything.

**Gate logic:**
```
mAP50_int8 >= mAP50_fp32 × 0.97   →  PASS — register all three artifacts
mAP50_int8 <  mAP50_fp32 × 0.97   →  FAIL — exit 1, CI job fails
mAP50 missing from benchmark       →  skip gate, register directly
```

If gate passes, all three model artifacts are registered to MLflow Model Registry with tags:
- `git_commit` — short SHA
- `pipeline` — `rescuevision-mlops`

---

## Config files

### `configs/train_config.yaml` — full config

```yaml
data:
  dir: data/coco_person
  yaml: data/coco_person/dataset.yaml

model:
  name: yolov8n.pt

train:
  epochs: 50
  imgsz: 640
  batch: 16
  lr0: 0.01
  optimizer: AdamW
```

### `params.yaml` — DVC-tracked subset

Only the values DVC watches to decide whether to re-run a stage:

```yaml
train:
  epochs: 50
  imgsz: 640
  batch: 16
```

Changing any value here triggers downstream stages automatically on the next `dvc repro`.

---

## Logging format

Every stage outputs structured JSON to stdout:

```json
{"timestamp": "2026-04-08T14:23:01+00:00", "level": "INFO", "logger": "stage4_benchmark", "event": "benchmark_result", "format": "onnx_fp32", "mean_latency_ms": 42.3, "p95_latency_ms": 58.1, "model_size_mb": 6.2}
{"timestamp": "2026-04-08T14:23:44+00:00", "level": "INFO", "logger": "stage4_benchmark", "event": "benchmark_result", "format": "onnx_int8", "mean_latency_ms": 21.7, "p95_latency_ms": 29.4, "model_size_mb": 3.1}
```

No `print()`, no `logging.basicConfig()`. All log lines are machine-readable and greppable.
