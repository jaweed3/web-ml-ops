# Pipeline Reference — Phase 1

The ML pipeline has five stages, each in its own file under `pipeline/`. Every stage can be run independently or through DVC.

```
Stage 1: Data       →  Stage 2: Train  →  Stage 3: Export
                                              ↓
Stage 5: Register   ←  Stage 4: Benchmark
```

> **Testing status:** Stage 3 (export & quantization) is verified. Stages 1, 2, 4, 5 are implemented but not yet end-to-end tested.

---

## Running the pipeline

### Full pipeline via DVC (recommended)

```bash
# Reruns only stages whose inputs changed
dvc repro
```

DVC resolves the DAG defined in `dvc.yaml` and skips stages that are already up to date.

### Individual stages

```bash
uv run python -m pipeline.stage1_data
uv run python -m pipeline.stage2_train
uv run python -m pipeline.stage3_export
uv run python -m pipeline.stage4_benchmark
uv run python -m pipeline.stage5_register
```

### Debug mode (no GPU, no DagsHub)

```bash
# Stages 1-3 with dummy data
make debug

# Or per-stage
DEBUG_MODE=true uv run python -m pipeline.stage1_data
DEBUG_MODE=true uv run python -m pipeline.stage2_train
DEBUG_MODE=true uv run python -m pipeline.stage3_export
```

### Quality check + export + register

```bash
make pipeline-full
# equivalent: ruff + mypy, then export stage (stamp-based), then stage5_register
```

---

## Stage 1 — Data versioning

**File:** `pipeline/stage1_data.py`

Pulls the versioned dataset from DagsHub via DVC, then validates it before any downstream stage runs. When `DEBUG_MODE=true`, skips the DVC pull and generates a synthetic dataset instead.

**What it validates:**
- Required directory structure exists (`images/train`, `images/val`, `labels/train`, `labels/val`)
- Image count matches label count (no orphaned annotations)
- Spot-checks up to 10 random images for corruption via PIL verify

**Inputs:** `data/coco_person.dvc` (DVC pointer), or generates dummy data in debug mode  
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
- Params: model name, epochs, imgsz, batch, lr0, optimizer, device
- Metrics: mAP50, mAP50-95, precision, recall, train_time_seconds
- Artifact: `runs/train/weights/best.pt`

**Inputs:** `data/coco_person/`, `configs/train_config.yaml`, `params.yaml`  
**Outputs:** `runs/train/weights/best.pt`

**Requires:** `make install-train` (ultralytics + torch)

---

## Stage 3 — Export & quantization

**File:** `pipeline/stage3_export.py`

Converts `best.pt` into three edge-ready formats.

| Format | Method | Output |
|---|---|---|
| ONNX FP32 | Ultralytics export API | `artifacts/model.onnx` |
| ONNX INT8 | `onnxruntime.quantization.quantize_dynamic` (QUInt8) | `artifacts/model_int8.onnx` |
| TFLite INT8 | Ultralytics export with `int8=True` | `artifacts/model_int8.tflite` |

TFLite export requires tensorflow. If tensorflow is not installed, the stage logs a warning and skips TFLite — ONNX artifacts are still produced.

**Inputs:** `runs/train/weights/best.pt`  
**Outputs:** all three files in `artifacts/`

---

## Stage 4 — Benchmark

**File:** `pipeline/stage4_benchmark.py`

Runs inference on up to 100 val images per ONNX format and records objective metrics. TFLite is not benchmarked.

**Metrics recorded per format:**
- `mean_latency_ms`
- `p95_latency_ms`
- `model_size_mb`

**Output:** `artifacts/benchmark_report.json` + all metrics logged to MLflow with prefix (`onnx_fp32_`, `onnx_int8_`)

**Inputs:** `artifacts/model.onnx`, `artifacts/model_int8.onnx`, `data/coco_person/images/val`  
**Fails if:** no validation images found

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

If the gate passes, all three model artifacts are registered to MLflow Model Registry with tags:
- `git_commit` — short SHA
- `pipeline` — `rescuevision-mlops`

**Registered model names:**
- `rescuevision-onnx-fp32`
- `rescuevision-onnx-int8`
- `rescuevision-tflite-int8`

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
  device: cuda   # cuda | mps | cpu

debug:
  epochs: 1
  imgsz: 320
  batch: 4
  device: cpu
  max_samples: 50
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

## Logging

Every stage uses `core.logger.get_logger()`. By default logs are rendered with Rich (colored, human-readable). To switch to JSON output:

```bash
LOG_FORMAT=json uv run python -m pipeline.stage3_export
```

JSON log line example:
```json
{"timestamps": "2026-04-08T14:23:01+00:00", "level": "INFO", "logger": "stage4_benchmark", "event": "benchmark_result"}
```

No `print()`, no `logging.basicConfig()`.
