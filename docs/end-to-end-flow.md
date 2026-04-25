# RescueVision MLOps — End-to-End Flow

Complete reference for how data, code, and models move through the system.

---

## 1. System Overview

```mermaid
graph TD
    subgraph INGEST["Data Ingestion (manual / on-demand)"]
        YT["YouTube / Video Sources\n(data/sources.txt)"]
        COL["collect_frames.py\nextract JPEGs via yt-dlp + ffmpeg"]
        FIL["filter_frames.py\nblur / brightness / size filter"]
        PL["pseudo_label.py\nauto-label with ONNX FP32 model"]
        MRG["merge_dataset.py\ntrain/val split → DVC push"]
    end

    subgraph PIPELINE["Training Pipeline (CI/CD + Dagster)"]
        S1["Stage 1 — Data\npull via DVC, validate, hash"]
        S2["Stage 2 — Train\nYOLOv8n fine-tune → best.pt"]
        S3["Stage 3 — Export\nONNX FP32 + INT8 + TFLite INT8"]
        S4["Stage 4 — Benchmark\nlatency + mAP50 measurement"]
        S5["Stage 5 — Register\nMLflow Model Registry → Staging"]
        SP["Stage Promote\nregression guard → Production"]
    end

    subgraph SERVE["Serving (FastAPI)"]
        API["app/main.py\nFastAPI + ONNX Runtime"]
        ML["ModelLoader\ndownload from MLflow Registry"]
        PRED["PredictionPipeline\npreprocess → infer → postprocess"]
    end

    subgraph MLFLOW["MLflow (DagHub remote)"]
        EXP["Experiments\nparams + metrics + artifacts"]
        REG["Model Registry\nStaging / Production versions"]
    end

    subgraph ORCH["Orchestration (Dagster)"]
        SENS["dataset_change_sensor\npoll dataset_hash.txt every 5m"]
        SCHED["weekly_retrain\nMonday 02:00 WIB"]
        JOBS["staging_job / production_job"]
    end

    YT --> COL --> FIL --> PL --> MRG
    MRG -->|"dvc push → DagHub"| S1
    S1 --> S2 --> S3 --> S4 --> S5 --> SP
    S2 -->|"metrics + artifacts"| EXP
    S5 -->|"register model version"| REG
    SP -->|"transition stage"| REG
    REG -->|"pull on startup"| ML
    ML --> PRED --> API

    SENS -->|"hash changed"| JOBS
    SCHED --> JOBS
    JOBS --> S1

    style INGEST fill:#1a1a2e,stroke:#e94560,color:#eee
    style PIPELINE fill:#16213e,stroke:#0f3460,color:#eee
    style SERVE fill:#0f3460,stroke:#533483,color:#eee
    style MLFLOW fill:#533483,stroke:#e94560,color:#eee
    style ORCH fill:#e94560,stroke:#533483,color:#eee
```

---

## 2. Data Ingestion Pipeline

Run manually when new video data is available. Output feeds directly into training via DVC.

```mermaid
flowchart LR
    SRC["data/sources.txt\n(YouTube URLs)"]

    subgraph COLLECT["scripts/collect_frames.py"]
        DL["yt-dlp download\n(worst quality)"]
        FF["ffmpeg extract\n1 frame / interval_sec"]
        META["metadata.jsonl\nprovenance tracking"]
    end

    subgraph FILTER["scripts/filter_frames.py"]
        BLR["Laplacian blur score\n≥ blur_threshold (100)"]
        BRT["Brightness check\n30 – 225"]
        SZ["Min resolution\n640 × 360"]
    end

    subgraph LABEL["scripts/pseudo_label.py"]
        INFER["ONNX FP32 inference\ninfer_conf_threshold=0.25"]
        GATE{"mean conf\n≥ threshold (0.5)?"}
        AUTO["auto-label\ndata/labeled/"]
        REVIEW["manual review\ndata/pending_review/"]
    end

    subgraph MERGE["scripts/merge_dataset.py"]
        SPLIT["video-group split\nval_ratio=0.20"]
        HASH["dataset_hash.txt\nMD5 of all labels"]
        DVC["dvc push → DagHub"]
    end

    SRC --> DL --> FF --> META
    FF -->|"raw frames"| BLR
    BLR -->|pass| BRT
    BRT -->|pass| SZ
    SZ -->|pass| INFER
    BLR -->|fail| TRASH1["discard"]
    BRT -->|fail| TRASH1
    SZ -->|fail| TRASH1
    INFER --> GATE
    GATE -->|yes| AUTO
    GATE -->|no / 0 detections| REVIEW
    AUTO -->|"labeled images + labels"| SPLIT
    REVIEW -->|"human corrects labels"| SPLIT
    SPLIT --> HASH --> DVC

    style COLLECT fill:#1a1a2e,stroke:#e94560,color:#eee
    style FILTER fill:#16213e,stroke:#0f3460,color:#eee
    style LABEL fill:#0f3460,stroke:#533483,color:#eee
    style MERGE fill:#533483,stroke:#e94560,color:#eee
```

### Key thresholds (all in `configs/train_config.yaml → ingestion`)

| Parameter | Value | Location |
|---|---|---|
| `interval_sec` | 2 s | `ingestion.collect.interval_sec` |
| `max_frames_per_video` | 500 | `ingestion.collect.max_frames_per_video` |
| `blur_threshold` | 100 | `ingestion.filter.blur_threshold` |
| `brightness_min/max` | 30 / 225 | `ingestion.filter.brightness_*` |
| `min_width/height` | 640 / 360 | `ingestion.filter.min_*` |
| `val_ratio` | 0.20 | `ingestion.merge.val_ratio` |
| `infer_conf_threshold` | 0.25 | `DEFAULT_INFER_CONF_THRESHOLD` in pseudo_label.py |
| `threshold` (mean conf gate) | 0.50 | `DEFAULT_THRESHOLD` in pseudo_label.py |

---

## 3. Training Pipeline — Stage by Stage

```mermaid
flowchart TD
    ENV{"Run mode?"}
    ENV -->|"DEBUG_MODE=true"| DBG["generate dummy data\n(no DVC, 1 epoch)"]
    ENV -->|"USE_SUBSET=true"| SUB["dvc pull subset\n(data/coco_person_subset)"]
    ENV -->|"normal"| FULL["dvc pull full dataset\n(data/coco_person)"]

    DBG --> S1OUT
    SUB --> S1OUT
    FULL --> S1OUT

    S1OUT["validate_dataset()\nimage/label count, YOLO format check\n+ dataset_hash.txt → artifacts/"]

    S1OUT --> S2["Stage 2 — Train\nYOLO(model.name).train()\ndevice: cuda / mps / cpu"]
    S2 -->|"results.results_dict"| METRICS["log_params + log_metrics\n→ MLflow run"]
    S2 -->|"best.pt path"| REF["artifacts/checkpoint_path.txt"]

    REF --> S3["Stage 3 — Export"]
    S3 --> FP32["ONNX FP32\nartifacts/model.onnx"]
    S3 --> INT8["ONNX INT8 (dynamic quant)\nartifacts/model_int8.onnx"]
    S3 --> TFL["TFLite INT8\nartifacts/model_int8.tflite\n(skipped if no TensorFlow)"]

    FP32 --> S4["Stage 4 — Benchmark\n100 val images"]
    INT8 --> S4
    S4 --> LAT["latency: mean + p95 ms"]
    S4 --> MAP["mAP50 via YOLO.val()"]
    S4 --> RPT["artifacts/benchmark_report.json"]

    RPT --> GATE{"metric gate\nINT8/FP32 mAP50 ≥ 0.97?"}
    GATE -->|fail| ABORT["abort — no registration"]
    GATE -->|pass| S5["Stage 5 — Register\n3 models → MLflow Registry\nstage=Staging"]

    S5 --> TAGS["tags: git_commit,\ndataset_hash,\nper_class_ap50"]

    TAGS --> RGRD{"regression guard\nstaging mAP50 ≥\nprod mAP50 × (1−0.01)?"}
    RGRD -->|fail| SKIP["skip promotion\nTelegram alert"]
    RGRD -->|pass| PROMOTE["promote → Production\narcive existing versions"]

    PROMOTE --> HEALTH{"SERVING_URL set?"}
    HEALTH -->|no| DONE["done ✓"]
    HEALTH -->|yes| POLL["poll /health every 10s\nfor VERIFY_SECS (60s)"]
    POLL --> ERRRATE{"error rate > 5%?"}
    ERRRATE -->|no| DONE
    ERRRATE -->|yes| ROLLBACK["rollback prev Production\nTelegram alert"]

    style GATE fill:#e94560,color:#fff
    style RGRD fill:#e94560,color:#fff
    style ERRRATE fill:#e94560,color:#fff
    style ABORT fill:#333,color:#aaa
    style SKIP fill:#333,color:#aaa
    style ROLLBACK fill:#333,color:#aaa
    style DONE fill:#0f3460,color:#eee
```

### Artifact files produced

| File | Stage | Description |
|---|---|---|
| `artifacts/checkpoint_path.txt` | S2 | Path to `best.pt` used by S3 |
| `artifacts/model.onnx` | S3 | FP32 export |
| `artifacts/model_int8.onnx` | S3 | Dynamic INT8 quantized |
| `artifacts/model_int8.tflite` | S3 | TFLite INT8 (GPU machine only) |
| `artifacts/benchmark_report.json` | S4 | Latency + mAP50 per format |
| `artifacts/dataset_hash.txt` | S1 | MD5 of all label files |

---

## 4. Config Flow — Single Source of Truth

```mermaid
flowchart TD
    YAML["configs/train_config.yaml"]
    PYDANTIC["core/config.py\nConfig (Pydantic BaseModel)"]
    YAML --> PYDANTIC

    PYDANTIC --> DC["DataConfig\n.dir, .train_dir, .test_dir\n.active_root(subset)\n.train_images_val, ..."]
    PYDANTIC --> AC["ArtifactsConfig\n.onnx_fp32_path\n.checkpoint_ref_path\n.benchmark_report_path\n..."]
    PYDANTIC --> MLC["MLflowConfig\n.experiment\n.yolo_map50_key\n.map_metric"]
    PYDANTIC --> QGC["QualityGateConfig\n.int8_degradation_tolerance (0.97)\n.promotion_regression_tolerance (0.01)\n.health_error_threshold (0.05)\n.sensor_interval_sec (300)"]
    PYDANTIC --> ING["IngestionConfig\n.collect / .filter / .merge"]
    PYDANTIC --> TC["TrainConfig\n.epochs, .imgsz, .batch\n.lr0, .optimizer, .device"]

    CONST["app/constant/__init__.py"]
    CONST --> MN["MODEL_NAME_FP32/INT8/TFLITE"]
    CONST --> INF["DEFAULT_CONF_THRESHOLD (0.25)\nDEFAULT_IMGSZ (640)\nDEFAULT_IOU_THRESHOLD (0.45)"]
    CONST --> CLS["PERSON_CLASS_ID = 0\nCLASS_NAMES = {0: 'person'}"]

    DC --> S1["stage1_data.py"]
    AC --> S2["stage2_train.py"]
    AC --> S3["stage3_export.py"]
    AC --> S4["stage4_benchmark.py"]
    MLC --> S4
    AC --> S5["stage5_register.py"]
    QGC --> S5
    QGC --> SP["stage_promote.py"]
    MN --> S5
    MN --> SP
    ING --> SCR["scripts/collect_frames.py\nscripts/filter_frames.py\nscripts/merge_dataset.py"]

    SERVE_YAML["configs/serve_config.yaml"]
    CM["app/config/configuration.py\nConfigurationManager"]
    SERVE_YAML --> CM
    CONST -->|"fallback defaults"| CM
    CM --> APP["app/ (FastAPI serving)"]

    style YAML fill:#e94560,color:#fff
    style CONST fill:#e94560,color:#fff
```

---

## 5. Dagster Orchestration

```mermaid
flowchart TD
    subgraph TRIGGERS["Triggers"]
        SENS["dataset_change_sensor\npoll artifacts/dataset_hash.txt\nevery 300s"]
        SCHED["weekly_retrain\ncron: 0 19 * * 0\n= Senin 02:00 WIB"]
        MANUAL["manual / Dagster UI"]
    end

    subgraph ASSETS["Software-Defined Assets (orchestration/assets/pipeline_assets.py)"]
        direction TB
        A1["dataset\nDVC pull + validate + hash"]
        A2["trained_model\ntrain YOLOv8n → checkpoint"]
        A3["exported_model\nONNX FP32 + INT8 + TFLite"]
        A4["benchmark_report\nlatency + mAP50"]
        A5["registered_model\nMLflow Registry → Staging"]
        A6["promoted_model\nStaging → Production\n+ health check"]
        A1 --> A2 --> A3 --> A4 --> A5 --> A6
    end

    subgraph JOBS["Jobs"]
        STAGING["rescuevision_staging\ndataset→registered_model\n(no promote)"]
        PRODUCTION["rescuevision_production\nall assets including promote"]
    end

    SENS -->|"hash changed"| STAGING
    SCHED --> STAGING
    MANUAL --> PRODUCTION

    style TRIGGERS fill:#1a1a2e,stroke:#e94560,color:#eee
    style ASSETS fill:#16213e,stroke:#0f3460,color:#eee
    style JOBS fill:#0f3460,stroke:#533483,color:#eee
```

### Asset metadata emitted

Each asset returns `MaterializeResult` with metadata visible in the Dagster UI:

| Asset | Key metadata |
|---|---|
| `dataset` | `train_count`, `val_count`, `dataset_hash` |
| `trained_model` | `checkpoint_path`, `device`, `epochs`, `imgsz` |
| `exported_model` | `onnx_fp32` MB, `onnx_int8` MB, `tflite_int8` MB |
| `benchmark_report` | `fp32_mAP50`, `int8_mAP50`, `fp32_mean_latency_ms` |
| `registered_model` | `status`, `models` list |
| `promoted_model` | `promoted_models`, `serving_verified` |

---

## 6. CI/CD Pipeline (GitHub Actions)

```mermaid
flowchart TD
    PUSH["git push (any branch)\nor PR to main\nor workflow_dispatch"]

    subgraph JOB1["Job 1: quality (all branches)"]
        RUFF["ruff check + format"]
        MYPY["mypy app/ core/ pipeline/\n(continue-on-error)"]
    end

    subgraph JOB2["Job 2: test (all branches)"]
        PYTEST["pytest tests/serve/\ntests/test_metric_gate.py\ntests/test_label_validation.py\ncoverage ≥ 70%"]
    end

    subgraph JOB3["Job 3: pipeline-staging (main only)"]
        direction TB
        CI1["Stage 1 — dummy data\n(DEBUG_MODE=true)"]
        CI2["Stage 2 — train 1 epoch CPU"]
        CI3["Stage 3 — export"]
        CI4["Stage 4 — benchmark"]
        CI5["Stage 5 — register --stage Staging"]
        NOTIF["Telegram: 🟡 awaiting approval\nlink to GH Actions run"]
        CI1 --> CI2 --> CI3 --> CI4 --> CI5 --> NOTIF
    end

    subgraph JOB4["Job 4: promote-production\n(environment: production = manual approval)"]
        PROMOTE["stage_promote.py\nregression guard → Production\nTelegram: ✅ or ⚠️ rollback"]
    end

    PUSH --> JOB1 --> JOB2
    JOB2 -->|"main branch only"| JOB3
    JOB3 -->|"success + human approves"| JOB4
    PUSH -->|"workflow_dispatch\npromote_to_production=true"| JOB4

    style JOB3 fill:#16213e,stroke:#0f3460,color:#eee
    style JOB4 fill:#e94560,stroke:#533483,color:#fff
```

---

## 7. Serving — FastAPI Inference Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant ModelLoader
    participant MLflow Registry
    participant Preprocessor
    participant ONNXRuntime
    participant Postprocessor

    Note over FastAPI,MLflow Registry: Startup
    FastAPI->>ModelLoader: load model on startup
    ModelLoader->>MLflow Registry: fetch latest Production version\n(rescuevision-onnx-int8)
    MLflow Registry-->>ModelLoader: model artifact
    ModelLoader-->>FastAPI: InferenceSession cached

    Note over Client,Postprocessor: Inference request
    Client->>FastAPI: POST /predict\nmultipart image (JPEG/PNG/WEBP ≤ 10MB)
    FastAPI->>Preprocessor: decode + letterbox resize to 640×640\nnormalize to [0,1] NCHW float32
    Preprocessor-->>FastAPI: blob tensor
    FastAPI->>ONNXRuntime: session.run(blob)
    ONNXRuntime-->>FastAPI: raw output (1, 5, N_anchors)
    FastAPI->>Postprocessor: conf filter (≥0.25)\nNMS (IoU ≤0.45)\nscale back to original coords
    Postprocessor-->>FastAPI: list of BoundingBox
    FastAPI-->>Client: JSON {detections: [{class, conf, bbox}]}

    Note over FastAPI: GET /health
    Client->>FastAPI: GET /health
    FastAPI->>FastAPI: check model loaded\ncheck error rate ≤ 5%
    FastAPI-->>Client: {status: ok, model_version, uptime_s}
```

---

## 8. Environment Variables Reference

| Variable | Used by | Required |
|---|---|---|
| `DAGSHUB_USERNAME` | stage1, stage_promote, mlflow_client | yes (prod) |
| `DAGSHUB_REPO` | stage1, stage_promote, mlflow_client | yes (prod) |
| `DAGSHUB_TOKEN` | stage1, stage_promote, mlflow_client | yes (prod) |
| `DEBUG_MODE` | all stages | no (default: false) |
| `USE_SUBSET` | all stages | no (default: false) |
| `TRAIN_DEVICE` | stage2 | no (override train.device from yaml) |
| `SERVING_URL` | stage_promote | no (skips health check if unset) |
| `VERIFY_SECS` | stage_promote | no (default: 60) |
| `TELEGRAM_TOKEN` | app/utils/telegram.py | no (silent if unset) |
| `TELEGRAM_CHAT_ID` | app/utils/telegram.py | no |

---

## 9. Local Development Cheatsheet

```bash
# Run full pipeline locally (subset + debug)
USE_SUBSET=true DEBUG_MODE=true make all

# Individual stages
uv run python -m pipeline.stage1_data
uv run python -m pipeline.stage2_train
uv run python -m pipeline.stage3_export
uv run python -m pipeline.stage4_benchmark
uv run python -m pipeline.stage5_register --stage Staging
uv run python -m pipeline.stage_promote

# Start Dagster UI
make dagster-dev   # http://localhost:3000

# Serve the API
make serve         # http://localhost:8080

# Data ingestion
python scripts/collect_frames.py --sources data/sources.txt
python scripts/filter_frames.py --input data/raw_frames
python scripts/pseudo_label.py --input data/filtered_frames
python scripts/merge_dataset.py

# Quality
make quality       # ruff + mypy
make test          # pytest
```
