# Setup

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Use pyenv or conda |
| uv | latest | Package manager — `pip install uv` |
| Git | any | — |
| DagsHub account | — | Free — used for MLflow tracking + DVC remote |
| Docker + Docker Compose | any | Only needed for the serving stack |

---

## Install

Dependencies are split by machine type. All installs use `uv`.

### Dev install — all machines

No torch, no ultralytics. Covers: debug mode, serving tests, code editing, running export/benchmark if artifacts already exist.

```bash
make install-dev
# equivalent: uv sync --only-group dev
```

### Training install — GPU machine only

Adds ultralytics (~2 GB torch) and tensorflow (TFLite export). Run this only on RTX 4060 lab or Mac Mini M4 if you want to train locally.

```bash
make install-train
# equivalent: uv sync --all-groups
```

### Debug install — CPU dev with torch

Adds CPU-only torch, matplotlib, and ipykernel. Useful for local logic testing and notebooks.

```bash
make install-debug
# equivalent: uv sync --group dev --group debug
```

### Code quality hooks

```bash
pip install pre-commit
make pre-commit-install
```

---

## Device setup per machine

| Machine | Install | `device` in train_config.yaml |
|---|---|---|
| Laptop i3 gen 7 | `install-dev` | — (no training) |
| Mac Mini M4 | `install-dev` or `install-train` | `mps` |
| PC lab RTX 4060 | `install-train` | `cuda` |

Edit `configs/train_config.yaml`:
```yaml
train:
  device: cuda   # or mps for Mac Mini M4, cpu as fallback
```

---

## Environment variables

Create a `.env` file in the repo root:

```bash
# .env
DAGSHUB_USERNAME=your-dagshub-username
DAGSHUB_REPO=rescuevision-mlops
DAGSHUB_TOKEN=your-dagshub-access-token

# Serving (Phase 2)
MODEL_NAME=rescuevision-onnx-int8
MODEL_VERSION=latest
MODEL_FORMAT=onnx_int8

CONF_THRESHOLD=0.25
IOU_THRESHOLD=0.45
```

Load before running any stage locally:

```bash
export $(cat .env | xargs)
```

> Never commit `.env`. It is already in `.gitignore`.

---

## Debug mode

Set `DEBUG_MODE=true` to skip DVC pull and generate a small synthetic dataset instead. Stages run with reduced epochs/imgsz/batch so the full flow can be validated on any machine without GPU or DagsHub credentials.

```bash
DEBUG_MODE=true uv run python -m pipeline.stage1_data
DEBUG_MODE=true uv run python -m pipeline.stage2_train
DEBUG_MODE=true uv run python -m pipeline.stage3_export

# Or run all three in one shot:
make debug
```

Debug overrides are set in `configs/train_config.yaml` under the `debug:` key:

```yaml
debug:
  epochs: 1
  imgsz: 320
  batch: 4
  device: cpu
  max_samples: 50
```

---

## DVC remote setup

Run once after cloning. Authenticates the DVC S3-compatible remote on DagsHub:

```bash
dvc remote add origin s3://your-username/rescuevision-mlops
dvc remote modify origin endpointurl https://dagshub.com/your-username/rescuevision-mlops.s3
dvc remote modify origin --local access_key_id $DAGSHUB_TOKEN
dvc remote modify origin --local secret_access_key $DAGSHUB_TOKEN
dvc remote default origin
```

The `--local` flag writes credentials to `.dvc/config.local` (gitignored). The base `.dvc/config` file is committed and contains no secrets.

---

## Verify setup

```bash
# Check DVC can see the remote
dvc remote list

# Check MLflow connection (requires DAGSHUB_* env vars)
python -c "
import os, mlflow
uri = f\"https://dagshub.com/{os.environ['DAGSHUB_USERNAME']}/{os.environ['DAGSHUB_REPO']}.mlflow\"
mlflow.set_tracking_uri(uri)
print('MLflow URI:', mlflow.get_tracking_uri())
"
```
