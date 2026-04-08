# Setup

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Use pyenv or conda |
| Git | any | — |
| DagsHub account | — | Free — used for MLflow tracking + DVC remote |
| Docker + Docker Compose | any | Only needed for serving stack |

---

## Install

```bash
git clone https://github.com/your-username/rescuevision-mlops
cd rescuevision-mlops
pip install -r requirements.txt
```

To also enable code quality hooks:

```bash
pip install pre-commit
make pre-commit-install
```

---

## Environment variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

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
