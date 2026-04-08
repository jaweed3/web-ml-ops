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

Dependencies dibagi dua tergantung mesin yang dipakai.

### Base install — semua mesin

Tidak ada torch, tidak ada ultralytics. Cocok untuk:
- Laptop i3 (debug, serve tests, code editing)
- Mac Mini M4 (debug serving layer, jalankan test_export / test_benchmark kalau artifacts sudah ada)

```bash
git clone https://github.com/your-username/rescuevision-mlops
cd rescuevision-mlops
make install   # pip install -r requirements.txt
```

### Training install — GPU machine only

Tambahan: ultralytics (narik torch ~2GB) + tensorflow (TFLite export).  
Jalankan ini hanya di RTX 4060 lab atau Mac Mini M4 kalau mau train lokal.

```bash
make install-train   # pip install -r requirements.txt -r requirements-train.txt
```

### Code quality hooks

```bash
pip install pre-commit
make pre-commit-install
```

---

## Device setup per mesin

| Mesin | Install | `device` di train_config.yaml |
|---|---|---|
| Laptop i3 gen 7 | base only | — (tidak perlu train) |
| Mac Mini M4 | base atau install-train | `mps` |
| PC lab RTX 4060 | install-train | `cuda` |

Edit `configs/train_config.yaml`:
```yaml
train:
  device: cuda   # ganti ke mps untuk Mac Mini M4
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
