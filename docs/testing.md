# Testing Guide

Panduan testing dari awal sampai akhir — unit test, pipeline test, API test, sampai smoke test full stack.

---

## Overview

```
Level 1 — Unit tests (no external deps)
  tests/serve/test_preprocessor.py
  tests/serve/test_postprocessor.py
  tests/serve/test_prediction_logger.py
  tests/test_label_validation.py
  tests/test_metric_gate.py

Level 2 — API tests (mocked runner, no model needed)
  tests/serve/test_health.py
  tests/serve/test_predict.py
  tests/serve/test_security.py        ← auth, rate limit, content-length

Level 3 — Pipeline tests (requires dataset + artifacts)
  tests/test_data.py
  tests/test_export.py
  tests/test_benchmark.py

Level 4 — Smoke test (requires running server)
  scripts/smoke_test.sh

Level 5 — Load test (requires running server)
  scripts/locustfile.py
```

---

## Prerequisites

```bash
make install-dev   # base deps — cukup untuk Level 1, 2, dan sebagian Level 3
```

Level 3 tests yang butuh ultralytics/tensorflow di-**skip otomatis** kalau package tidak terinstall — tidak error, hanya `SKIPPED`.

| Level | Deps | Laptop i3 | Mac Mini M4 | RTX 4060 lab |
|---|---|---|---|---|
| 1 — Unit | `install-dev` | ✅ | ✅ | ✅ |
| 2 — API | `install-dev` | ✅ | ✅ | ✅ |
| 3 — Pipeline (data, export, benchmark, register) | `install-dev` + artifacts | ⚠️ bisa kalau artifacts dicopy | ✅ | ✅ |
| 3 — Pipeline (train) | `install-train` | ❌ | ✅ (MPS) | ✅ (CUDA) |
| 4 — Smoke | `install-dev` + server running | ✅ | ✅ | ✅ |

---

## Level 1 — Unit tests

Test komponen preprocessing dan postprocessing secara isolated. Tidak butuh ONNX model, tidak butuh network.

### Preprocessor

```bash
pytest tests/serve/test_preprocessor.py -v
```

**Yang di-test:**
- Output shape selalu `(1, 3, 640, 640)` untuk semua rasio gambar
- Landscape (480×640), portrait (1080×720), square (640×640)
- Output dtype adalah `float32`
- Nilai pixel ternormalisasi antara 0.0 dan 1.0
- Image corrupt atau bytes kosong raise `ValueError`
- Custom `imgsz=320` menghasilkan shape `(1, 3, 320, 320)`

**Expected output:**
```
tests/serve/test_preprocessor.py::test_output_shape_square_input PASSED
tests/serve/test_preprocessor.py::test_output_shape_landscape_input PASSED
tests/serve/test_preprocessor.py::test_output_shape_portrait_input PASSED
tests/serve/test_preprocessor.py::test_output_dtype_is_float32 PASSED
tests/serve/test_preprocessor.py::test_values_normalized_between_0_and_1 PASSED
tests/serve/test_preprocessor.py::test_corrupt_bytes_raises_value_error PASSED
tests/serve/test_preprocessor.py::test_empty_bytes_raises_value_error PASSED
tests/serve/test_preprocessor.py::test_custom_imgsz_320 PASSED
```

---

### Postprocessor

```bash
pytest tests/serve/test_postprocessor.py -v
```

**Yang di-test:**
- Deteksi di-return kalau confidence di atas threshold
- Tidak ada deteksi kalau semua confidence di bawah threshold
- Output kosong kalau tidak ada box sama sekali
- Schema tiap deteksi: `class_id`, `class_name`, `confidence`, `bbox`
- Bbox punya semua field: `x1`, `y1`, `x2`, `y2`, `cx`, `cy`, `width`, `height`
- `class_name` selalu `"person"`, `class_id` selalu `0`
- Confidence di-round 4 desimal
- `max_detections` cap bekerja

**Expected output:**
```
tests/serve/test_postprocessor.py::test_detections_returned_above_threshold PASSED
tests/serve/test_postprocessor.py::test_no_detections_below_threshold PASSED
tests/serve/test_postprocessor.py::test_empty_output_returns_empty_list PASSED
tests/serve/test_postprocessor.py::test_detection_has_required_keys PASSED
tests/serve/test_postprocessor.py::test_bbox_has_required_fields PASSED
tests/serve/test_postprocessor.py::test_class_name_is_person PASSED
tests/serve/test_postprocessor.py::test_class_id_is_zero PASSED
tests/serve/test_postprocessor.py::test_confidence_rounded_to_4_decimal_places PASSED
tests/serve/test_postprocessor.py::test_max_detections_cap PASSED
```

---

## Level 2 — API tests

FastAPI TestClient dengan mocked runner. Tidak ada model yang di-load, tidak ada request ke DagsHub.

### Health endpoints

```bash
pytest tests/serve/test_health.py -v
```

**Yang di-test:**
- `GET /health` returns 200
- Response body punya `status: "ok"` dan `timestamp`
- `GET /ready` returns 200 kalau mock runner `is_ready = True`
- Response body punya `status: "ready"`, `model_version`, `loaded_at`

---

### Predict endpoint

```bash
pytest tests/serve/test_predict.py -v
```

**Yang di-test:**

| Test | Input | Expected |
|---|---|---|
| Valid JPEG | 480×640 JPEG | 200 |
| Response keys | — | `detections`, `inference_time_ms`, `model_version`, `request_id`, `image_shape` |
| Wrong MIME | `text/plain` | 422 |
| Corrupt JPEG | random bytes | 422 |
| Empty file | 0 bytes | 422 |
| X-Request-ID header | — | header ada di response |
| `detections` type | — | list |
| `image_shape` | — | `[640, 640]` |

---

### Security — auth, rate limit, content-length

```bash
pytest tests/serve/test_security.py -v
```

**Yang di-test:**

| Test | Skenario | Expected |
|------|----------|----------|
| `test_predict_passes_when_api_key_disabled` | `API_KEY` env tidak di-set | 200 |
| `test_predict_401_when_api_key_required_and_missing` | `API_KEY` di-set, header tidak ada | 401 |
| `test_predict_401_when_api_key_wrong` | Key salah | 401 |
| `test_predict_200_when_api_key_correct` | Key benar | 200 |
| `test_predict_422_when_content_length_exceeds_limit` | `Content-Length: 11MB` | 422 |
| `test_predict_passes_when_content_length_within_limit` | `Content-Length` valid | 200 |
| `test_predict_429_when_rate_limit_exceeded` | 11 request dalam 1 menit | 429 pada request ke-11 |

---

### Prediction logger

```bash
pytest tests/serve/test_prediction_logger.py -v
```

**Yang di-test:**
- File `predictions.jsonl` ter-create setelah inference pertama
- Setiap inference menghasilkan satu baris JSON baru
- Record punya field: `ts`, `request_id`, `model_version`, `n_detections`, `inference_time_ms`
- Timestamp dalam format UTC ISO 8601
- Multiple records ter-append dengan benar (tidak overwrite)
- Error I/O (misal path read-only) tidak crash server — silently ignored

---

### Label schema validation

```bash
pytest tests/test_label_validation.py -v
```

**Yang di-test:**

| Test | Input | Expected |
|------|-------|----------|
| Valid single annotation | `0 0.5 0.5 0.2 0.3` | Pass |
| Valid multiple annotations | 2 baris valid | Pass |
| Empty file | kosong | Pass (background image) |
| Boundary values | `0 0.0 0.0 1.0 1.0` | Pass |
| Wrong field count | 4 field | `ValueError: expected 5 fields` |
| Non-numeric value | `0 abc 0.5 0.2 0.3` | `ValueError: non-numeric` |
| Negative class_id | `-1 0.5 ...` | `ValueError: class_id must be ≥ 0` |
| Coord above 1 | `0 1.1 ...` | `ValueError: out of range` |
| Coord below 0 | `0 -0.1 ...` | `ValueError: out of range` |
| Invalid second line | baris pertama ok, kedua tidak | `ValueError` pada baris ke-2 |

---

### Metric gate

```bash
pytest tests/test_metric_gate.py -v
```

11 test pure logic — tidak butuh MLflow, tidak butuh file apapun.

---

### Run semua serve tests sekaligus

```bash
make test-serve
# atau
pytest tests/serve/ tests/test_metric_gate.py tests/test_label_validation.py -v \
  --cov=app --cov-report=term-missing
```

**Expected summary:**
```
=================== 64 passed in X.XXs ===================
```

---

## Level 3 — Pipeline tests

Butuh dataset dan artifacts tersedia. Jalankan pipeline stages dulu kalau belum.

### Setup

```bash
# Load credentials
export $(cat .env | xargs)

# Pull dataset
dvc pull

# Jalankan full pipeline
dvc repro
```

---

### test_data.py — Dataset integrity

```bash
pytest tests/test_data.py -v
```

**Yang di-test:**
- Required dirs ada (`images/train`, `images/val`, `labels/train`, `labels/val`)
- Training dan val set tidak kosong
- Jumlah image = jumlah label (tidak ada orphaned annotation)
- Setiap image punya label yang matching (berdasarkan stem)
- Format YOLO valid: 5 field, semua float, coords dalam `[0, 1]`
- Semua annotation class id = 0 (person only)
- Spot-check 20 random image untuk korupsi via PIL

Kalau fail setelah `dvc pull`: cek `DAGSHUB_TOKEN`, cek DVC remote setup.

---

### test_train.py — Training output

```bash
pytest tests/test_train.py -v
```

**Yang di-test:**
- `runs/train/weights/best.pt` ada dan > 5MB
- Checkpoint bisa di-load Ultralytics tanpa error
- MLflow experiment `rescuevision-yolov8n` ada dan punya run
- mAP50 di atas minimum threshold (0.30)
- Semua hyperparams ter-log (`model`, `epochs`, `batch`)
- `best.pt` artifact ter-log di MLflow run

> Test MLflow di-skip otomatis kalau `DAGSHUB_*` env vars tidak di-set.

---

### test_export.py — Artifacts

```bash
pytest tests/test_export.py -v
```

**Yang di-test:**
- Ketiga artifact ada dan > 0.5MB
- ONNX FP32 dan INT8 bisa di-load ORT
- TFLite INT8 bisa di-load `tf.lite.Interpreter`
- Input shape ONNX = 4 dimensi, 3 channel
- Output shape FP32 = `(1, 5, 8400)` — YOLOv8n expected
- Output shape INT8 = sama dengan FP32
- Output FP32 dan INT8 tidak mengandung NaN/Inf
- INT8 lebih kecil dari FP32
- Compression ratio ≥ 1.5x (kurang dari itu berarti quantization gagal)

---

### test_benchmark.py — Latency SLOs

```bash
pytest tests/test_benchmark.py -v
```

**Yang di-test:**
- Report schema valid (semua required fields ada)
- Kedua format ONNX ada di report
- INT8 lebih kecil dari FP32
- FP32 model < 15MB, INT8 < 10MB (sanity cap)
- FP32 mean latency < 200ms
- INT8 mean latency < 100ms
- INT8 p95 latency < 150ms
- INT8 lebih cepat dari FP32
- Semua latency positif
- p95 ≥ mean (secara statistik harus selalu true)

> SLO threshold bisa disesuaikan di bagian atas `test_benchmark.py` sesuai target hardware.

---

### test_register.py — MLflow Model Registry

```bash
pytest tests/test_register.py -v
```

**Yang di-test:**
- Ketiga model terdaftar di registry (`rescuevision-onnx-fp32`, `rescuevision-onnx-int8`, `rescuevision-tflite-int8`)
- Ada versi di stage Staging atau None
- Setiap versi punya tag `git_commit` (untuk traceability)
- Setiap versi punya tag `pipeline`
- Setiap versi ter-link ke training run (`run_id` tidak kosong)

> Test ini di-skip otomatis kalau `DAGSHUB_*` env vars tidak di-set.

---

### Run semua pipeline tests

```bash
make test-pipeline
# atau
pytest tests/ --ignore=tests/serve -v
```

---

## Level 4 — Smoke test + manual checks

End-to-end test terhadap server yang beneran running. Menggunakan `curl`.

### Start the server

```bash
# Option A: local
export $(cat .env | xargs)
make run

# Option B: Docker (recommended — lebih mirip production)
make up

# Tunggu sampai server ready
curl http://localhost:8080/ready
# {"status":"ready","model_version":"3","loaded_at":"..."}
```

### Run smoke test

```bash
make smoke
# atau
bash scripts/smoke_test.sh
```

**Yang di-cek:**
```
PASS  /health → status ok
PASS  /ready → status ready
PASS  /model/info → has name
PASS  /model/info → has version
PASS  /model/version → has version
PASS  /predict → detections key
PASS  /predict → request_id key
PASS  /predict → inference_time_ms
PASS  /predict → 422 for text/plain

Results: 9 passed, 0 failed
```

---

### Manual: Rate limiting

Verifikasi 429 langsung via curl (butuh server running):

```bash
# Kirim 11 request cepat — request ke-11 harus 429
for i in $(seq 1 11); do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -F "file=@test_data/sample.jpg;type=image/jpeg" \
    http://localhost:8080/predict)
  echo "Request $i: $STATUS"
done
```

Expected output:
```
Request 1: 200
...
Request 10: 200
Request 11: 429
```

---

### Manual: Prediction logging

Setelah beberapa request ke `/predict`, cek file log:

```bash
cat artifacts/predictions.jsonl
```

Expected (satu record per baris):
```json
{"ts": "2026-04-22T10:00:00.123456+00:00", "request_id": "a1b2c3d4", "model_version": "3", "n_detections": 2, "inference_time_ms": 9.4}
{"ts": "2026-04-22T10:00:01.456789+00:00", "request_id": "b2c3d4e5", "model_version": "3", "n_detections": 0, "inference_time_ms": 8.1}
```

---

### Manual: Model warm-up

Cek log saat server start — harus ada `warmup_complete` sebelum `server_ready`:

```bash
make up && docker compose logs rescuevision-serve | grep -E "warmup|server_ready"
```

Expected:
```
INFO  warmup_complete  warmup_ms=12.3
INFO  server_ready     version=3
```

---

### Manual: Telegram notification

Test kirim notifikasi manual:

```bash
export TELEGRAM_TOKEN="your_token"
export TELEGRAM_CHAT_ID="8126957752"

python - <<'EOF'
from app.utils.telegram import notify
notify("🧪 *Test notifikasi RescueVision*\nKalau pesan ini masuk, Telegram integration berjalan.")
EOF
```

Cek HP — pesan harus muncul di bot chat dalam beberapa detik.

---

## Level 5 — Load test

Butuh server running. Install locust jika belum:

```bash
uv add locust --group dev
```

### Run load test

```bash
# Server harus running dulu
make run   # atau: make up

# Jalankan load test (50 users, 60 detik)
make load-test
```

Output contoh:
```
 Name            # reqs  # fails  Avg  Min  Max  Med   req/s failures/s
 POST /predict   2841    0(0.00%) 47   12   198  42    47.4  0.0

 Percentiles:
 50%   42ms
 75%   68ms
 95%   142ms   ← harus < 200ms
 99%   187ms

 Total: 2841 requests, 0 failures
```

**Pass criteria:**
- p95 latency < 200 ms
- Error rate < 1%
- Tidak ada 5xx (429 dihitung sebagai expected, bukan error)

### Run dengan auth

```bash
API_KEY=your_key make load-test
```

---

## Jalankan semuanya sekaligus

```bash
# Level 1 + 2 (tidak butuh model — cukup install-dev)
make test-serve

# Level 3 (butuh dataset + artifacts — jalankan dvc repro dulu)
make test-pipeline

# Semua unit + API tests
make test

# Level 4 — smoke test (butuh server running)
make up && sleep 30 && make smoke

# Level 5 — load test (butuh server running)
make load-test
```

### Urutan lengkap dari nol

```bash
# 1. Install deps
make install-dev

# 2. Unit + API tests (tidak butuh model)
pytest tests/serve/ tests/test_metric_gate.py tests/test_label_validation.py -v

# 3. Start server (Docker)
make up
sleep 30  # tunggu warm-up selesai

# 4. Smoke test
make smoke

# 5. Telegram test
python -c "from app.utils.telegram import notify; notify('✅ smoke test passed')"

# 6. Rate limit manual check
for i in $(seq 1 11); do
  echo -n "req $i: "
  curl -s -o /dev/null -w "%{http_code}\n" \
    -F "file=@data/coco_person/images/val/$(ls data/coco_person/images/val | head -1);type=image/jpeg" \
    http://localhost:8080/predict
done

# 7. Check prediction log
cat artifacts/predictions.jsonl | python -m json.tool | head -20

# 8. Load test
make load-test
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'app'`

```bash
# Jalankan pytest dari root direktori repo
cd /path/to/rescuevision-mlops
pytest tests/serve/ -v
```

### `RuntimeError: ONNXRunner not initialized`

Test serve menggunakan mocked runner. Kalau muncul error ini, berarti `conftest.py` tidak di-load dengan benar. Pastikan file `tests/serve/conftest.py` ada.

### `ConnectionRefusedError` saat smoke test

Server belum ready. Cek:
```bash
curl http://localhost:8080/health    # harus 200
curl http://localhost:8080/ready     # harus status=ready (bukan loading)
docker compose logs rescuevision-serve
```

### Test pipeline fail karena dataset tidak ada

```bash
export $(cat .env | xargs)
dvc pull
pytest tests/test_data.py -v
```

### Coverage report

```bash
pytest tests/serve/ --cov=app --cov-report=html
open htmlcov/index.html
```
