# Testing Guide

Panduan testing dari awal sampai akhir — unit test, pipeline test, API test, sampai smoke test full stack.

---

## Overview

```
Level 1 — Unit tests (no external deps)
  tests/serve/test_preprocessor.py
  tests/serve/test_postprocessor.py

Level 2 — API tests (mocked runner, no model needed)
  tests/serve/test_health.py
  tests/serve/test_predict.py

Level 3 — Pipeline tests (requires dataset + artifacts)
  tests/test_data.py
  tests/test_export.py
  tests/test_benchmark.py

Level 4 — Smoke test (requires running server)
  scripts/smoke_test.sh
```

---

## Prerequisites

```bash
pip install -r requirements.txt
```

Level 1 dan 2 tidak butuh DagsHub, model weights, atau server yang running.  
Level 3 butuh dataset sudah di-pull dan pipeline sudah dijalankan.  
Level 4 butuh `make up` sudah jalan.

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

### Run semua serve tests sekaligus

```bash
make test-serve
# atau
pytest tests/serve/ -v --cov=app --cov-report=term-missing
```

**Expected summary:**
```
=================== 25 passed in X.XXs ===================
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

# Jalankan pipeline sampai stage yang relevan
make data      # untuk test_data.py
make all       # untuk test_export.py dan test_benchmark.py
```

---

### Dataset integrity test

```bash
pytest tests/test_data.py -v
```

**Yang di-test:**
- `data/coco_person/images/train/` ada
- `data/coco_person/images/val/` ada
- `data/coco_person/labels/train/` ada
- `data/coco_person/labels/val/` ada
- Jumlah image = jumlah label (tidak ada orphaned annotation)
- Training set tidak kosong

Kalau test ini fail setelah `dvc pull`, kemungkinan:
1. Credentials salah → cek `DAGSHUB_TOKEN`
2. DVC remote belum di-setup → lihat [setup.md](setup.md)
3. Dataset belum pernah di-push ke remote

---

### Export artifact test

```bash
pytest tests/test_export.py -v
```

**Yang di-test:**
- `artifacts/model.onnx` ada
- `artifacts/model_int8.onnx` ada
- `artifacts/model_int8.tflite` ada
- ONNX FP32 bisa di-load dengan `onnxruntime`
- ONNX INT8 bisa di-load dengan `onnxruntime`
- TFLite INT8 bisa di-load dengan `tensorflow.lite.Interpreter`

Kalau test ini fail, kemungkinan:
1. `make export` belum dijalankan
2. TensorFlow tidak terinstall → `pip install tensorflow`

---

### Benchmark report test

```bash
pytest tests/test_benchmark.py -v
```

**Yang di-test:**
- `artifacts/benchmark_report.json` ada
- Report punya key `results` dan `n_samples`
- Tiap result punya: `mean_latency_ms`, `model_size_mb`, `format`
- INT8 model lebih kecil dari FP32

Kalau `test_int8_smaller_than_fp32` fail berarti quantization tidak bekerja dengan benar.

---

### Run semua pipeline tests

```bash
make test-pipeline
# atau
pytest tests/ --ignore=tests/serve -v
```

---

## Level 4 — Smoke test

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

## Jalankan semuanya sekaligus

```bash
# Level 1 + 2 (tidak butuh model)
make test-serve

# Level 3 (butuh dataset + artifacts)
make test-pipeline

# Semua (Level 1 + 2 + 3)
make test

# Level 4 (butuh server running)
make up && sleep 30 && make smoke
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
