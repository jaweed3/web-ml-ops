# Architecture Decisions

Dokumentasi keputusan desain yang non-obvious — kenapa pipeline dibuat seperti ini, bukan yang lain.

---

## Repo structure

### Kenapa `core/` terpisah dari `pipeline/`?

Setiap stage import dari `core/` untuk logging, config, dan MLflow. Tidak ada stage yang import dari stage lain. Artinya:
- Stage bisa dijalankan secara isolated tanpa menjalankan stage sebelumnya
- Tiap stage bisa di-test secara independent
- Satu stage bisa diganti tanpa breaking yang lain

### Kenapa `params.yaml` ada di samping `train_config.yaml`?

`train_config.yaml` adalah config lengkap (semua field). `params.yaml` adalah subset khusus DVC — hanya nilai yang perlu DVC pantau untuk mendeteksi kapan stage harus dijalankan ulang. Mengubah nilai di `params.yaml` akan trigger downstream stages otomatis saat `dvc repro`.

### Kenapa `app/` pakai layered structure?

Layered dependency injection — layer atas gak boleh import layer bawah secara langsung:

```
constant   (tidak import apapun dari app)
  ↓
config     (import constant)
  ↓
components (import config, constant, core)
  ↓
pipeline   (import components)
  ↓
router     (import pipeline, schema)
```

`entity/` dihapus di refactor — semua config model unified ke Pydantic di `core/config.py`. Tidak perlu layer terpisah untuk config entity.

---

## Serving layer

### Kenapa FastAPI bukan Flask?

- Async request handling — Flask synchronous, blocking di concurrent inference requests
- Native Pydantic integration — response schemas otomatis jadi OpenAPI contract
- Automatic `/docs` dan `/redoc` tanpa tambahan kode
- `lifespan` context manager untuk startup/shutdown yang clean

### Kenapa model di-pull dari MLflow registry, bukan di-bake ke Docker image?

Model artifact berubah lebih sering dari serving code. Kalau model di-bake ke image:
- Setiap retrain = full image rebuild + redeploy
- Image bisa 500MB+ hanya karena model weights

Dengan registry-based loading:
- Image yang sama bisa serve model versi berapapun
- Switch versi = ubah satu env var + restart container
- Rollback = transisi versi di MLflow, tidak perlu rebuild

### Kenapa `ONNXRunner` singleton, bukan dibuat per request?

ONNX Runtime session initialization mahal — bisa 2-5 detik untuk YOLOv8n. Kalau dibuat per request, throughput turun drastis. Singleton dibuat sekali saat startup, disimpan di memori, dan di-lock per inference dengan `threading.Lock()` untuk thread safety.

### Kenapa `workers: 1` di uvicorn?

ONNX Runtime InferenceSession tidak fork-safe. Kalau workers > 1, tiap worker butuh session sendiri. Untuk sekarang dibiarkan 1 worker + async I/O untuk concurrent request handling. Kalau butuh scale, gunakan multiple container instances di belakang load balancer — bukan multiple workers dalam satu container.

### Kenapa config entity di-unify ke Pydantic?

Dulu ada tiga approach parallel: frozen dataclass di `entity/`, raw dict dari yaml, dan Pydantic model. Ini menyebabkan inconsistency — kadang field name beda, kadang type coercion gak jalan. Semua config sekarang didefinisikan sebagai Pydantic model di `core/config.py`:
- `TrainConfig` — training hyperparams
- `ServeConfig` — serving layer config
- `InferenceConfig` — inference hyperparams
- Nested `CacheConfig`, `CandidateModelConfig`, `ServerConfig`, dll

Env var override built-in via `Field(validation_alias=...)`.

---

## Data & model versioning

### Kenapa DVC untuk dataset, bukan Git LFS?

Dataset coco_person puluhan GB. Git LFS:
- Tidak support partial checkout
- Semua history di-clone, bukan hanya versi yang dibutuhkan
- Mahal untuk storage

DVC:
- Hanya pointer file yang di-commit ke Git
- Dataset di-pull sesuai `dvc.lock` yang aktif
- Remote bisa di-switch (S3, GCS, DagsHub) tanpa ubah pipeline

### Kenapa MLflow di DagsHub, bukan self-hosted?

DagsHub menyediakan MLflow tracking + DVC remote storage dalam satu akun gratis. Untuk proyek ini ukurannya cukup. Self-hosted MLflow butuh infra tambahan (server, storage, auth) yang di luar scope pipeline ini.

---

## Drift detection

### Kenapa drift score dihitung per-request, bukan batch?

Batch processing berarti nunggu akumulasi N request sebelum compute — delay deteksi. Per-request scoring langsung expose `rescuevision_input_drift_score` gauge. Alert bisa fire segera setelah drift terdeteksi.

### Kenapa baseline di-compute offline (`scripts/compute_baseline.py`)?

Baseline computation butuh akses ke training images (via DVC pull). Ini operasi berat yang gak perlu dilakukan di serving path. Baseline disimpan sebagai file JSON di `artifacts/drift_baseline.json` dan di-load DriftDetector saat startup.

### Kenapa Mahalanobis distance, bukan KL divergence?

Tiga statistik (mean brightness, std brightness, entropy) — distribution unknown, bisa non-Gaussian. Mahalanobis menangkap covariance antar statistik tanpa asumsi distribution shape. KL divergence butuh density estimation yang lebih kompleks.

---

## Feedback loop

### Kenapa feedback disimpan ke JSONL, bukan database?

JSONL (JSON Lines) adalah append-only log — tidak perlu schema migration, tidak perlu connection pooling, cocok untuk data yang jarang di-query. Feedback dapat dipairing dengan prediction log via `request_id` untuk offline evaluation.

### `artifacts/predictions.jsonl` + `artifacts/feedback.jsonl`

Prediction log menyimpan model output per request. Feedback log menyimpan ground truth. Join via `request_id` → compute serving mAP, precision-recall.

---

## Shadow deployment

### Kenapa shadow deployment, bukan A/B testing?

Shadow deployment menjalankan candidate model di production tanpa mempengaruhi response user. Output candidate di-log untuk comparison. Ini lebih aman dari A/B testing karena:
- Tidak ada risk candidate model serve hasil yang salah ke user
- Bisa collect data untuk validation sebelum cutover
- Tidak perlu routing infrastructure

### Candidate model config

Candidate model didefinisikan di `configs/serve_config.yaml` di bawah `candidate_model`. Kalau section ini tidak ada atau kosong, shadow runner tidak di-initialize. Tidak ada overhead runtime kalau shadow tidak aktif.

---

## Observability

### Kenapa Prometheus + Grafana, bukan logging saja?

Structured JSON logs bisa di-grep dan di-analyse, tapi tidak bisa di-alert. Prometheus:
- Scrape metrics setiap 15 detik
- Simpan time series — bisa lihat trend latency selama 7 hari
- Grafana bisa set alert kalau p95 latency > threshold

Kedua sistem berjalan paralel: logs untuk debugging per-request, metrics untuk visibility agregat.

### Kenapa `/metrics` tidak muncul di OpenAPI docs?

`/metrics` adalah endpoint internal yang di-scrape Prometheus — bukan bagian dari public API contract. Setting `include_in_schema=False` menjaga docs tetap bersih untuk konsumen API.

---

## CI/CD

### Kenapa ada dua environment (staging vs production)?

Staging dan Production adalah dua state berbeda di MLflow Model Registry:
- **Staging** — model sudah lulus benchmark gate, belum di-validate untuk production traffic
- **Production** — sudah di-review dan di-approve secara manual

Pemisahan ini mencegah model baru langsung ke production hanya karena metric gate lulus. Tim bisa review benchmark report di Staging sebelum promote.

### Kenapa promote-production hanya bisa di-trigger manual?

Karena promosi ke production adalah keputusan yang irreversible (walaupun bisa di-rollback). Otomatis promosi berarti setiap merge ke main bisa langsung mengubah model yang sedang serve request di production — ini terlalu beresiko tanpa human review.

### Kenapa docker-build parallel dengan pipeline-staging?

Kedua job tidak punya dependency satu sama lain. Pipeline-staging menghasilkan model baru; docker-build membangun serving image tanpa model. Dengan fail-fast paralel, keduanya jalan bareng — feedback lebih cepat tanpa menunggu pipeline selesai.
