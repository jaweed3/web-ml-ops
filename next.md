# 🗺️ MLOps Engineering Roadmap & Backlog

Dokumen ini berisi rencana pengembangan teknis untuk mengubah proyek ini dari status *Localhost Prototype* menjadi *Production-Grade System*.

---

## 🛡️ Phase 1: Quality Assurance (The Safety Net)
**Goal:** Mencegah kode sampah atau model buruk masuk ke production secara otomatis.
**Status:** [ ] Not Started

### 1.1 Unit Testing (`pytest`)
- [ ] Buat folder `tests/` di backend.
- [ ] **Data Transformation Test:** Pastikan fungsi cleaning menghapus NULL/Duplikat dengan benar.
- [ ] **Model Config Test:** Pastikan config me-load parameter yang benar dari `params.yaml`.
- [ ] Integrasikan `pytest` ke dalam GitHub Actions (Pipeline harus gagal jika test gagal).

### 1.2 Integration Testing
- [ ] **API Endpoint Test:** Kirim mock JSON ke `/predict` dan pastikan return status 200 + format JSON valid.
- [ ] **Edge Case Test:** Kirim data kosong/invalid ke API, pastikan backend melempar error 400 (bukan Crash 500).

### 1.3 Model Sanity Check
- [ ] Tambahkan logic di `main.py`: Sebelum upload ke DagsHub, cek apakah `accuracy > 0.5` (better than random). Jika tidak, batalkan upload.

---

## 🔭 Phase 2: Observability (The Watchtower)
**Goal:** Memantau kesehatan sistem dan model secara real-time (bukan meraba-raba log terminal).
**Status:** [ ] Not Started

### 2.1 System Monitoring (Prometheus)
- [ ] Tambahkan container `prometheus` di `docker-compose.yaml`.
- [ ] Expose metrics dari Flask backend (gunakan library `prometheus-flask-exporter`).
- [ ] Pantau: Request per second, Latency, CPU/RAM usage container.

### 2.2 Visualization (Grafana)
- [ ] Tambahkan container `grafana` di `docker-compose.yaml`.
- [ ] Hubungkan Grafana ke Prometheus datasource.
- [ ] Buat Dashboard sederhana: "Traffic Real-time" dan "Error Rate".

### 2.3 Data Drift Monitoring (Evidently AI)
- [ ] Setup job mingguan untuk membandingkan statistik data training vs data live (inference).
- [ ] Alert jika distribusi data bergeser (Drift Detected).

---

## ☁️ Phase 3: Cloud Deployment (The Wild)
**Goal:** Memindahkan sistem dari Localhost ke Public Internet.
**Status:** [ ] Not Started

### 3.1 Infrastructure as a Service (AWS EC2)
- [ ] Sewa instance EC2 (Ubuntu).
- [ ] Setup Docker & Docker Compose di server.
- [ ] Clone repo & pull `.env` (secara aman).
- [ ] Pointing domain (misal: `api.my-ml-project.com`) ke IP Public server.

### 3.2 SSL/HTTPS Security (Nginx / Certbot)
- [ ] Pasang Nginx sebagai Reverse Proxy di depan container Backend & Frontend.
- [ ] Generate sertifikat SSL gratis via Let's Encrypt (agar gembok hijau https).

### 3.3 CI/CD Deployment (Continuous Deployment)
- [ ] Update GitHub Actions: Setelah training selesai & test pass, otomatis SSH ke server EC2 dan jalankan `docker-compose up -d --build` untuk update aplikasi live.

---

## 🔒 Phase 4: Security & Hardening
**Goal:** Memastikan sistem aman dari serangan dasar dan kebocoran data.
**Status:** [ ] Not Started

- [ ] **Secrets Management:** Pindahkan `.env` production ke Secret Manager (AWS Secrets Manager / HashiCorp Vault) alih-alih file teks.
- [ ] **Rate Limiting:** Batasi request API (misal: max 100 req/menit) untuk mencegah DDoS sederhana.
- [ ] **Docker Hardening:** Pastikan semua container berjalan sebagai *non-root user* (sudah diterapkan di backend/frontend, validasi ulang).

---

## 📂 Resources & References
* [Pytest Documentation](https://docs.pytest.org/)
* [Prometheus for Python Apps](https://github.com/rycus86/prometheus_flask_exporter)
* [Evidently AI (Drift Detection)](https://github.com/evidentlyai/evidently)
* [GitHub Actions for AWS EC2](https://github.com/marketplace/actions/ssh-remote-commands)
