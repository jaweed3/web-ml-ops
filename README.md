# 🚀 End-to-End MLOps Pipeline Template

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Next.js](https://img.shields.io/badge/Frontend-Next.js-black?style=for-the-badge&logo=next.js&logoColor=white)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![DagsHub](https://img.shields.io/badge/Storage-DagsHub-green?style=for-the-badge)

A robust, production-ready MLOps project template designed for scalability and automation. This architecture decouples **Model Training**, **Inference Serving**, and **User Interface** into isolated microservices.

---

## 🏗 Architecture Overview

1.  **Training Pipeline (Python/DVC):** Modular pipeline (`main.py`) that handles Data Ingestion, Transformation, Training, and Evaluation.
2.  **Model Registry (DagsHub/MLflow):** Successfully trained models are versioned and uploaded to a remote S3 bucket via DagsHub.
3.  **Backend (Flask + Gunicorn):** A containerized API that downloads the production model from the cloud and serves predictions.
4.  **Frontend (Next.js):** A clean, responsive UI for end-users to interact with the model.
5.  **CI/CD (GitHub Actions):** Automated training and reporting triggered by git push events.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10, Node.js 18
* **Orchestration:** Docker & Docker Compose
* **Tracking & Registry:** MLflow (Remote DagsHub)
* **Backend:** Flask (Production Server: Gunicorn)
* **Frontend:** Next.js (Standalone Output)
* **CI/CD:** GitHub Actions (CML)

---

## 🚀 Quick Start (Local Development)

### Prerequisites
* Docker Desktop / Docker Engine
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

```

### 2. Configure Environment Variables

Create a `.env` file in the **root directory**. You can copy the template:

```bash
cp .env.example .env

```

Fill in your DagsHub credentials (Required for model downloading):

```env
MLFLOW_TRACKING_URI=[https://dagshub.com/](https://dagshub.com/)<username>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-password-or-token>
DAGSHUB_REPO_OWNER=<your-dagshub-username>
DAGSHUB_REPO_NAME=<your-dagshub-repo-name>
DAGSHUB_USER_TOKEN=<your-dagshub-access-token>

```

> **Note:** The `DAGSHUB_USER_TOKEN` is critical for containerized authentication without browser interaction.

### 3. Launch with Docker Compose

Run the entire stack (Backend + Frontend) with a single command:

```bash
docker-compose up --build

```

### 4. Access the Application

* **Frontend UI:** [http://localhost:3000](https://www.google.com/search?q=http://localhost:3000)
* **Backend API:** [http://localhost:8080](https://www.google.com/search?q=http://localhost:8080)
* **MLflow UI:** [https://dagshub.com](https://dagshub.com) (Check your repo)

---

## 🔄 MLOps Workflow

### Manual Training (Local)

To retrain the model locally and push artifacts to DagsHub:

```bash
cd backend
# Ensure env vars are set
python main.py

```

### Automated Training (CI/CD)

This project is configured with **GitHub Actions**.

1. Push changes to the `main` branch.
2. The pipeline will automatically provision a runner.
3. It executes `main.py`, trains the model, and uploads the new version to DagsHub Registry.

---

## 📂 Project Structure

```
├── .github/workflows   # CI/CD Pipeline (CML)
├── backend             # Python ML Pipeline & Flask API
│   ├── artifacts       # Local artifacts (ignored by git)
│   ├── config          # Pipeline configurations
│   ├── src             # Source code (Components & Pipelines)
│   ├── app.py          # Flask Entrypoint
│   ├── main.py         # Training Orchestrator
│   └── Dockerfile      # Backend Container Config
├── frontend            # Next.js Application
│   ├── src             # React Components
│   └── Dockerfile      # Frontend Container Config
├── docker-compose.yaml # Orchestration
└── requirements.txt    # Python Dependencies

```

---

## 🔮 Future Roadmap

* [ ] Implement Unit Testing (`pytest`) for data validation.
* [ ] Add Prometheus & Grafana for system & model monitoring.
* [ ] Deploy to AWS EC2 / Kubernetes.

---

## 👨‍💻 Author

**Fatih Jawwad** - *Machine Learning Engineer*

