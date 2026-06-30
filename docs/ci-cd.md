# CI/CD

**File:** `.github/workflows/pipeline.yml`

---

## Jobs overview

```
push to any branch
  └── quality (ruff lint + mypy)
        └── test (pytest tests/serve/)

push to main
  └── quality
        └── test
              fail-fast:
                ├── pipeline-staging
                │     → dvc repro
                │     → register to Staging
                └── docker-build
                      → build & push image to ghcr.io

workflow_dispatch (manual)
  └── promote-production
        → transition Staging → Production in MLflow
```

---

## Job 1 — Code Quality

**Trigger:** every push to any branch  
**Runs:** ruff lint, ruff format check, mypy

```bash
ruff check .
ruff format --check .
mypy app/ core/ pipeline/ --ignore-missing-imports
```

Mypy is set to `continue-on-error: true` until full type coverage is complete — it warns but does not block.

---

## Job 2 — Tests

**Trigger:** after Job 1 passes, every branch  
**Runs:** `pytest tests/serve/` with coverage report

Coverage report is uploaded as a GitHub Actions artifact (`coverage-report`).

---

## Job 3 — Pipeline → Staging

**Trigger:** push to `main` only (not on `workflow_dispatch`)  
**Environment:** `staging`

Steps:
1. `dvc remote modify origin --local` — inject DagsHub credentials
2. `dvc pull` — fetch dataset matching current `dvc.lock`
3. `pytest tests/test_data.py` — pre-pipeline gate
4. `dvc repro` — reruns only stages whose inputs changed
5. `pytest tests/test_export.py tests/test_benchmark.py` — post-pipeline gate
6. `python -m pipeline.stage5_register --stage Staging` — register to Staging
7. Upload `artifacts/` as GitHub artifact (retained 30 days)

---

## Job 4 — Docker Build & Push

**Trigger:** push to `main` (runs in parallel with pipeline-staging)  
**Environment:** `staging`

Builds the serving image and pushes to GitHub Container Registry:

```bash
docker build -t ghcr.io/${{ github.repository }}/rescuevision-serve:latest .
docker tag ...:latest ...:${{ github.sha }}
docker push --all-tags ghcr.io/${{ github.repository }}/rescuevision-serve
```

The image includes only the serving layer (`app/`, `core/`, `configs/`). The model is pulled at runtime from MLflow, not baked into the image.

---

## Job 5 — Promote to Production

**Trigger:** manual only (`workflow_dispatch` with `promote_to_production = true`)  
**Environment:** `production`

Transitions all three model versions from Staging → Production in MLflow:
- `rescuevision-onnx-fp32`
- `rescuevision-onnx-int8`
- `rescuevision-tflite-int8`

Archives any previously active Production versions automatically.

**To trigger:** GitHub → Actions → RescueVision MLOps Pipeline → Run workflow → set `promote_to_production = true`

---

## GitHub secrets required

| Secret | Value |
|---|---|
| `DAGSHUB_USERNAME` | Your DagsHub username |
| `DAGSHUB_REPO` | `rescuevision-mlops` |
| `DAGSHUB_TOKEN` | DagsHub access token |
| `GITHUB_TOKEN` | Auto-injected — has push permission to ghcr.io for the repo |

For `docker-build`, the built-in `GITHUB_TOKEN` authenticates to `ghcr.io`. No additional PAT needed.

---

## GitHub environments

Two environments must be created in repo settings:

| Environment | Protection |
|---|---|
| `staging` | None — auto-deployed on push to main |
| `production` | Require manual approval before deployment |

Set up: **Settings → Environments → New environment**

For `production`, enable "Required reviewers" and add yourself or your team.

---

## Branch strategy

| Branch pattern | What happens |
|---|---|
| `feat/*`, `fix/*` | lint + test only |
| `main` | lint → test → full pipeline + docker build → register to Staging |
| manual trigger | promote Staging → Production |

Pre-commit hook prevents committing directly to `main`:
```yaml
- id: no-commit-to-branch
  args: [--branch, main]
```
