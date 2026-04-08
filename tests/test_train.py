"""
Stage 2 — Training tests.

Requires:
  - `make train` (or `dvc repro`) already run
  - DAGSHUB_USERNAME, DAGSHUB_REPO, DAGSHUB_TOKEN set in environment
"""

import os
from pathlib import Path

import pytest

CHECKPOINT = Path("runs/train/weights/best.pt")

# ── Checkpoint ────────────────────────────────────────────────────────────────

def test_checkpoint_exists():
    assert CHECKPOINT.exists(), "best.pt not found — run `make train` first"


def test_checkpoint_not_empty():
    """A valid YOLOv8n checkpoint is at least 5 MB."""
    size_mb = CHECKPOINT.stat().st_size / 1_000_000
    assert size_mb > 5, f"Checkpoint suspiciously small: {size_mb:.2f} MB"


def test_checkpoint_is_loadable():
    """Verify the checkpoint can be loaded by Ultralytics without errors."""
    pytest.importorskip("ultralytics", reason="ultralytics not installed — skipping (training machine only)")
    from ultralytics import YOLO

    model = YOLO(str(CHECKPOINT))
    assert model is not None


# ── MLflow experiment ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mlflow_runs():
    mlflow = pytest.importorskip("mlflow")
    username = os.environ.get("DAGSHUB_USERNAME")
    repo = os.environ.get("DAGSHUB_REPO")

    if not username or not repo:
        pytest.skip("DAGSHUB_USERNAME / DAGSHUB_REPO not set — skipping MLflow tests")

    uri = f"https://dagshub.com/{username}/{repo}.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("DAGSHUB_TOKEN", "")
    mlflow.set_tracking_uri(uri)

    runs = mlflow.search_runs(experiment_names=["rescuevision-yolov8n"])
    if runs.empty:
        pytest.skip("No MLflow runs found for experiment 'rescuevision-yolov8n'")
    return runs


def test_mlflow_run_exists(mlflow_runs):
    assert len(mlflow_runs) > 0


def test_mlflow_run_has_map50(mlflow_runs):
    assert "metrics.mAP50" in mlflow_runs.columns, (
        "mAP50 not logged — check stage2_train.py"
    )


def test_mlflow_map50_above_minimum(mlflow_runs):
    """
    Minimum mAP50 quality gate: 0.30.
    Adjust this threshold based on your dataset size and training epochs.
    """
    best_map50 = mlflow_runs["metrics.mAP50"].max()
    assert best_map50 > 0.30, (
        f"Best mAP50 {best_map50:.4f} is below minimum threshold 0.30 — "
        "model quality too low to proceed"
    )


def test_mlflow_run_has_training_params(mlflow_runs):
    for param in ("params.model", "params.epochs", "params.batch"):
        assert param in mlflow_runs.columns, f"Missing logged param: {param}"


def test_mlflow_checkpoint_artifact_logged(mlflow_runs):
    """The best.pt artifact should be logged in at least one run."""
    import mlflow
    latest_run_id = mlflow_runs.iloc[0]["run_id"]
    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(latest_run_id)]
    assert any("best.pt" in a or "weights" in a for a in artifacts), (
        f"best.pt not found in MLflow artifacts. Found: {artifacts}"
    )
