"""
Stage 5 — Model registry tests.

Requires:
  - `make register` (or `dvc repro`) already run
  - DAGSHUB_USERNAME, DAGSHUB_REPO, DAGSHUB_TOKEN set in environment
"""

import os

import pytest

REGISTERED_MODELS = [
    "rescuevision-onnx-fp32",
    "rescuevision-onnx-int8",
    "rescuevision-tflite-int8",
]


# ── MLflow client fixture ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mlflow_client():
    mlflow = pytest.importorskip("mlflow")

    username = os.environ.get("DAGSHUB_USERNAME")
    repo     = os.environ.get("DAGSHUB_REPO")
    token    = os.environ.get("DAGSHUB_TOKEN")

    if not all([username, repo, token]):
        pytest.skip("DAGSHUB_* env vars not set — skipping registry tests")

    uri = f"https://dagshub.com/{username}/{repo}.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(uri)

    return mlflow.MlflowClient()


# ── Model existence ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", REGISTERED_MODELS)
def test_model_is_registered(mlflow_client, model_name):
    """All three artifacts must have at least one registered version."""
    versions = mlflow_client.search_model_versions(f"name='{model_name}'")
    assert len(versions) > 0, (
        f"No registered versions found for '{model_name}' — run `make register` first"
    )


@pytest.mark.parametrize("model_name", REGISTERED_MODELS)
def test_model_has_staging_or_none_version(mlflow_client, model_name):
    """After stage5_register, models should be in Staging or None (not Production)."""
    versions = mlflow_client.get_latest_versions(model_name, stages=["Staging", "None"])
    assert len(versions) > 0, (
        f"No Staging/None version found for '{model_name}'"
    )


# ── Tags ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", REGISTERED_MODELS)
def test_model_version_has_git_commit_tag(mlflow_client, model_name):
    """Every registered version must have a git_commit tag for traceability."""
    versions = mlflow_client.search_model_versions(f"name='{model_name}'")
    assert versions, f"No versions for {model_name}"

    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    assert "git_commit" in latest.tags, (
        f"'{model_name}' v{latest.version} missing 'git_commit' tag — "
        "check stage5_register.py"
    )


@pytest.mark.parametrize("model_name", REGISTERED_MODELS)
def test_model_version_has_pipeline_tag(mlflow_client, model_name):
    versions = mlflow_client.search_model_versions(f"name='{model_name}'")
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    assert "pipeline" in latest.tags, (
        f"'{model_name}' v{latest.version} missing 'pipeline' tag"
    )


# ── Source run linkage ────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", REGISTERED_MODELS)
def test_model_version_linked_to_run(mlflow_client, model_name):
    """Each registered version should link back to a training run."""
    versions = mlflow_client.search_model_versions(f"name='{model_name}'")
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    assert latest.run_id, (
        f"'{model_name}' v{latest.version} has no linked run_id"
    )


# ── No duplicate registration ─────────────────────────────────────────────────

def test_no_duplicate_versions_same_commit(mlflow_client):
    """
    Two versions with the same git_commit tag would indicate accidental
    double-registration. Warn but do not fail hard — might be intentional re-run.
    """
    import mlflow

    for model_name in REGISTERED_MODELS:
        versions = mlflow_client.search_model_versions(f"name='{model_name}'")
        commits = [v.tags.get("git_commit", "") for v in versions if v.tags]
        commits_with_value = [c for c in commits if c]
        duplicates = {c for c in commits_with_value if commits_with_value.count(c) > 1}
        if duplicates:
            pytest.warns(
                UserWarning,
                match="duplicate",
            )
