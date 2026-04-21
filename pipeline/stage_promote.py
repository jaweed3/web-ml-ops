"""
Stage: Promote Staging → Production with mAP50 regression guard.

Fetches mAP50 from the MLflow run linked to each Staging model version,
compares it against the current Production model's mAP50, and only promotes
if the new model is not worse by more than REGRESSION_TOLERANCE.

Usage:
    python -m pipeline.stage_promote

Environment variables (required):
    DAGSHUB_USERNAME, DAGSHUB_REPO, DAGSHUB_TOKEN
"""

import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

from core.logger import get_logger

log = get_logger("stage_promote")

REGRESSION_TOLERANCE = 0.01  # allow up to 1% mAP50 drop vs current production
MAP_METRIC_KEY = "mAP50"

MODELS = [
    "rescuevision-onnx-fp32",
    "rescuevision-onnx-int8",
    "rescuevision-tflite-int8",
]


def _setup_mlflow() -> MlflowClient:
    username = os.environ["DAGSHUB_USERNAME"]
    repo = os.environ["DAGSHUB_REPO"]
    token = os.environ["DAGSHUB_TOKEN"]
    uri = f"https://dagshub.com/{username}/{repo}.mlflow"
    mlflow.set_tracking_uri(uri)
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    return MlflowClient()


def _get_map50(client: MlflowClient, run_id: str | None) -> float | None:
    if not run_id:
        return None
    try:
        run = client.get_run(run_id)
        return run.data.metrics.get(MAP_METRIC_KEY)
    except Exception as exc:
        log.warning("failed_to_fetch_run_metrics", extra={"run_id": run_id, "error": str(exc)})
        return None


def _latest_version_in_stage(client: MlflowClient, name: str, stage: str):
    versions = client.search_model_versions(
        f"name = '{name}' and tags.stage = '{stage}'"
    )
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))


def promote(client: MlflowClient, name: str) -> bool:
    staging_v = _latest_version_in_stage(client, name, "Staging")
    if not staging_v:
        log.warning("no_staging_version", extra={"model": name})
        return False

    prod_v = _latest_version_in_stage(client, name, "Production")

    staging_map = _get_map50(client, staging_v.run_id)
    prod_map = _get_map50(client, prod_v.run_id if prod_v else None)

    log.info(
        "promotion_comparison",
        extra={
            "model": name,
            "staging_version": staging_v.version,
            "staging_mAP50": staging_map,
            "prod_version": prod_v.version if prod_v else None,
            "prod_mAP50": prod_map,
        },
    )

    # If no production exists yet, promote unconditionally
    if prod_map is None:
        log.info("no_production_baseline_promoting", extra={"model": name})
    elif staging_map is None:
        log.warning("staging_has_no_map50_skipping", extra={"model": name})
        return False
    else:
        threshold = prod_map * (1 - REGRESSION_TOLERANCE)
        if staging_map < threshold:
            log.error(
                "regression_guard_failed_skipping_promotion",
                extra={
                    "model": name,
                    "staging_mAP50": staging_map,
                    "required_min": round(threshold, 6),
                    "prod_mAP50": prod_map,
                },
            )
            return False

    client.transition_model_version_stage(
        name, staging_v.version, "Production", archive_existing_versions=True
    )
    log.info(
        "model_promoted",
        extra={"model": name, "version": staging_v.version, "stage": "Production"},
    )
    return True


if __name__ == "__main__":
    client = _setup_mlflow()
    failed = []

    for model_name in MODELS:
        ok = promote(client, model_name)
        if not ok:
            failed.append(model_name)

    if failed:
        log.error("some_promotions_failed", extra={"models": failed})
        sys.exit(1)

    log.info("all_models_promoted_to_production")
