"""
Stage: Promote Staging → Production with mAP50 regression guard + auto-rollback.

Flow:
    1. Compare Staging vs Production mAP50.
    2. Promote if regression ≤ REGRESSION_TOLERANCE (1%).
    3. Optionally verify the live serving endpoint via /health polling.
    4. If verification fails, roll back to the previous Production version.

Usage:
    python -m pipeline.stage_promote

Environment variables:
    Required: DAGSHUB_USERNAME, DAGSHUB_REPO, DAGSHUB_TOKEN
    Optional: SERVING_URL   — if set, health check + auto-rollback are performed
              VERIFY_SECS   — seconds to poll after promotion (default: 60)
"""

import os
import sys
import time

import mlflow
import requests
from mlflow.tracking import MlflowClient

from app.utils.telegram import notify
from core.logger import get_logger

log = get_logger("stage_promote")

REGRESSION_TOLERANCE = 0.01   # allow up to 1% mAP50 drop vs current production
MAP_METRIC_KEY = "mAP50"
HEALTH_ERROR_THRESHOLD = 0.05  # rollback if error rate > 5%

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
    versions = client.search_model_versions(f"name = '{name}' and tags.stage = '{stage}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))


def _rollback(client: MlflowClient, name: str, previous_version: str) -> None:
    """Re-promote a previous Production version after a failed health check."""
    try:
        client.transition_model_version_stage(
            name, previous_version, "Production", archive_existing_versions=True
        )
        log.info("rollback_complete", extra={"model": name, "restored_version": previous_version})
    except Exception as exc:
        log.error("rollback_failed", extra={"model": name, "error": str(exc)})


def _verify_serving(url: str, verify_secs: int) -> bool:
    """
    Poll /health every 10 s for `verify_secs` seconds.
    Returns True if the endpoint stays healthy; False on repeated failures.
    """
    deadline = time.time() + verify_secs
    failures = 0
    total = 0

    log.info("health_poll_start", extra={"url": url, "duration_s": verify_secs})
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            total += 1
            if r.status_code != 200:
                failures += 1
                log.warning("health_check_failed", extra={"status": r.status_code})
        except Exception as exc:
            total += 1
            failures += 1
            log.warning("health_check_error", extra={"error": str(exc)})
        time.sleep(10)

    if total == 0:
        return True  # no checks ran — assume ok

    error_rate = failures / total
    log.info(
        "health_poll_complete",
        extra={"total": total, "failures": failures, "error_rate": round(error_rate, 3)},
    )
    return error_rate <= HEALTH_ERROR_THRESHOLD


def promote(client: MlflowClient, name: str) -> bool:
    staging_v = _latest_version_in_stage(client, name, "Staging")
    if not staging_v:
        log.warning("no_staging_version", extra={"model": name})
        return False

    prod_v = _latest_version_in_stage(client, name, "Production")
    prev_prod_version = prod_v.version if prod_v else None

    staging_map = _get_map50(client, staging_v.run_id)
    prod_map = _get_map50(client, prod_v.run_id if prod_v else None)

    log.info(
        "promotion_comparison",
        extra={
            "model": name,
            "staging_version": staging_v.version,
            "staging_mAP50": staging_map,
            "prod_version": prev_prod_version,
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


def verify_and_rollback_if_needed(
    client: MlflowClient, name: str, prev_prod_version: str | None, serving_url: str, verify_secs: int
) -> bool:
    """
    Returns True if serving is healthy after promotion.
    Triggers rollback and returns False if health check fails.
    """
    healthy = _verify_serving(serving_url, verify_secs)
    if not healthy:
        log.error(
            "post_promotion_health_check_failed_rolling_back",
            extra={"model": name, "prev_version": prev_prod_version},
        )
        if prev_prod_version:
            _rollback(client, name, prev_prod_version)
        return False
    log.info("post_promotion_health_check_passed", extra={"model": name})
    return True


if __name__ == "__main__":
    client = _setup_mlflow()
    serving_url = os.environ.get("SERVING_URL", "")
    verify_secs = int(os.environ.get("VERIFY_SECS", "60"))

    failed = []
    # Track previous production versions per model for potential rollback
    prev_versions: dict[str, str | None] = {}

    for model_name in MODELS:
        prod_v = _latest_version_in_stage(client, model_name, "Production")
        prev_versions[model_name] = prod_v.version if prod_v else None

        ok = promote(client, model_name)
        if not ok:
            failed.append(model_name)

    if failed:
        log.error("some_promotions_failed", extra={"models": failed})
        notify(f"❌ *Promotion failed*\nModels: `{', '.join(failed)}`\nCommit: `{os.environ.get('GITHUB_SHA', 'local')[:7]}`")
        sys.exit(1)

    log.info("all_models_promoted_to_production")
    notify(f"✅ *All models promoted to Production*\nCommit: `{os.environ.get('GITHUB_SHA', 'local')[:7]}`")

    # Optional: verify serving health and rollback if unhealthy
    if serving_url:
        log.info("starting_post_promotion_verification", extra={"url": serving_url, "secs": verify_secs})
        rollback_failures = []
        for model_name in MODELS:
            ok = verify_and_rollback_if_needed(
                client, model_name, prev_versions[model_name], serving_url, verify_secs
            )
            if not ok:
                rollback_failures.append(model_name)

        if rollback_failures:
            log.error("rollback_triggered_for_models", extra={"models": rollback_failures})
            notify(f"🔴 *Auto-rollback triggered*\nHealth check failed after promotion.\nModels rolled back: `{', '.join(rollback_failures)}`")
            sys.exit(1)
    else:
        log.info("serving_url_not_set_skipping_health_verification")
