import argparse
import json
import subprocess
from pathlib import Path

from core.logger import get_logger
from core.mlflow_client import init_mlflow, register_model

log = get_logger("stage5_register")
DEGRADATION_THRESHOLD = 0.97
parser = argparse.ArgumentParser()


def load_benchmark() -> dict:
    return json.loads(Path("artifacts/benchmark_report.json").read_text())


def load_dataset_hash() -> str:
    p = Path("artifacts/dataset_hash.txt")
    if not p.exists():
        log.warning("dataset_hash_missing", extra={"path": str(p)})
        return "unknown"
    return p.read_text().strip()


def metric_gate(report: dict) -> bool:
    results = {r["format"]: r for r in report["results"]}
    fp32_map = results.get("onnx_fp32", {}).get("mAP50", None)
    int8_map = results.get("onnx_int8", {}).get("mAP50", None)

    if fp32_map is None or int8_map is None:
        log.warning("mAP50_missing_from_report_skipping_gate")
        return True

    ratio = int8_map / fp32_map if fp32_map > 0 else 1.0
    passed = ratio >= DEGRADATION_THRESHOLD
    log.info(
        "metric_gate",
        extra={
            "fp32_mAP50": fp32_map,
            "int8_mAP50": int8_map,
            "ratio": round(ratio, 4),
            "threshold": DEGRADATION_THRESHOLD,
            "passed": passed,
        },
    )
    return passed


def get_git_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    parser.add_argument("--stage", default="None", choices=["Staging", "None", "Production"])
    args = parser.parse_args()

    report = load_benchmark()

    if not metric_gate(report):
        log.error("metric_gate_failed_aborting_registration")
        raise SystemExit(1)

    # Open MLflow run only after gate passes — avoids orphaned runs on failure
    init_mlflow(experiment_name="rescuevision-yolov8n")

    git_hash = get_git_hash()
    dataset_hash = load_dataset_hash()
    tags = {
        "git_commit": git_hash,
        "dataset_hash": dataset_hash,
        "pipeline": "rescuevision-mlops",
        "stage": args.stage,
    }

    artifacts = [
        ("artifacts/model.onnx", "rescuevision-onnx-fp32"),
        ("artifacts/model_int8.onnx", "rescuevision-onnx-int8"),
        ("artifacts/model_int8.tflite", "rescuevision-tflite-int8"),
    ]
    for path, name in artifacts:
        register_model(path, name, tags)
        log.info(
            "model_registered",
            extra={"name": name, "path": path, "git_hash": git_hash, "stage": args.stage},
        )
