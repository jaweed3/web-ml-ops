import json
import time
from pathlib import Path

import cv2
import mlflow
import numpy as np
import onnxruntime as ort

from core.config import load_config
from core.logger import get_logger
from core.mlflow_client import init_mlflow, log_artifact, log_metrics

log = get_logger("stage4_benchmark")
N_SAMPLES = 100


def _preprocess(img_path: str, imgsz: int) -> np.ndarray:
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (imgsz, imgsz))
    blob = img.astype(np.float32) / 255.0
    return np.transpose(blob, (2, 0, 1))[None]


def benchmark_onnx(model_path: str, val_images: list, label: str, imgsz: int) -> dict:
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    latencies = []

    for img_path in val_images[:N_SAMPLES]:
        blob = _preprocess(img_path, imgsz)
        t0 = time.perf_counter()
        sess.run(None, {input_name: blob})
        latencies.append((time.perf_counter() - t0) * 1000)

    size_mb = round(Path(model_path).stat().st_size / 1e6, 2)
    result = {
        "format": label,
        "mean_latency_ms": round(np.mean(latencies), 2),
        "p95_latency_ms": round(np.percentile(latencies, 95), 2),
        "model_size_mb": size_mb,
    }
    log.info("benchmark_result", extra=result)
    return result

def get_latest_map() -> dict:
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("rescuevision-yolov8n")
    if exp is None:
        log.warning("mlflow_experiment is not found ", extra={
            "name": "rescuevision-yolov8n"
        })
        return {}
    runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
    if not runs:
        return {}
    metrics = runs[0].data.metrics
    return {
        "mAP50_fp32": metrics.get("mAP50", None)
    }

if __name__ == "__main__":
    init_mlflow(experiment_name="rescuevision_benchmark")
    cfg = load_config()
    val_images = sorted(Path("data/coco_person/images/val").glob("*.jpg"))

    if len(val_images) == 0:
        raise RuntimeError("No validation images found — run Stage 1 first")

    results = [
        benchmark_onnx("artifacts/model.onnx", val_images, "onnx_fp32", cfg.train.imgsz),
        benchmark_onnx("artifacts/model_int8.onnx", val_images, "onnx_int8", imgsz=cfg.train.imgsz),
    ]

    map_metrics = get_latest_map()
    for r in results:
        if r["format"] == "onnx_fp32":
            r["mAP50"] = map_metrics.get("mAP50_fp32")
        elif r["format"] == "onnx_int8":
            r["mAP50"] = map_metrics.get("mAP50_fp32")

    report = {"n_samples": N_SAMPLES, "results": results}
    report_path = Path("artifacts/benchmark_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    log.info("benchmark_report_saved", extra={
        "path":str(report_path)})

    for r in results:
        prefix = r["format"]
        log_metrics(
            {
                f"{prefix}_mean_latency_ms": r["mean_latency_ms"],
                f"{prefix}_p95_latency_ms": r["p95_latency_ms"],
                f"{prefix}_model_size_mb": r["model_size_mb"],
            }
        )
    log_artifact(str(report_path))
