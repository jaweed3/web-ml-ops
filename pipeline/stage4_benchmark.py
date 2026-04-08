import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from core.logger import get_logger
from core.config import load_config
from core.mlflow_client import log_metrics, log_artifact

log = get_logger("stage4_benchmark")
N_SAMPLES = 100


def _preprocess(img_path: str) -> np.ndarray:
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (640, 640))
    blob = img.astype(np.float32) / 255.0
    return np.transpose(blob, (2, 0, 1))[None]


def benchmark_onnx(model_path: str, val_images: list, label: str) -> dict:
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    latencies = []

    for img_path in val_images[:N_SAMPLES]:
        blob = _preprocess(img_path)
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
    log.info("benchmark_result", **result)
    return result


if __name__ == "__main__":
    cfg = load_config()
    val_images = sorted(Path("data/coco_person/images/val").glob("*.jpg"))

    if len(val_images) == 0:
        raise RuntimeError("No validation images found — run Stage 1 first")

    results = [
        benchmark_onnx("artifacts/model.onnx", val_images, "onnx_fp32"),
        benchmark_onnx("artifacts/model_int8.onnx", val_images, "onnx_int8"),
    ]

    report = {"n_samples": N_SAMPLES, "results": results}
    report_path = Path("artifacts/benchmark_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    log.info("benchmark_report_saved", path=str(report_path))

    for r in results:
        prefix = r["format"]
        log_metrics({
            f"{prefix}_mean_latency_ms": r["mean_latency_ms"],
            f"{prefix}_p95_latency_ms": r["p95_latency_ms"],
            f"{prefix}_model_size_mb": r["model_size_mb"],
        })
    log_artifact(str(report_path))
