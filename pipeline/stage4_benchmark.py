import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics.models import YOLO

from core.config import is_subset_mode, load_config
from core.logger import get_logger
from core.mlflow_client import init_mlflow, log_artifact, log_metrics

log = get_logger("stage4_benchmark")
N_SAMPLES = 100


def _preprocess(img_path: str, imgsz: int) -> np.ndarray:
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (imgsz, imgsz))
    blob = img.astype(np.float32) / 255.0
    return np.transpose(blob, (2, 0, 1))[None]


def benchmark_onnx(
    model_path: str, val_images: list, label: str, imgsz: int, data_yaml: str
) -> dict:
    # 1. Measure latency over N_SAMPLES warm-up-free runs
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    latencies = []
    for img_path in val_images[:N_SAMPLES]:
        blob = _preprocess(img_path, imgsz)
        t0 = time.perf_counter()
        sess.run(None, {input_name: blob})
        latencies.append((time.perf_counter() - t0) * 1000)

    # 2. Compute real mAP50 via the YOLO validation backend
    log.info(f"evaluating_mAP_for_{label}")
    model = YOLO(model_path, task="detect")
    val_result = model.val(data=data_yaml, imgsz=imgsz, split="val", verbose=False)

    size_mb = round(Path(model_path).stat().st_size / 1_000_000, 2)

    # Per-class AP50 — list of {class_id: int, ap50: float}
    per_class = []
    if hasattr(val_result.box, "ap_class_index") and val_result.box.ap_class_index is not None:
        for cls_idx, ap50_val in zip(
            val_result.box.ap_class_index, val_result.box.ap50, strict=True
        ):
            per_class.append({"class_id": int(cls_idx), "ap50": round(float(ap50_val), 4)})

    result = {
        "format": label,
        "mean_latency_ms": round(np.mean(latencies), 2),
        "p95_latency_ms": round(np.percentile(latencies, 95), 2),
        "model_size_mb": size_mb,
        "mAP50": val_result.box.map50,
        "per_class_ap50": per_class,
    }
    log.info("benchmark_result", extra=result)
    return result


if __name__ == "__main__":
    init_mlflow(experiment_name="rescuevision-yolov8n")
    cfg = load_config()
    data_dir = cfg.data.subset_dir if is_subset_mode() else cfg.data.dir
    data_yaml = cfg.data.subset_yaml if is_subset_mode() else cfg.data.yaml
    val_images = sorted((Path(data_dir) / "images" / "val").glob("*.jpg"))

    if len(val_images) == 0:
        raise RuntimeError("No validation images found — run Stage 1 first")

    results = [
        benchmark_onnx(
            "artifacts/model.onnx", val_images, "onnx_fp32", cfg.train.imgsz, data_yaml
        ),
        benchmark_onnx(
            "artifacts/model_int8.onnx", val_images, "onnx_int8", cfg.train.imgsz, data_yaml
        ),
    ]

    report = {"n_samples": N_SAMPLES, "results": results}
    report_path = Path("artifacts/benchmark_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    log.info("benchmark_report_saved", extra={"path": str(report_path)})

    for r in results:
        prefix = r["format"]
        log_metrics(
            {
                f"{prefix}_mean_latency_ms": r["mean_latency_ms"],
                f"{prefix}_p95_latency_ms": r["p95_latency_ms"],
                f"{prefix}_model_size_mb": r["model_size_mb"],
                f"{prefix}_mAP50": r["mAP50"],
            }
        )
    log_artifact(str(report_path))
