import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from ultralytics import YOLO

from app.constant import EXPERIMENT_NAME
from core.config import is_subset_mode, load_config
from core.logger import get_logger
from core.mlflow_client import init_mlflow
from core.mlflow_client import log_artifact as mlflow_log

log = get_logger("stage3_export")
ARTIFACTS_DIR = Path("artifacts")


def export_onnx_fp32(checkpoint: str, imgsz: int) -> Path:
    model = YOLO(checkpoint)
    model.export(format="onnx", imgsz=imgsz, dynamic=False)
    src = Path(checkpoint).with_suffix(".onnx")
    dst = ARTIFACTS_DIR / "model.onnx"
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    shutil.copy(src, dst)
    log.info(
        "export_onnx_fp32", extra={"path": str(dst), "size_mb": round(dst.stat().st_size / 1e6, 2)}
    )
    return dst


def export_onnx_int8(fp32_path: Path) -> Path:
    dst = ARTIFACTS_DIR / "model_int8.onnx"
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(dst),
        weight_type=QuantType.QUInt8,
    )
    log.info(
        "export_onnx_int8", extra={"path": str(dst), "size_mb": round(dst.stat().st_size / 1e6, 2)}
    )
    return dst


def export_tflite_int8(checkpoint: str, imgsz: int, data_yaml: str) -> Path | None:
    """
    TFLite export requires TensorFlow (heavy dep, GPU machine only).
    Skips gracefully if TF is not installed so the ONNX artifacts
    can still be produced on machines without TF.
    """
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        log.warning(
            "tflite_export_skipped",
            extra={
                "reason": "tensorflow not installed — run `make install train`",
            },
        )
        return None

    model = YOLO(checkpoint)
    try:
        model.export(format="tflite", imgsz=imgsz, int8=True, data=data_yaml)
    except Exception as exc:
        log.warning("tflite_export_failed", extra={"error": str(exc)})
        return None

    saved_model_dir = Path(checkpoint).parent / "best_saved_model"
    candidates = list(saved_model_dir.glob("best_full_integer_quant*.tflite"))
    if not candidates:
        candidates = list(saved_model_dir.glob("best_integer_quant*.tflite"))
    if not candidates:
        candidates = list(saved_model_dir.glob("*.tflite"))
    if not candidates:
        log.warning("tflite model is not found : ", extra={"searched": str(saved_model_dir)})
        return None

    src = candidates[0]
    dst = ARTIFACTS_DIR / "model_int8.tflite"
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    shutil.copy(src, dst)
    log.info(
        "export_tflite_int8",
        extra={"path": str(dst), "size_mb": round(dst.stat().st_size / 1e6, 2)},
    )
    return dst


if __name__ == "__main__":
    init_mlflow(experiment_name=EXPERIMENT_NAME)
    cfg = load_config()
    checkpoint_file = Path("artifacts/checkpoint_path.txt")
    if checkpoint_file.exists():
        checkpoint = checkpoint_file.read_text().strip()
    else:
        # fallback for local runs
        best = Path("runs/detect/runs/train/weights/best.pt")
        last = Path("runs/detect/runs/train/weights/last.pt")
        checkpoint = str(best if best.exists() else last)
    data_yaml = cfg.data.subset_yaml if is_subset_mode() else cfg.data.yaml
    fp32 = export_onnx_fp32(checkpoint, cfg.train.imgsz)
    export_onnx_int8(fp32)
    tflite = export_tflite_int8(checkpoint, cfg.train.imgsz, data_yaml)
    mlflow_log("artifacts/model.onnx")
    mlflow_log("artifacts/model_int8.onnx")
    if tflite:
        mlflow_log("artifacts/model_int8.tflite")
