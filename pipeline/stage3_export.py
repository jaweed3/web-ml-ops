import shutil
from pathlib import Path

from core.logger import get_logger
from core.config import load_config
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

log = get_logger("stage3_export")
ARTIFACTS_DIR = Path("artifacts")


def export_onnx_fp32(checkpoint: str, imgsz: int) -> Path:
    model = YOLO(checkpoint)
    model.export(format="onnx", imgsz=imgsz, dynamic=False)
    src = Path(checkpoint).with_suffix(".onnx")
    dst = ARTIFACTS_DIR / "model.onnx"
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    shutil.copy(src, dst)
    log.info("export_onnx_fp32", path=str(dst), size_mb=round(dst.stat().st_size / 1e6, 2))
    return dst


def export_onnx_int8(fp32_path: Path) -> Path:
    dst = ARTIFACTS_DIR / "model_int8.onnx"
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(dst),
        weight_type=QuantType.QUInt8,
    )
    log.info("export_onnx_int8", path=str(dst), size_mb=round(dst.stat().st_size / 1e6, 2))
    return dst


def export_tflite_int8(checkpoint: str, imgsz: int) -> Path:
    model = YOLO(checkpoint)
    model.export(format="tflite", imgsz=imgsz, int8=True)
    src = Path(checkpoint).parent / "best_int8.tflite"
    dst = ARTIFACTS_DIR / "model_int8.tflite"
    shutil.copy(src, dst)
    log.info("export_tflite_int8", path=str(dst), size_mb=round(dst.stat().st_size / 1e6, 2))
    return dst


if __name__ == "__main__":
    cfg = load_config()
    checkpoint = "runs/train/weights/best.pt"
    fp32 = export_onnx_fp32(checkpoint, cfg.train.imgsz)
    export_onnx_int8(fp32)
    export_tflite_int8(checkpoint, cfg.train.imgsz)
