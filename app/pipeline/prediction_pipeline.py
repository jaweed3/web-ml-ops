from typing import Any

from app.components.metrics import record_inference
from app.components.postprocessor import DetectionPostprocessor
from app.components.prediction_logger import log_prediction
from app.components.preprocessor import ImagePreprocessor
from app.components.runner import ONNXRunner
from core.config import InferenceConfig
from core.logger import get_logger

log = get_logger("pipeline.prediction")


class PredictionPipeline:
    """
    Orchestrates the full inference pipeline for a single request.
    Wires preprocessor → ONNX runner → postprocessor in sequence.
    Fully testable in isolation — no module globals, no FastAPI imports.
    """

    def __init__(self, runner: ONNXRunner, inference_cfg: InferenceConfig) -> None:
        self._runner = runner
        self._imgsz = inference_cfg.imgsz
        self._preprocessor = ImagePreprocessor(imgsz=inference_cfg.imgsz)
        self._postprocessor = DetectionPostprocessor(
            imgsz=inference_cfg.imgsz,
            conf_threshold=inference_cfg.conf_threshold,
            iou_threshold=inference_cfg.iou_threshold,
            max_detections=inference_cfg.max_detections,
        )

    def run(self, image_bytes: bytes) -> dict[str, Any]:
        if not self._runner.is_ready:
            raise RuntimeError("Model is not ready yet — try again shortly")

        blob = self._preprocessor.run(image_bytes)
        raw_output, latency_ms, req_id = self._runner.run(blob)
        detections = self._postprocessor.run(raw_output)

        record_inference(latency_ms, len(detections))

        log_prediction(
            request_id=req_id,
            model_version=self._runner.version,
            n_detections=len(detections),
            inference_time_ms=latency_ms,
        )

        log.info(
            "prediction_complete",
            request_id=req_id,
            n_detections=len(detections),
            latency_ms=latency_ms,
        )

        return {
            "model_version": self._runner.version,
            "inference_time_ms": latency_ms,
            "detections": detections,
            "image_shape": [self._imgsz, self._imgsz],
            "request_id": req_id,
        }
