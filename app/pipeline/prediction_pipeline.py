from typing import Any

from app.components.metrics import record_inference
from app.components.postprocessor import DetectionPostprocessor
from app.components.preprocessor import ImagePreprocessor
from app.components.runner import get_runner
from app.entity.config_entity import InferenceConfig
from core.logger import get_logger

log = get_logger("pipeline.prediction")


class PredictionPipeline:
    """
    Orchestrates the full inference pipeline for a single request.

    Components wired together here (in order):
        1. ImagePreprocessor          — bytes → float32 tensor
        2. ONNXRunner                 — tensor → raw model output
        3. DetectionPostprocessor     — raw output → structured detections

    Also records Prometheus metrics after every successful inference.

    The router calls ``PredictionPipeline.run(image_bytes)`` and receives
    a result dict ready to be returned as a JSON response.

    This class does NOT import FastAPI — it is fully testable in isolation.
    """

    def __init__(self, inference_cfg: InferenceConfig) -> None:
        self._cfg = inference_cfg
        self._preprocessor = ImagePreprocessor(imgsz=inference_cfg.imgsz)
        self._postprocessor = DetectionPostprocessor(
            imgsz=inference_cfg.imgsz,
            conf_threshold=inference_cfg.conf_threshold,
            iou_threshold=inference_cfg.iou_threshold,
            max_detections=inference_cfg.max_detections,
        )

    def run(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Parameters
        ----------
        image_bytes : bytes
            Raw bytes of the uploaded image.

        Returns
        -------
        dict
            Keys: model_version, inference_time_ms, detections,
                  image_shape, request_id.

        Raises
        ------
        ValueError
            If the image cannot be decoded.
        RuntimeError
            If the ONNX runner is not yet initialized.
        """
        runner = get_runner()

        if not runner.is_ready:
            raise RuntimeError("Model is not ready yet — try again shortly")

        blob = self._preprocessor.run(image_bytes)
        raw_output, latency_ms, req_id = runner.run(blob)
        detections = self._postprocessor.run(raw_output)

        # Record Prometheus metrics
        record_inference(latency_ms, len(detections))

        log.info(
            "prediction_complete",
            request_id=req_id,
            n_detections=len(detections),
            latency_ms=latency_ms,
        )

        return {
            "model_version": runner.version,
            "inference_time_ms": latency_ms,
            "detections": detections,
            "image_shape": [self._cfg.imgsz, self._cfg.imgsz],
            "request_id": req_id,
        }
