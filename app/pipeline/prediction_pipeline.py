from typing import Any

from app.components.metrics import record_drift, record_inference
from app.components.postprocessor import DetectionPostprocessor
from app.components.prediction_logger import log_prediction
from app.components.preprocessor import ImagePreprocessor
from app.components.runner import ONNXRunner
from app.monitoring.drift import DriftDetector
from core.config import InferenceConfig
from core.logger import get_logger

log = get_logger("pipeline.prediction")


class PredictionPipeline:
    """
    Orchestrates the full inference pipeline for a single request.
    Wires preprocessor → ONNX runner → postprocessor in sequence.
    Optionally runs drift detection and shadow model inference.
    """

    def __init__(
        self,
        runner: ONNXRunner,
        inference_cfg: InferenceConfig,
        drift_detector: DriftDetector | None = None,
        shadow_runner: ONNXRunner | None = None,
    ) -> None:
        self._runner = runner
        self._shadow_runner = shadow_runner
        self._imgsz = inference_cfg.imgsz
        self._drift = drift_detector
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

        blob, img_stats = self._preprocessor.run(image_bytes)
        raw_output, latency_ms, req_id = self._runner.run(blob)
        detections = self._postprocessor.run(raw_output)

        record_inference(latency_ms, len(detections))

        if self._drift and self._drift.is_active:
            drift = self._drift.score(img_stats)
            record_drift(drift)

        # ponytail: shadow runner — silent dual inference for A/B comparison
        if self._shadow_runner is not None:
            self._run_shadow(blob, req_id)

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

    def _run_shadow(self, blob: Any, req_id: str) -> None:
        try:
            shadow_out, shadow_latency, _ = self._shadow_runner.run(blob)
            shadow_det = self._postprocessor.run(shadow_out)
            log.info(
                "shadow_inference",
                request_id=req_id,
                shadow_version=self._shadow_runner.version,
                shadow_detections=len(shadow_det),
                shadow_latency_ms=shadow_latency,
            )
        except Exception as exc:
            log.warning("shadow_inference_failed", request_id=req_id, error=str(exc))
