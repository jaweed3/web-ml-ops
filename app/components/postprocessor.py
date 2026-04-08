from typing import Any

import numpy as np

from app.constant import (
    CLASS_NAMES,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_IMGSZ,
    DEFAULT_MAX_DETECTIONS,
    PERSON_CLASS_ID,
)
from core.logger import get_logger

log = get_logger("components.postprocessor")


class DetectionPostprocessor:
    """
    Converts raw YOLOv8 ONNX output into structured detection dicts.

    YOLOv8 output shape: ``[1, 5, 8400]``
    where the 5 channels are ``[cx, cy, w, h, confidence]`` (normalized 0-1).

    Steps:
    1. Transpose output to ``[8400, 5]``.
    2. Filter boxes below *conf_threshold*.
    3. Convert centre-format boxes to pixel-space xyxy.
    4. Run NMS.
    5. Build structured detection dicts.
    """

    def __init__(
        self,
        imgsz: int           = DEFAULT_IMGSZ,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float  = DEFAULT_IOU_THRESHOLD,
        max_detections: int   = DEFAULT_MAX_DETECTIONS,
    ) -> None:
        self.imgsz          = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.max_detections = max_detections

    # ── public ────────────────────────────────────────────────────────────────

    def run(self, raw_output: list[np.ndarray]) -> list[dict[str, Any]]:
        """
        Parameters
        ----------
        raw_output : list[np.ndarray]
            Direct output from ``ort.InferenceSession.run()``.

        Returns
        -------
        list[dict]
            Each element has keys: class_id, class_name, confidence, bbox.
        """
        output = raw_output[0][0].T                   # [8400, 5]

        boxes  = output[:, :4]                        # cx cy w h (normalised)
        scores = output[:, 4]                         # objectness

        mask   = scores >= self.conf_threshold
        boxes  = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            log.info("postprocess_ok", n_detections=0)
            return []

        boxes_xyxy = self._cxcywh_to_xyxy(boxes)
        keep       = self._nms(boxes_xyxy, scores)
        keep       = keep[: self.max_detections]

        boxes_xyxy = boxes_xyxy[keep]
        scores     = scores[keep]

        detections = [
            self._build_detection(box, score)
            for box, score in zip(boxes_xyxy, scores)
        ]
        log.info("postprocess_ok", n_detections=len(detections))
        return detections

    # ── private ───────────────────────────────────────────────────────────────

    def _cxcywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - w / 2) * self.imgsz
        y1 = (cy - h / 2) * self.imgsz
        x2 = (cx + w / 2) * self.imgsz
        y2 = (cy + h / 2) * self.imgsz
        return np.stack([x1, y1, x2, y2], axis=1)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        order = scores.argsort()[::-1]
        keep  = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            ious  = self._iou(boxes[i], boxes[order[1:]])
            order = order[1:][ious <= self.iou_threshold]
        return np.array(keep, dtype=np.intp)

    @staticmethod
    def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        xi1 = np.maximum(box[0], boxes[:, 0])
        yi1 = np.maximum(box[1], boxes[:, 1])
        xi2 = np.minimum(box[2], boxes[:, 2])
        yi2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(xi2 - xi1, 0) * np.maximum(yi2 - yi1, 0)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (area1 + area2 - inter + 1e-6)

    def _build_detection(self, box: np.ndarray, score: float) -> dict[str, Any]:
        x1, y1, x2, y2 = map(int, box)
        return {
            "class_id":   PERSON_CLASS_ID,
            "class_name": CLASS_NAMES[PERSON_CLASS_ID],
            "confidence": round(float(score), 4),
            "bbox": {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx":     (x1 + x2) // 2,
                "cy":     (y1 + y2) // 2,
                "width":  x2 - x1,
                "height": y2 - y1,
            },
        }
