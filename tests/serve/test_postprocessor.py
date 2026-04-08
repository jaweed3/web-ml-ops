import numpy as np
import pytest

from app.components.postprocessor import DetectionPostprocessor


def make_output(n_boxes: int = 1, conf: float = 0.9) -> list[np.ndarray]:
    """Simulate YOLOv8 output [1, 5, 8400]."""
    out = np.zeros((1, 5, 8400), dtype=np.float32)
    for i in range(n_boxes):
        out[0, 0, i] = 0.5         # cx
        out[0, 1, i] = 0.5         # cy
        out[0, 2, i] = 0.2         # w
        out[0, 3, i] = 0.4         # h
        out[0, 4, i] = conf        # confidence
    return [out]


PP = DetectionPostprocessor()


# ── Basic thresholding ────────────────────────────────────────────────────────

def test_detections_returned_above_threshold():
    dets = PP.run(make_output(n_boxes=3, conf=0.9))
    assert len(dets) > 0


def test_no_detections_below_threshold():
    dets = PP.run(make_output(n_boxes=5, conf=0.1))
    assert len(dets) == 0


def test_empty_output_returns_empty_list():
    dets = PP.run(make_output(n_boxes=0, conf=0.9))
    assert dets == []


# ── Detection schema ──────────────────────────────────────────────────────────

def test_detection_has_required_keys():
    dets = PP.run(make_output(n_boxes=1, conf=0.9))
    assert len(dets) == 1
    d = dets[0]
    for key in ("class_id", "class_name", "confidence", "bbox"):
        assert key in d, f"Missing key: {key}"


def test_bbox_has_required_fields():
    dets = PP.run(make_output(n_boxes=1, conf=0.9))
    bbox = dets[0]["bbox"]
    for field in ("x1", "y1", "x2", "y2", "cx", "cy", "width", "height"):
        assert field in bbox, f"Missing bbox field: {field}"


def test_class_name_is_person():
    dets = PP.run(make_output(n_boxes=1, conf=0.9))
    assert dets[0]["class_name"] == "person"


def test_class_id_is_zero():
    dets = PP.run(make_output(n_boxes=1, conf=0.9))
    assert dets[0]["class_id"] == 0


# ── Confidence precision ───────────────────────────────────────────────────────

def test_confidence_rounded_to_4_decimal_places():
    dets = PP.run(make_output(n_boxes=1, conf=0.876543))
    if dets:
        assert len(str(dets[0]["confidence"]).split(".")[-1]) <= 4


# ── max_detections cap ────────────────────────────────────────────────────────

def test_max_detections_cap():
    pp  = DetectionPostprocessor(max_detections=2)
    out = np.zeros((1, 5, 8400), dtype=np.float32)
    for i in range(10):
        out[0, 0, i] = 0.1 * (i + 1)   # spread so NMS keeps all
        out[0, 1, i] = 0.1 * (i + 1)
        out[0, 2, i] = 0.05
        out[0, 3, i] = 0.05
        out[0, 4, i] = 0.9
    dets = pp.run([out])
    assert len(dets) <= 2
