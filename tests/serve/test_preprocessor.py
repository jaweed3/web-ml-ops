import numpy as np
import pytest

from app.components.preprocessor import ImagePreprocessor


def make_jpeg(h: int = 480, w: int = 640) -> bytes:
    import cv2
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ── Shape & dtype ──────────────────────────────────────────────────────────────

def test_output_shape_square_input():
    blob = ImagePreprocessor(imgsz=640).run(make_jpeg(640, 640))
    assert blob.shape == (1, 3, 640, 640)


def test_output_shape_landscape_input():
    blob = ImagePreprocessor(imgsz=640).run(make_jpeg(480, 640))
    assert blob.shape == (1, 3, 640, 640)


def test_output_shape_portrait_input():
    blob = ImagePreprocessor(imgsz=640).run(make_jpeg(1080, 720))
    assert blob.shape == (1, 3, 640, 640)


def test_output_dtype_is_float32():
    blob = ImagePreprocessor().run(make_jpeg())
    assert blob.dtype == np.float32


# ── Normalization ──────────────────────────────────────────────────────────────

def test_values_normalized_between_0_and_1():
    blob = ImagePreprocessor().run(make_jpeg())
    assert blob.min() >= 0.0
    assert blob.max() <= 1.0


# ── Error handling ────────────────────────────────────────────────────────────

def test_corrupt_bytes_raises_value_error():
    with pytest.raises(ValueError, match="Could not decode"):
        ImagePreprocessor().run(b"this is not an image at all")


def test_empty_bytes_raises_value_error():
    with pytest.raises(ValueError, match="Could not decode"):
        ImagePreprocessor().run(b"")


# ── Custom imgsz ──────────────────────────────────────────────────────────────

def test_custom_imgsz_320():
    blob = ImagePreprocessor(imgsz=320).run(make_jpeg())
    assert blob.shape == (1, 3, 320, 320)
