from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset slowapi's in-memory counter before every test so tests are independent."""
    from app.dependencies import limiter

    limiter._storage.reset()
    yield
    limiter._storage.reset()


def _make_jpeg(h: int = 480, w: int = 640) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture(scope="session")
def sample_image_bytes() -> bytes:
    """Minimal valid JPEG for upload tests."""
    return _make_jpeg()


@pytest.fixture(scope="session")
def portrait_image_bytes() -> bytes:
    """Tall portrait image to test letterbox padding."""
    return _make_jpeg(h=1080, w=720)


@pytest.fixture(scope="session")
def client():
    """
    FastAPI TestClient with a fully mocked ONNX runner.
    Does NOT contact DagsHub or load any real model weights.
    """
    from app.main import app

    dummy_output = np.zeros((1, 5, 2100), dtype=np.float32)
    # Place one high-confidence box at centre
    dummy_output[0, 0, 0] = 0.5  # cx
    dummy_output[0, 1, 0] = 0.5  # cy
    dummy_output[0, 2, 0] = 0.2  # w
    dummy_output[0, 3, 0] = 0.4  # h
    dummy_output[0, 4, 0] = 0.9  # confidence

    mock_runner = MagicMock()
    mock_runner.is_ready = True
    mock_runner.version = "test-v1"
    mock_runner.loaded_at = 1_700_000_000.0
    mock_runner.input_shape = [1, 3, 320, 320]
    mock_runner.run.return_value = ([dummy_output], 12.3, "req_test001")

    with (
        patch("app.components.runner._runner", mock_runner),
        patch("app.router.health.get_runner", return_value=mock_runner),
        patch("app.router.meta.get_runner", return_value=mock_runner),
        patch("app.pipeline.prediction_pipeline.get_runner", return_value=mock_runner),
    ):
        app.state.pipeline = _make_pipeline()
        app.state.model_name = "rescuevision-onnx-int8"
        app.state.model_format = "onnx_int8"
        yield TestClient(app, raise_server_exceptions=False)


def _make_pipeline():
    from app.entity.config_entity import InferenceConfig
    from app.pipeline.prediction_pipeline import PredictionPipeline

    cfg = InferenceConfig(
        imgsz=320,
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_detections=100,
        n_threads=2,
    )
    return PredictionPipeline(cfg)
