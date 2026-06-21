from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.config import InferenceConfig


@pytest.fixture(autouse=True)
def reset_rate_limiter():
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
    return _make_jpeg()


@pytest.fixture(scope="session")
def portrait_image_bytes() -> bytes:
    return _make_jpeg(h=1080, w=720)


@pytest.fixture(scope="session")
def client():
    from app.main import app

    dummy_output = np.zeros((1, 5, 2100), dtype=np.float32)
    dummy_output[0, 0, 0] = 0.5
    dummy_output[0, 1, 0] = 0.5
    dummy_output[0, 2, 0] = 0.2
    dummy_output[0, 3, 0] = 0.4
    dummy_output[0, 4, 0] = 0.9

    mock_runner = MagicMock()
    mock_runner.is_ready = True
    mock_runner.version = "test-v1"
    mock_runner.loaded_at = 1_700_000_000.0
    mock_runner.input_shape = [1, 3, 320, 320]
    mock_runner.run.return_value = ([dummy_output], 12.3, "req_test001")

    cfg = InferenceConfig(
        imgsz=320, conf_threshold=0.25, iou_threshold=0.45,
        max_detections=100, n_threads=2,
    )
    pipeline = _make_pipeline(mock_runner, cfg)

    app.state.runner = mock_runner
    app.state.pipeline = pipeline
    app.state.model_name = "rescuevision-onnx-int8"
    app.state.model_format = "onnx_int8"

    yield TestClient(app, raise_server_exceptions=False)


def _make_pipeline(runner, cfg):
    from app.pipeline.prediction_pipeline import PredictionPipeline

    return PredictionPipeline(runner, cfg)
