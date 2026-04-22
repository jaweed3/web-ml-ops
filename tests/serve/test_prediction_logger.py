"""
Unit tests for app/components/prediction_logger.py.
No FastAPI or ONNX runner needed — tests the logger in isolation.
"""

import json
from pathlib import Path

import pytest

from app.components.prediction_logger import log_prediction


@pytest.fixture()
def log_file(tmp_path) -> Path:
    return tmp_path / "predictions.jsonl"


def test_creates_file_and_appends_record(log_file):
    log_prediction(
        request_id="abc123",
        model_version="5",
        n_detections=2,
        inference_time_ms=9.4,
        log_path=log_file,
    )
    assert log_file.exists()
    record = json.loads(log_file.read_text().strip())
    assert record["request_id"] == "abc123"
    assert record["model_version"] == "5"
    assert record["n_detections"] == 2
    assert record["inference_time_ms"] == 9.4


def test_appends_multiple_records(log_file):
    for i in range(3):
        log_prediction(
            request_id=f"req{i}",
            model_version="5",
            n_detections=i,
            inference_time_ms=float(i),
            log_path=log_file,
        )
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[2])["request_id"] == "req2"


def test_record_has_timestamp(log_file):
    log_prediction(
        request_id="ts_test",
        model_version="1",
        n_detections=0,
        inference_time_ms=1.0,
        log_path=log_file,
    )
    record = json.loads(log_file.read_text().strip())
    assert "ts" in record
    # isoformat() with UTC produces either "Z" or "+00:00" depending on Python version
    assert record["ts"].endswith("Z") or record["ts"].endswith("+00:00")


def test_does_not_raise_on_unwritable_path(tmp_path):
    """Logging failure must never crash the serving path."""
    bad_path = tmp_path / "no_such_dir" / "subdir" / "predictions.jsonl"
    bad_path.parent.mkdir(parents=True)  # parent exists but make it read-only
    bad_path.parent.chmod(0o444)
    try:
        log_prediction(
            request_id="x",
            model_version="1",
            n_detections=0,
            inference_time_ms=1.0,
            log_path=bad_path,
        )
    finally:
        bad_path.parent.chmod(0o755)  # restore so tmp_path cleanup works
