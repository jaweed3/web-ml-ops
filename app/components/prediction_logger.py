"""
Appends one JSON record per inference to artifacts/predictions.jsonl.

Thread-safe: writes are protected by a lock since PredictionPipeline.run()
is called from a thread-pool executor.

Format (one JSON object per line):
    {
        "ts": "2026-04-22T10:00:00.123456Z",
        "request_id": "a1b2c3d4",
        "model_version": "3",
        "n_detections": 2,
        "inference_time_ms": 9.4
    }
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from core.logger import get_logger

log = get_logger("components.prediction_logger")

_DEFAULT_PATH = Path("artifacts/predictions.jsonl")
_lock = threading.Lock()


def log_prediction(
    *,
    request_id: str,
    model_version: str,
    n_detections: int,
    inference_time_ms: float,
    log_path: Path = _DEFAULT_PATH,
) -> None:
    """
    Append a prediction record to the JSONL log file.

    Silently swallows I/O errors so a logging failure never takes down
    the serving path.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model_version": model_version,
        "n_detections": n_detections,
        "inference_time_ms": round(inference_time_ms, 3),
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with log_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        log.error("prediction_log_write_failed", error=str(exc))
