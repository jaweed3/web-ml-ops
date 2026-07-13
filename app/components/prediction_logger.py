import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from core.logger import get_logger

log = get_logger("components.prediction_logger")

_DEFAULT_PATH = Path("artifacts/predictions.jsonl")
_lock = threading.Lock()


_FEEDBACK_PATH = Path("artifacts/feedback.jsonl")


def log_prediction(
    *,
    request_id: str,
    model_version: str,
    n_detections: int,
    inference_time_ms: float,
    log_path: Path = _DEFAULT_PATH,
) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model_version": model_version,
        "n_detections": n_detections,
        "inference_time_ms": round(inference_time_ms, 3),
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with _lock, log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        log.error("prediction_log_write_failed", error=str(exc))


def log_feedback(
    *,
    request_id: str,
    ground_truth: list[dict],
    annotator: str = "",
    log_path: Path = _FEEDBACK_PATH,
) -> None:
    """Append a ground-truth feedback record to the JSONL feedback log."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "ground_truth": ground_truth,
        "annotator": annotator,
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with _lock, log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        log.error("feedback_log_write_failed", error=str(exc))
