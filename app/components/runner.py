import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.constant import ORT_INTRA_THREADS, ORT_PROVIDERS
from app.utils.common import elapsed_ms, new_request_id
from core.logger import get_logger

log = get_logger("components.runner")


class ONNXRunner:
    """
    Wraps an ONNX Runtime InferenceSession as a thread-safe singleton.

    Session initialization is expensive — it happens once when the object is
    constructed (at server startup). Inference calls are protected by a lock
    so multiple concurrent requests don't corrupt session state.

    Attributes
    ----------
    version : str
        Model version string pulled from MLflow registry.
    loaded_at : float | None
        ``time.time()`` timestamp when the session became ready, or None.
    """

    def __init__(self, model_path: Path, version: str, n_threads: int = ORT_INTRA_THREADS) -> None:
        self.version = version
        self.loaded_at: float | None = None
        self._lock = threading.Lock()

        log.info("session_init", path=str(model_path), version=version)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = n_threads

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=ORT_PROVIDERS,
        )
        self._input_name = self._session.get_inputs()[0].name
        self._input_shape = self._session.get_inputs()[0].shape
        self.loaded_at = time.time()

        log.info(
            "session_ready",
            input_name=self._input_name,
            input_shape=self._input_shape,
        )

    # ── public ────────────────────────────────────────────────────────────────

    def run(self, blob: np.ndarray) -> tuple[list[np.ndarray], float, str]:
        """
        Run inference on a preprocessed image blob.

        Parameters
        ----------
        blob : np.ndarray
            Shape ``(1, 3, H, W)``, dtype float32.

        Returns
        -------
        (outputs, latency_ms, request_id) : tuple
        """
        request_id = new_request_id()
        t0 = time.perf_counter()

        with self._lock:
            outputs = self._session.run(None, {self._input_name: blob})

        latency = elapsed_ms(t0)
        log.info("inference_complete", request_id=request_id, latency_ms=latency)
        return outputs, latency, request_id

    @property
    def input_shape(self) -> list:
        return list(self._input_shape)

    @property
    def is_ready(self) -> bool:
        return self.loaded_at is not None


# ── Module-level singleton ────────────────────────────────────────────────────

_runner: ONNXRunner | None = None


def get_runner() -> ONNXRunner:
    if _runner is None:
        raise RuntimeError("ONNXRunner not initialized — call init_runner() at application startup")
    return _runner


def init_runner(model_path: Path, version: str, n_threads: int = ORT_INTRA_THREADS) -> ONNXRunner:
    global _runner
    _runner = ONNXRunner(model_path, version, n_threads)
    return _runner
