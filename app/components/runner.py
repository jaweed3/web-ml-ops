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
    Wraps an ONNX Runtime InferenceSession.

    Session initialization is expensive — it happens once when the object is
    constructed (at server startup). Inference calls are thread-safe.
    """

    def __init__(self, model_path: Path, version: str, n_threads: int = ORT_INTRA_THREADS) -> None:
        self.version = version
        self.loaded_at: float | None = None

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

        log.info("session_ready", input_name=self._input_name, input_shape=self._input_shape)

    def run(self, blob: np.ndarray) -> tuple[list[np.ndarray], float, str]:
        request_id = new_request_id()
        t0 = time.perf_counter()
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
