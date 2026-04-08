"""
Prometheus metrics for RescueVision serving layer.

All metrics are defined here as module-level singletons so they are shared
across the entire process. Import `METRICS` wherever you need to record values.

Registered metrics
------------------
rescuevision_requests_total          Counter   method, path, status_code
rescuevision_request_duration_seconds Histogram method, path
rescuevision_inference_latency_seconds Histogram (pure ONNX time)
rescuevision_detections_per_request   Histogram
rescuevision_model_info               Info      version, format
rescuevision_startup_timestamp_seconds Gauge
"""

import time

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    REGISTRY,
    CollectorRegistry,
)

# ── Request-level metrics ────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "rescuevision_requests_total",
    "Total HTTP requests processed",
    ["method", "path", "status_code"],
)

REQUEST_DURATION = Histogram(
    "rescuevision_request_duration_seconds",
    "End-to-end HTTP request duration (including pre/post-processing)",
    ["method", "path"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# ── Inference-level metrics ───────────────────────────────────────────────────

INFERENCE_LATENCY = Histogram(
    "rescuevision_inference_latency_seconds",
    "Pure ONNX Runtime inference time (excludes HTTP, pre/post-processing)",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

DETECTIONS_PER_REQUEST = Histogram(
    "rescuevision_detections_per_request",
    "Number of person detections returned per prediction request",
    buckets=[0, 1, 2, 3, 5, 10, 20, 50],
)

# ── Model metadata ────────────────────────────────────────────────────────────

MODEL_INFO = Info(
    "rescuevision_model",
    "Static metadata about the currently loaded model",
)

STARTUP_TIMESTAMP = Gauge(
    "rescuevision_startup_timestamp_seconds",
    "Unix timestamp when the inference server became ready",
)


# ── Helper ────────────────────────────────────────────────────────────────────

def record_model_ready(version: str, fmt: str) -> None:
    """Call once at startup after the model session is initialized."""
    MODEL_INFO.info({"version": version, "format": fmt})
    STARTUP_TIMESTAMP.set(time.time())


def record_inference(latency_ms: float, n_detections: int) -> None:
    """Call from PredictionPipeline after each successful inference."""
    INFERENCE_LATENCY.observe(latency_ms / 1000)
    DETECTIONS_PER_REQUEST.observe(n_detections)
