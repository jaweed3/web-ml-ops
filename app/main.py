import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.components.metrics import record_model_ready
from app.components.model_loader import ModelLoader
from app.components.runner import ONNXRunner
from app.config.configuration import ConfigurationManager
from app.dependencies import limiter
from app.middleware.request_logger import RequestLoggerMiddleware
from app.monitoring.drift import DriftDetector
from app.pipeline.prediction_pipeline import PredictionPipeline
from app.router import feedback, health, meta, metrics, predict
from core.logger import get_logger

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    cm = ConfigurationManager()

    dagshub_cfg = cm.get_dagshub_config()
    model_cfg = cm.get_model_registry_config()
    inf_cfg = cm.get_inference_config()
    cache_cfg = cm.get_cache_config()

    log.info("server_starting", model_name=model_cfg.name, version=model_cfg.version)

    try:
        loader = ModelLoader(dagshub_cfg, model_cfg, cache_cfg)
        model_path, version = loader.load()
        runner = ONNXRunner(model_path, version, n_threads=inf_cfg.n_threads)
    except Exception as exc:
        log.error("startup_failed", error=str(exc))
        raise

    # Optional shadow (candidate) runner — same model registry, different version
    shadow_runner: ONNXRunner | None = None
    candidate_cfg = cm.get_candidate_model_config()
    if candidate_cfg is not None:
        try:
            c_loader = ModelLoader(dagshub_cfg, candidate_cfg, cache_cfg)
            c_path, c_version = c_loader.load()
            shadow_runner = ONNXRunner(c_path, c_version, n_threads=inf_cfg.n_threads)
            log.info("shadow_runner_loaded", version=c_version)
        except Exception as exc:
            log.warning("shadow_runner_skipped", error=str(exc))

    drift = DriftDetector()
    pipeline = PredictionPipeline(
        runner, inf_cfg, drift_detector=drift, shadow_runner=shadow_runner,
    )

    # Warm up: one dummy inference so the first real request is not cold
    try:
        dummy = np.zeros((1, 3, inf_cfg.imgsz, inf_cfg.imgsz), dtype=np.float32)
        t0 = time.perf_counter()
        runner.run(dummy)
        warmup_ms = (time.perf_counter() - t0) * 1000
        log.info("warmup_complete", warmup_ms=round(warmup_ms, 1))
    except Exception as exc:
        log.warning("warmup_skipped", error=str(exc))

    # Stash shared state for routers
    app.state.runner = runner
    app.state.pipeline = pipeline
    app.state.model_name = model_cfg.name
    app.state.model_format = model_cfg.format

    # Register model metadata in Prometheus
    record_model_ready(version=version, fmt=model_cfg.format)

    log.info("server_ready", version=version)
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("server_shutdown")


app = FastAPI(
    title="RescueVision Inference API",
    description="Edge-optimized victim detection for SAR drone footage.",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(RequestLoggerMiddleware)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(meta.router)
app.include_router(metrics.router)
app.include_router(feedback.router)
