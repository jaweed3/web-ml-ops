from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.components.metrics import record_model_ready
from app.components.model_loader import ModelLoader
from app.components.runner import init_runner
from app.config.configuration import ConfigurationManager
from app.entity.config_entity import InferenceConfig
from app.middleware.request_logger import RequestLoggerMiddleware
from app.pipeline.prediction_pipeline import PredictionPipeline
from app.router import health, meta, metrics, predict
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
        model_path, version, imgsz = loader.load()
        int_cfg = InferenceConfig(
            imgsz=imgsz,
            conf_threshold=inf_cfg.conf_threshold,
            iou_threshold=inf_cfg.iou_threshold,
            max_detections=inf_cfg.max_detections,
            n_threads=inf_cfg.n_threads
        )
        init_runner(model_path, version, n_threads=inf_cfg.n_threads)
    except Exception as exc:
        log.error("startup_failed", error=str(exc))
        raise

    # Stash shared state for routers
    app.state.pipeline = PredictionPipeline(int_cfg)
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

app.add_middleware(RequestLoggerMiddleware)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(meta.router)
app.include_router(metrics.router)
