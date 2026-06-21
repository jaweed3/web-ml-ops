import os
from pathlib import Path

import yaml
from pydantic import BaseModel

# ── Training config ──────────────────────────────────────────────────────────


class DataConfig(BaseModel):
    dir: str
    yaml: str
    subset_dir: str = "data/coco_person_subset"
    subset_yaml: str = "data/coco_person_subset/dataset.yaml"


class ModelConfig(BaseModel):
    name: str


class TrainConfig(BaseModel):
    epochs: int
    imgsz: int
    batch: int
    lr0: float
    optimizer: str
    device: str = "cpu"


class DebugConfig(BaseModel):
    epochs: int
    imgsz: int
    batch: int
    device: str
    max_samples: int


class TrainConfigRoot(BaseModel):
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    debug: DebugConfig


# ── Serving config ───────────────────────────────────────────────────────────


class DagsHubConfig(BaseModel):
    username: str = ""
    repo: str = ""
    token: str = ""


class ModelRegistryConfig(BaseModel):
    name: str = "rescuevision-onnx-int8"
    version: str = "latest"
    format: str = "onnx_int8"


class InferenceConfig(BaseModel):
    imgsz: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
    n_threads: int = 2


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    max_file_size_mb: int = 10


class CacheConfig(BaseModel):
    model_dir: str = "/tmp/rescuevision_model"


class ServeConfig(BaseModel):
    dagshub: DagsHubConfig = DagsHubConfig()
    model: ModelRegistryConfig = ModelRegistryConfig()
    candidate_model: ModelRegistryConfig | None = None
    inference: InferenceConfig = InferenceConfig()
    server: ServerConfig = ServerConfig()
    cache: CacheConfig = CacheConfig()


# ── Helpers ──────────────────────────────────────────────────────────────────


def is_debug_mode() -> bool:
    return os.getenv("DEBUG_MODE", "false").lower() == "true"


def is_subset_mode() -> bool:
    return os.getenv("USE_SUBSET", "false").lower() == "true"


def load_config(path: str = "configs/train_config.yaml") -> TrainConfigRoot:
    raw = yaml.safe_load(Path(path).read_text())
    cfg = TrainConfigRoot(**raw)

    if is_debug_mode():
        cfg.train.epochs = cfg.debug.epochs
        cfg.train.imgsz = cfg.debug.imgsz
        cfg.train.batch = cfg.debug.batch
        cfg.train.device = cfg.debug.device

    return cfg
