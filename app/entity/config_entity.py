from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DagsHubConfig:
    username: str
    repo: str
    token: str


@dataclass(frozen=True)
class ModelRegistryConfig:
    name: str
    version: str
    format: str


@dataclass(frozen=True)
class InferenceConfig:
    imgsz: int
    conf_threshold: float
    iou_threshold: float
    max_detections: int
    n_threads: int


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    workers: int
    max_file_size_mb: int


@dataclass(frozen=True)
class CacheConfig:
    model_dir: Path
