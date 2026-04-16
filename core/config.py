import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    dir: str
    yaml: str


class ModelConfig(BaseModel):
    name: str


class TrainConfig(BaseModel):
    epochs: int
    imgsz: int
    batch: int
    lr0: float
    optimizer: str
    device: str = "cpu"  # override per machine: cuda | mps | cpu


class DebugConfig(BaseModel):
    epochs: int
    imgsz: int
    batch: int
    device: str
    max_samples: int


class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    debug: DebugConfig


def load_config(path: str = "configs/train_config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    cfg = Config(**raw)

    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        cfg.train.epochs = cfg.debug.epochs
        cfg.train.imgsz = cfg.debug.imgsz
        cfg.train.batch = cfg.debug.batch
        cfg.train.device = cfg.debug.device

    return cfg
