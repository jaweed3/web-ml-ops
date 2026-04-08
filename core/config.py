from pydantic import BaseModel
from pathlib import Path
import yaml


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
    device: str = "cpu"   # override per machine: cuda | mps | cpu


class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def load_config(path: str = "configs/train_config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(**raw)
