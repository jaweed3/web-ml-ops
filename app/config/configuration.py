import os
from pathlib import Path

import yaml

from app.constant import CONFIG_FILE_PATH
from core.config import (
    DagsHubConfig,
    ServeConfig,
)


def _env_overlay(cfg: DagsHubConfig) -> DagsHubConfig:
    return DagsHubConfig(
        username=os.environ.get("DAGSHUB_USERNAME", cfg.username),
        repo=os.environ.get("DAGSHUB_REPO", cfg.repo),
        token=os.environ.get("DAGSHUB_TOKEN", cfg.token),
    )


def load_serve_config(config_path: Path = CONFIG_FILE_PATH) -> ServeConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Serve config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text())
    cfg = ServeConfig(**raw)
    cfg.dagshub = _env_overlay(cfg.dagshub)
    return cfg
