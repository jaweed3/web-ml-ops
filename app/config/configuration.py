import os
from pathlib import Path

import yaml

from app.constant import CONFIG_FILE_PATH
from core.config import (
    CacheConfig,
    DagsHubConfig,
    InferenceConfig,
    ModelRegistryConfig,
    ServeConfig,
    ServerConfig,
)


class ConfigurationManager:
    """
    Loads serve_config.yaml and overlays environment variables so CI/CD
    secrets (DAGSHUB_TOKEN, etc.) take precedence over the yaml file.
    """

    def __init__(self, config_path: Path = CONFIG_FILE_PATH) -> None:
        raw = self._load(config_path)
        self._cfg = ServeConfig(**raw)
        self._cfg.dagshub = self._dagshub_with_env_overlay(self._cfg.dagshub)

    @staticmethod
    def _load(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Serve config not found: {path}")
        return yaml.safe_load(path.read_text())

    @staticmethod
    def _dagshub_with_env_overlay(cfg: DagsHubConfig) -> DagsHubConfig:
        return DagsHubConfig(
            username=os.environ.get("DAGSHUB_USERNAME", cfg.username),
            repo=os.environ.get("DAGSHUB_REPO", cfg.repo),
            token=os.environ.get("DAGSHUB_TOKEN", cfg.token),
        )

    def get_dagshub_config(self) -> DagsHubConfig:
        return self._cfg.dagshub

    def get_model_registry_config(self) -> ModelRegistryConfig:
        return self._cfg.model

    def get_inference_config(self) -> InferenceConfig:
        return self._cfg.inference

    def get_server_config(self) -> ServerConfig:
        return self._cfg.server

    def get_cache_config(self) -> CacheConfig:
        return self._cfg.cache
