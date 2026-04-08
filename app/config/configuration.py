import os
from pathlib import Path
import yaml

from app.constant import CONFIG_FILE_PATH, MODEL_CACHE_DIR
from app.entity.config_entity import (
    CacheConfig,
    DagsHubConfig,
    InferenceConfig,
    ModelRegistryConfig,
    ServerConfig,
)


class ConfigurationManager:
    """
    Single entry point for all serving configuration.

    Loads configs/serve_config.yaml, then overlays any environment
    variables so CI/CD secrets take precedence over the yaml file.

    Usage:
        cm  = ConfigurationManager()
        inf = cm.get_inference_config()
        mdl = cm.get_model_registry_config()
    """

    def __init__(self, config_path: Path = CONFIG_FILE_PATH) -> None:
        self._raw = self._load(config_path)

    # ── private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Serve config not found: {path}")
        return yaml.safe_load(path.read_text())

    def _dagshub_section(self) -> dict:
        cfg = self._raw.get("dagshub", {})
        return {
            "username": os.environ.get("DAGSHUB_USERNAME", cfg.get("username", "")),
            "repo":     os.environ.get("DAGSHUB_REPO",     cfg.get("repo",     "")),
            "token":    os.environ.get("DAGSHUB_TOKEN",    cfg.get("token",    "")),
        }

    # ── public getters ─────────────────────────────────────────────────────────

    def get_dagshub_config(self) -> DagsHubConfig:
        d = self._dagshub_section()
        return DagsHubConfig(**d)

    def get_model_registry_config(self) -> ModelRegistryConfig:
        m = self._raw.get("model", {})
        return ModelRegistryConfig(
            name    = m.get("name",    "rescuevision-onnx-int8"),
            version = m.get("version", "latest"),
            format  = m.get("format",  "onnx_int8"),
        )

    def get_inference_config(self) -> InferenceConfig:
        i = self._raw.get("inference", {})
        return InferenceConfig(
            imgsz          = int(i.get("imgsz",          640)),
            conf_threshold = float(i.get("conf_threshold", 0.25)),
            iou_threshold  = float(i.get("iou_threshold",  0.45)),
            max_detections = int(i.get("max_detections",  100)),
            n_threads      = int(i.get("n_threads",       2)),
        )

    def get_server_config(self) -> ServerConfig:
        s = self._raw.get("server", {})
        return ServerConfig(
            host             = s.get("host",             "0.0.0.0"),
            port             = int(s.get("port",         8080)),
            workers          = int(s.get("workers",      1)),
            max_file_size_mb = int(s.get("max_file_size_mb", 10)),
        )

    def get_cache_config(self) -> CacheConfig:
        c = self._raw.get("cache", {})
        return CacheConfig(
            model_dir=Path(c.get("model_dir", str(MODEL_CACHE_DIR))),
        )
