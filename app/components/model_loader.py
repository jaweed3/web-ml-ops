import os
from pathlib import Path

import mlflow

from app.utils.common import ensure_dir
from core.config import CacheConfig, DagsHubConfig, ModelRegistryConfig
from core.logger import get_logger

log = get_logger("components.model_loader")


class ModelLoader:
    """Pulls a registered ONNX model from the MLflow Registry and caches it locally."""

    def __init__(
        self,
        dagshub_cfg: DagsHubConfig,
        model_cfg: ModelRegistryConfig,
        cache_cfg: CacheConfig,
    ) -> None:
        self._dagshub = dagshub_cfg
        self._model = model_cfg
        self._cache = cache_cfg

    def _init_tracking(self) -> None:
        uri = f"https://dagshub.com/{self._dagshub.username}/{self._dagshub.repo}.mlflow"
        mlflow.set_tracking_uri(uri)
        os.environ["MLFLOW_TRACKING_USERNAME"] = self._dagshub.username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self._dagshub.token
        log.info("mlflow_tracking_configured", uri=uri)

    def _resolve_version(self, name: str, requested: str) -> str:
        if requested != "latest":
            return requested
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(name, stages=["Production", "None"])
        if not versions:
            raise RuntimeError(f"No registered versions found for model: {name}")
        resolved = versions[0].version
        log.info("resolved_latest_version", extra={"name": name, "version": resolved})
        return resolved

    def _find_onnx(self, local_dir: str) -> Path:
        candidates = list(Path(local_dir).glob("**/*.onnx"))
        if not candidates:
            raise RuntimeError(f"No .onnx file found in downloaded artifact directory: {local_dir}")
        return candidates[0]

    def _find_cached_onnx(self) -> Path:
        """Return the most recently modified .onnx in the cache dir, or raise."""
        cache_dir = Path(self._cache.model_dir)
        candidates = list(cache_dir.glob("**/*.onnx"))
        if not candidates:
            raise RuntimeError(
                "MLflow registry unreachable and no cached model found in "
                f"{cache_dir} — cannot start server"
            )
        best = max(candidates, key=lambda p: p.stat().st_mtime)
        log.warning("using_cached_model", extra={"path": str(best)})
        return best

    def load(self) -> tuple[Path, str]:
        """Download model from MLflow, falling back to local cache if unreachable."""
        self._init_tracking()
        cache_dir = Path(self._cache.model_dir)
        ensure_dir(cache_dir)

        name = self._model.name
        try:
            version = self._resolve_version(name, self._model.version)
            log.info("downloading_artifact", extra={"name": name, "version": version})
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{name}/{version}",
                dst_path=str(cache_dir),
            )
            onnx_path = self._find_onnx(local_path)
        except Exception as exc:
            log.warning("registry_unreachable_falling_back_to_cache", extra={"error": str(exc)})
            onnx_path = self._find_cached_onnx()
            version = "cached"

        log.info("model_ready", extra={"path": str(onnx_path), "version": version})
        return onnx_path, version
