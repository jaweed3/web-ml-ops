import os
from pathlib import Path

import mlflow
import onnxruntime as ort

from app.entity.config_entity import CacheConfig, DagsHubConfig, ModelRegistryConfig
from app.utils.common import ensure_dir
from core.logger import get_logger

log = get_logger("components.model_loader")


class ModelLoader:
    """
    Pulls a registered ONNX model artifact from the MLflow Model Registry
    (hosted on DagsHub) and caches it locally.

    Responsibilities:
    - Authenticate to DagsHub MLflow tracking server.
    - Resolve "latest" version to a concrete version number.
    - Download the artifact to a local cache directory.
    - Return the local .onnx file path + resolved version string.

    The loader runs ONCE at server startup, never per request.
    """

    def __init__(
        self,
        dagshub_cfg: DagsHubConfig,
        model_cfg: ModelRegistryConfig,
        cache_cfg: CacheConfig,
    ) -> None:
        self._dagshub = dagshub_cfg
        self._model = model_cfg
        self._cache = cache_cfg

    # ── private ────────────────────────────────────────────────────────────────

    def _init_tracking(self) -> None:
        uri = f"https://dagshub.com/{self._dagshub.username}/{self._dagshub.repo}.mlflow"
        mlflow.set_tracking_uri(uri)
        os.environ["MLFLOW_TRACKING_USERNAME"] = self._dagshub.username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self._dagshub.token
        log.info("mlflow_tracking_configured", extra=uri)

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

    # ── public ────────────────────────────────────────────────────────────────

    def load(self) -> tuple[Path, str, int]:
        """
        Returns
        -------
        (model_path, version) : tuple[Path, str]
            Local path to the .onnx file and the resolved version string.
        """
        self._init_tracking()
        ensure_dir(self._cache.model_dir)

        name = self._model.name
        version = self._resolve_version(name, self._model.version)

        log.info("downloading_artifact", extra={"name": name, "version": version})
        model_uri = f"models:/{name}/{version}"
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=str(self._cache.model_dir),
        )

        onnx_path = self._find_onnx(local_path)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        imgsz = sess.get_inputs()[0].shape[2]
        log.info(
            "artifact_cached", extra={"path": str(onnx_path), "version": version, "imgsz": imgsz}
        )
        return onnx_path, version, imgsz
