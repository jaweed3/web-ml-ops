from pathlib import Path

import cv2
import numpy as np

from app.constant import ARTIFACTS_DIR
from app.utils.common import read_json
from core.logger import get_logger

log = get_logger("monitoring.drift")

_BASELINE_FILE = ARTIFACTS_DIR / "image_baseline.json"


def compute_image_stats(img: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist))) if hist.size > 0 else 0.0
    return {
        "brightness": float(img.mean()),
        "contrast": float(img.std()),
        "entropy": entropy,
    }


def compute_baseline(image_dir: Path) -> dict[str, float]:
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths = []
    for ext in exts:
        paths.extend(image_dir.rglob(ext))
    if not paths:
        log.warning("no_images_for_baseline", dir=str(image_dir))
        return {}

    all_stats = {k: [] for k in ("brightness", "contrast", "entropy")}
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        stats = compute_image_stats(img)
        for k in all_stats:
            all_stats[k].append(stats[k])

    baseline = {}
    for k, vals in all_stats.items():
        arr = np.array(vals)
        baseline[f"mean_{k}"] = float(arr.mean())
        baseline[f"std_{k}"] = float(arr.std()) + 1e-8
    return baseline


class DriftDetector:
    def __init__(self, baseline_path: Path = _BASELINE_FILE) -> None:
        self._baseline = self._load(baseline_path)

    @staticmethod
    def _load(path: Path) -> dict:
        if path.exists():
            data = read_json(path)
            keys = str(list(data.keys()))
            log.info("drift_baseline_loaded", path=str(path), stats=keys)
            return data
        log.warning(
            "drift_baseline_not_found",
            path=str(path),
            hint="run scripts/compute_baseline.py",
        )
        return {}

    def score(self, stats: dict[str, float]) -> float:
        if not self._baseline:
            return 0.0
        scores = []
        for key in stats:
            mean_key = f"mean_{key}"
            std_key = f"std_{key}"
            if mean_key not in self._baseline or std_key not in self._baseline:
                continue
            z = abs(stats[key] - self._baseline[mean_key]) / self._baseline[std_key]
            scores.append(z)
        return float(np.mean(scores)) if scores else 0.0

    @property
    def is_active(self) -> bool:
        return bool(self._baseline)
