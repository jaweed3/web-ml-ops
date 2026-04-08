import subprocess
import random
from pathlib import Path
from PIL import Image

from core.logger import get_logger
from core.config import load_config

log = get_logger("stage1_data")


def pull_dataset() -> None:
    log.info("pulling_dataset", source="dagshub_dvc_remote")
    result = subprocess.run(["dvc", "pull"], capture_output=True, text=True)
    if result.returncode != 0:
        log.error("dvc_pull_failed", stderr=result.stderr)
        raise RuntimeError(f"DVC pull failed:\n{result.stderr}")
    log.info("pull_complete")


def validate_dataset(data_dir: str = "data/coco_person") -> dict:
    path = Path(data_dir)
    required_dirs = [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ]
    for d in required_dirs:
        if not (path / d).exists():
            raise FileNotFoundError(f"Missing required directory: {path / d}")

    train_images = sorted((path / "images/train").glob("*.jpg"))
    train_labels = sorted((path / "labels/train").glob("*.txt"))
    val_images = sorted((path / "images/val").glob("*.jpg"))

    if len(train_images) == 0:
        raise ValueError("No training images found — check DVC pull")

    if len(train_images) != len(train_labels):
        raise ValueError(
            f"Image/label mismatch: {len(train_images)} images, "
            f"{len(train_labels)} labels"
        )

    samples = random.sample(train_images, min(10, len(train_images)))
    for img_path in samples:
        try:
            Image.open(img_path).verify()
        except Exception as e:
            raise ValueError(f"Corrupt image detected: {img_path} — {e}")

    stats = {
        "train_count": len(train_images),
        "val_count": len(val_images),
        "label_count": len(train_labels),
    }
    log.info("dataset_valid", **stats)
    return stats


if __name__ == "__main__":
    cfg = load_config()
    pull_dataset()
    validate_dataset(cfg.data.dir)
