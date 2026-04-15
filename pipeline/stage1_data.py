import subprocess
import sys
import random
import os
from pathlib import Path
from PIL import Image

from core.logger import get_logger
from core.config import load_config
from app.utils.dataset import generate_dummy_data

log = get_logger("stage1_data")


def pull_dataset() -> None:
    log.info("pulling_dataset", source="dagshub_dvc_remote")
    result = subprocess.run(["dvc", "pull"], capture_output=True, text=True)
    if result.returncode != 0:
        log.error("dvc_pull_failed", stderr=result.stderr)
        raise RuntimeError(f"DVC pull failed:\n{result.stderr}")
    log.info("pull_complete")


def validate_dataset(data_dir: str = "data/") -> dict:
    log.info("dataset_validation_started!")
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
    log.info("your_dataset_is_valid", **stats)
    return stats

def run_stage():
    cfg = load_config()
    is_debug = os.getenv("DEBUG_MODE", "false").lower() == "true"

    if is_debug:
        log.info("debug mode active ", action="generating dummy dataset")
        generate_dummy_data(data_dir=cfg.data.dir)
        validate_dataset(data_dir=cfg.data.dir)
    else:
        pull_dataset()
        validate_dataset(data_dir=cfg.data.dir)

def subset_dataset(data_dir: str, max_samples: int) ->  None:
    path = Path(data_dir)

    for split in ["train", "test"]:
        images = sorted((path / "images" / split).glob("*.jpg"))
        if len(images) <= max_samples:
            continue

        keep = set(img.stem for img in images[:max_samples])

        for img in images:
            if img.stem not in keep:
                img.unlink()

        for lbl in (path / "labels" / split).glob("*.txt"):
            if lbl.stem not in keep:
                lbl.unlink()

    log.info("dataset subsetted : ", max_samples=max_samples)

if __name__ == "__main__":
    run_stage()

    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        log.info("DEBUG_MODE_DETECTED")
