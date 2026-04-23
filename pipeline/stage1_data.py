import hashlib
import random
import subprocess
from pathlib import Path

from PIL import Image

from app.utils.dummy_dataset import generate_dummy_data
from app.utils.dummy_yaml import generate_dummy_yaml
from core.config import is_debug_mode, is_subset_mode, load_config
from core.logger import get_logger

log = get_logger("stage1_data")


def compute_dataset_hash(data_dir: str) -> str:
    """
    Stable MD5 over all label files (sorted by path).
    Labels are small text files — cheap to hash, fully reproducible.
    """
    h = hashlib.md5()
    label_files = sorted(Path(data_dir).glob("labels/**/*.txt"))
    for lbl in label_files:
        h.update(lbl.read_bytes())
    return h.hexdigest()


def write_dataset_hash(data_dir: str) -> str:
    digest = compute_dataset_hash(data_dir)
    out = Path("artifacts/dataset_hash.txt")
    out.parent.mkdir(exist_ok=True)
    out.write_text(digest)
    log.info("dataset_hash_written", extra={"hash": digest, "path": str(out)})
    return digest


def _init_dagshub() -> None:
    """Configure DVC credentials via DagHub SDK (no manual token management)."""
    try:
        import dagshub
        dagshub.init(repo_owner="jaweed3", repo_name="web-ml-ops", mlflow=False)
        log.info("dagshub_init_complete")
    except Exception as e:
        log.warning("dagshub_init_skipped", extra={"reason": str(e)})


def _rewrite_subset_yaml(subset_dir: Path) -> None:
    """Overwrite dataset.yaml with absolute paths for the current machine."""
    yaml_path = subset_dir / "dataset.yaml"
    abs_path = subset_dir.resolve()
    yaml_path.write_text(
        f"path: {abs_path}\ntrain: images/train\nval: images/val\n\nnc: 1\nnames:\n  0: person\n"
    )
    log.info("subset_yaml_rewritten", extra={"path": str(yaml_path)})


def pull_dataset(subset: bool = False) -> None:
    _init_dagshub()
    target = "data/coco_person_subset.dvc" if subset else None
    cmd = ["dvc", "pull"] + (["--force", target] if subset else [])
    mode = "subset" if subset else "full"
    log.info("pulling_dataset", extra={"source": "dagshub_dvc_remote", "mode": mode})
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("dvc_pull_failed", extra={"stderr": result.stderr})
        raise RuntimeError(f"DVC pull failed:\n{result.stderr}")
    if subset:
        _rewrite_subset_yaml(Path("data/coco_person_subset"))
    log.info("pull_complete", extra={"mode": mode})


def _validate_label_file(lbl_path: Path) -> None:
    """
    Validate YOLO format: each non-empty line must be
    ``class_id cx cy w h`` with class_id int ≥ 0 and
    cx, cy, w, h floats in [0, 1].
    """
    for lineno, raw in enumerate(lbl_path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{lbl_path}:{lineno} — expected 5 fields, got {len(parts)}: {line!r}"
            )
        try:
            cls_id = int(parts[0])
            coords = [float(p) for p in parts[1:]]
        except ValueError:
            raise ValueError(
                f"{lbl_path}:{lineno} — non-numeric value: {line!r}"
            )
        if cls_id < 0:
            raise ValueError(f"{lbl_path}:{lineno} — class_id must be ≥ 0, got {cls_id}")
        for i, v in enumerate(coords):
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"{lbl_path}:{lineno} — coordinate[{i}] out of range [0,1]: {v}"
                )


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
            f"Image/label mismatch: {len(train_images)} images, {len(train_labels)} labels"
        )

    yaml_file = "dataset.yaml"
    if Path(yaml_file).exists():
        log.info(f"yaml file exists : {yaml_file}")

    samples = random.sample(train_images, min(10, len(train_images)))
    for img_path in samples:
        try:
            Image.open(img_path).verify()
        except Exception as e:
            raise ValueError(f"Corrupt image detected: {img_path} — {e}") from e

    # Validate YOLO label format for all training labels
    for lbl_path in train_labels:
        _validate_label_file(lbl_path)

    stats = {
        "train_count": len(train_images),
        "val_count": len(val_images),
        "label_count": len(train_labels),
    }
    log.info("your_dataset_is_valid", extra={**stats})
    return stats


def run_stage():
    cfg = load_config()
    subset = is_subset_mode()
    data_dir = cfg.data.subset_dir if subset else cfg.data.dir

    log.info("stage1_start", extra={"mode": "subset" if subset else "full", "data_dir": data_dir})

    if is_debug_mode():
        log.info("debug mode active", extra={"action": "generating dummy dataset"})
        generate_dummy_data(data_dir=data_dir)
        generate_dummy_yaml(output_dir=data_dir)
        validate_dataset(data_dir=data_dir)
    else:
        pull_dataset(subset=subset)
        validate_dataset(data_dir=data_dir)

    write_dataset_hash(data_dir)


def subset_dataset(data_dir: str, max_samples: int) -> None:
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

    log.info("dataset subsetted : ", extra={"max_samples": max_samples})


if __name__ == "__main__":
    run_stage()

    if is_debug_mode():
        log.info("DEBUG_MODE_DETECTED")
