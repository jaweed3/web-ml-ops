import random
from pathlib import Path

from PIL import Image

DATA_ROOT = Path("data/dataset")
TRAIN_DIR = DATA_ROOT / "train_data"
TEST_DIR = DATA_ROOT / "test_data"

# ── Structure ─────────────────────────────────────────────────────────────────


def test_required_dirs_exist():
    required = [
        TRAIN_DIR / "images/train",
        TRAIN_DIR / "images/val",
        TRAIN_DIR / "labels/train",
        TRAIN_DIR / "labels/val",
        TEST_DIR / "images",
        TEST_DIR / "labels",
    ]
    for d in required:
        assert d.exists(), f"Missing: {d}"


def test_train_set_not_empty():
    imgs = list((TRAIN_DIR / "images/train").glob("*.jpg"))
    assert len(imgs) > 0, "No training images found — check DVC pull"


def test_val_set_not_empty():
    imgs = list((TRAIN_DIR / "images/val").glob("*.jpg"))
    assert len(imgs) > 0, "No validation images found — check DVC pull"


def test_test_set_not_empty():
    imgs = list((TEST_DIR / "images").glob("*.jpg"))
    assert len(imgs) > 0, "No test images found — check DVC pull"


# ── Image / label alignment ───────────────────────────────────────────────────


def test_no_image_label_mismatch():
    imgs = sorted((TRAIN_DIR / "images/train").glob("*.jpg"))
    lbls = sorted((TRAIN_DIR / "labels/train").glob("*.txt"))
    assert len(imgs) == len(lbls), f"Mismatch: {len(imgs)} images vs {len(lbls)} labels"


def test_every_image_has_label():
    """Each image stem must have a corresponding label file."""
    img_stems = {p.stem for p in (TRAIN_DIR / "images/train").glob("*.jpg")}
    lbl_stems = {p.stem for p in (TRAIN_DIR / "labels/train").glob("*.txt")}
    missing = img_stems - lbl_stems
    assert not missing, f"Images without labels: {sorted(missing)[:5]}"


# ── Annotation format (YOLO) ──────────────────────────────────────────────────


def test_yolo_annotation_format_valid():
    """
    Every line in every label file must be:
        <class_id> <cx> <cy> <w> <h>
    where all values are floats and cx/cy/w/h are in [0, 1].
    """
    label_dir = TRAIN_DIR / "labels/train"
    errors = []

    for label_path in label_dir.glob("*.txt"):
        for i, line in enumerate(label_path.read_text().strip().splitlines(), start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                errors.append(f"{label_path.name}:{i} — expected 5 fields, got {len(parts)}")
                continue
            try:
                int(parts[0])
                coords = [float(p) for p in parts[1:]]
            except ValueError:
                errors.append(f"{label_path.name}:{i} — non-numeric value")
                continue
            if not all(0.0 <= c <= 1.0 for c in coords):
                errors.append(f"{label_path.name}:{i} — coords out of [0,1]: {coords}")

    assert not errors, "Annotation format errors (first 5):\n" + "\n".join(errors[:5])


def test_only_person_class_present():
    """Dataset should only contain class id 0 (person)."""
    label_dir = TRAIN_DIR / "labels/train"
    non_zero = []

    for label_path in label_dir.glob("*.txt"):
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts and parts[0] != "0":
                non_zero.append(f"{label_path.name}: class {parts[0]}")

    assert not non_zero, "Non-person class ids found (expected 0 only):\n" + "\n".join(non_zero[:5])


# ── Image integrity ───────────────────────────────────────────────────────────


def test_sample_images_not_corrupt():
    """Spot-check up to 20 random training images for corruption via PIL."""
    imgs = list((TRAIN_DIR / "images/train").glob("*.jpg"))
    sample = random.sample(imgs, min(20, len(imgs)))
    corrupt = []

    for path in sample:
        try:
            Image.open(path).verify()
        except Exception as e:
            corrupt.append(f"{path.name}: {e}")

    assert not corrupt, "Corrupt images detected:\n" + "\n".join(corrupt)
