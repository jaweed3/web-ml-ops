from pathlib import Path


def test_required_dirs_exist():
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        assert Path(f"data/coco_person/{d}").exists(), f"Missing: data/coco_person/{d}"


def test_no_image_label_mismatch():
    imgs = list(Path("data/coco_person/images/train").glob("*.jpg"))
    lbls = list(Path("data/coco_person/labels/train").glob("*.txt"))
    assert len(imgs) == len(lbls), (
        f"Mismatch: {len(imgs)} images vs {len(lbls)} labels"
    )


def test_train_set_not_empty():
    imgs = list(Path("data/coco_person/images/train").glob("*.jpg"))
    assert len(imgs) > 0, "No training images found — check DVC pull"
