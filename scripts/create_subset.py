"""
Buat subset kecil dari full dataset lalu track dengan DVC.

Jalankan di PC Lab setelah full dataset tersedia:

    python scripts/create_subset.py --n_train 300 --n_val 80
    dvc push train_data_subset.dvc

Subset akan disimpan di data/coco_person_subset/ dengan struktur yang sama
dengan data/coco_person/ sehingga kompatibel dengan pipeline yang ada.

Args:
    --src       : direktori full dataset (default: data/coco_person)
    --dst       : direktori output subset (default: data/coco_person_subset)
    --n_train   : jumlah image train (default: 300)
    --n_val     : jumlah image val (default: 80)
    --seed      : random seed untuk reproducibility (default: 42)
"""

import argparse
import random
import shutil
import subprocess
from pathlib import Path

from core.logger import get_logger

log = get_logger("create_subset")


def copy_split(src_dir: Path, dst_dir: Path, split: str, n: int, seed: int) -> int:
    src_images = sorted((src_dir / "images" / split).glob("*.jpg"))
    if not src_images:
        raise FileNotFoundError(f"No images found in {src_dir / 'images' / split}")

    random.seed(seed)
    selected = random.sample(src_images, min(n, len(src_images)))
    selected_stems = {img.stem for img in selected}

    # Copy images
    dst_img_dir = dst_dir / "images" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    for img in selected:
        shutil.copy2(img, dst_img_dir / img.name)

    # Copy matching labels
    dst_lbl_dir = dst_dir / "labels" / split
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    copied_labels = 0
    for lbl in (src_dir / "labels" / split).glob("*.txt"):
        if lbl.stem in selected_stems:
            shutil.copy2(lbl, dst_lbl_dir / lbl.name)
            copied_labels += 1

    log.info(
        f"split_{split}_copied",
        extra={"images": len(selected), "labels": copied_labels},
    )
    return len(selected)


def write_dataset_yaml(dst_dir: Path) -> None:
    """Generate dataset.yaml pointing to the subset directory."""
    abs_path = dst_dir.resolve()
    yaml_content = f"""path: {abs_path}
train: images/train
val: images/val

nc: 1
names:
  0: person
"""
    (dst_dir / "dataset.yaml").write_text(yaml_content)
    log.info("dataset_yaml_written", extra={"path": str(dst_dir / "dataset.yaml")})


def dvc_add(path: Path) -> None:
    result = subprocess.run(["dvc", "add", str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"dvc add failed:\n{result.stderr}")
    log.info("dvc_add_complete", extra={"path": str(path)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset subset for Mac Mini training")
    parser.add_argument("--src", default="data/coco_person", help="Full dataset directory")
    parser.add_argument("--dst", default="data/coco_person_subset", help="Subset output directory")
    parser.add_argument("--n_train", type=int, default=300, help="Number of training images")
    parser.add_argument("--n_val", type=int, default=80, help="Number of validation images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}. Run `dvc pull` first.")

    if dst.exists():
        print(f"[!] {dst} already exists. Delete it first if you want to recreate.")
        raise SystemExit(1)

    log.info(
        "creating_subset",
        extra={
            "src": str(src),
            "dst": str(dst),
            "n_train": args.n_train,
            "n_val": args.n_val,
            "seed": args.seed,
        },
    )

    n_train = copy_split(src, dst, "train", args.n_train, args.seed)
    n_val = copy_split(src, dst, "val", args.n_val, args.seed)
    write_dataset_yaml(dst)

    log.info("subset_created", extra={"train": n_train, "val": n_val, "dst": str(dst)})

    # Auto DVC track
    dvc_add(dst)

    print(f"""
✅ Subset created: {n_train} train, {n_val} val → {dst}/

Next steps (PC Lab):
    git add data/coco_person_subset.dvc .gitignore
    git commit -m "feat: add coco_person_subset ({n_train} train, {n_val} val)"
    dvc push data/coco_person_subset.dvc

Then on Mac Mini:
    git pull
    USE_SUBSET=true make debug   # atau make all
""")
