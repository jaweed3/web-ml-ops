import random
from pathlib import Path

import numpy as np
from PIL import Image


def generate_dummy_data(data_dir: str, num_samples: int = 10, img_size: int = 640):
    path = Path(data_dir)

    for split in ["train", "val"]:
        img_path = path / "train_data" / "images" / split
        lbl_path = path / "train_data" / "labels" / split
        img_path.mkdir(parents=True, exist_ok=True)
        lbl_path.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            file_name = f"dummy_{split}_{i}"

            img_array = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(img_path / f"{file_name}.jpg")

            with open(lbl_path / f"{file_name}.txt", "w") as f:
                num_objects = random.randint(1, 3)
                for _ in range(num_objects):
                    cls = 0  # class 0 = person
                    x, y = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
                    w, h = random.uniform(0.1, 0.4), random.uniform(0.1, 0.4)
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    # test_data: flat structure (no train/val split)
    test_img_path = path / "test_data" / "images"
    test_lbl_path = path / "test_data" / "labels"
    test_img_path.mkdir(parents=True, exist_ok=True)
    test_lbl_path.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        file_name = f"dummy_test_{i}"
        img_array = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(test_img_path / f"{file_name}.jpg")
        with open(test_lbl_path / f"{file_name}.txt", "w") as f:
            x, y = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
            w, h = random.uniform(0.1, 0.4), random.uniform(0.1, 0.4)
            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print(f"Successfully generated {num_samples} dummy samples in {data_dir}")
