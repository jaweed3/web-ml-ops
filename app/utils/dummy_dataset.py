import numpy as np
from PIL import Image
from pathlib import Path
import random

def generate_dummy_data(data_dir: str, num_samples: int = 10, img_size: int = 640):
    path = Path(data_dir)
    
    for split in ["train", "val"]:
        img_path = path / "images" / split
        lbl_path = path / "labels" / split
        img_path.mkdir(parents=True, exist_ok=True)
        lbl_path.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            file_name = f"dummy_{split}_{i}"
            
            img_array = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(img_path / f"{file_name}.jpg")

            with open(lbl_path / f"{file_name}.txt", "w") as f:
                num_objects = random.randint(1, 3) # Berpura-pura ada 1-3 objek
                for _ in range(num_objects):
                    cls = 0  # assume class 0 is person.
                    x, y = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
                    w, h = random.uniform(0.1, 0.4), random.uniform(0.1, 0.4)
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    
    print(f"Successfully generated {num_samples} dummy samples in {data_dir}")

# example usage
# generate_dummy_data("data/coco_person", num_samples=5)
