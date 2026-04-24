from pathlib import Path

import yaml


def generate_dummy_yaml(output_dir: str) -> str:
    yaml_content = {
        "path": str(Path(output_dir).resolve()),
        "train": "train_data/images/train",
        "val": "train_data/images/val",
        "test": "test_data/images",
        "nc": 1,
        "names": ["person"],
    }
    yaml_dir = Path(output_dir) / "dataset.yaml"
    with open(yaml_dir, "w") as f:
        yaml.dump(yaml_content, f)
    return str(yaml_dir)
