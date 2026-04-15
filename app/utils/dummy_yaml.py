from pathlib import Path
import yaml
from core.logger import get_logger

def generate_dummy_yaml(output_dir: str) -> str:
    
    yaml_content = {
        "path": str(Path(output_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["person"]
    }
    yaml_dir = Path(output_dir) / "dataset.yaml"
    with open(yaml_dir, "w") as f:
        yaml.dump(yaml_content, f)
    return str(yaml_dir)
