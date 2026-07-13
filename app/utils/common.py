import json
import time
import uuid
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def write_json(data: Any, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent))


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text())


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_request_id(prefix: str = "req") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1_000_000, 2)


def assert_file_exists(path: Path, label: str = "file") -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {label} not found: {path}")


def assert_non_empty(value: list, label: str = "list") -> None:
    if not value:
        raise ValueError(f"{label} must not be empty")
