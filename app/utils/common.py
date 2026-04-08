import json
import time
import uuid
from pathlib import Path
from typing import Any

import yaml


# ── File I/O ──────────────────────────────────────────────────────────────────

def read_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def write_json(data: Any, path: Path, indent: int = 2) -> None:
    """Serialize *data* to JSON and write to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent))


def read_json(path: Path) -> Any:
    """Load JSON from *path*."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text())


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── ID helpers ────────────────────────────────────────────────────────────────

def new_request_id(prefix: str = "req") -> str:
    """Generate a short unique request ID, e.g. ``req_a3f92b1c``."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ── Timing ────────────────────────────────────────────────────────────────────

def elapsed_ms(start: float) -> float:
    """Return milliseconds elapsed since *start* (from ``time.perf_counter()``)."""
    return round((time.perf_counter() - start) * 1000, 2)


# ── File size ────────────────────────────────────────────────────────────────

def file_size_mb(path: Path) -> float:
    """Return file size in MB, rounded to 2 decimal places."""
    return round(path.stat().st_size / 1_000_000, 2)


# ── Validation helpers ────────────────────────────────────────────────────────

def assert_file_exists(path: Path, label: str = "file") -> None:
    """Raise FileNotFoundError with a clear message if *path* is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Expected {label} not found: {path}")


def assert_non_empty(value: list, label: str = "list") -> None:
    """Raise ValueError if *value* is empty."""
    if not value:
        raise ValueError(f"{label} must not be empty")
