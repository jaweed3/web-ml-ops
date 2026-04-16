from pathlib import Path

# ── Config paths ─────────────────────────────────────────────────────────────
CONFIG_FILE_PATH = Path("configs/serve_config.yaml")

# ── Artifact dirs ─────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path("artifacts")
MODEL_CACHE_DIR = Path("/tmp/rescuevision_model")

# ── Model registry names ──────────────────────────────────────────────────────
MODEL_NAME_FP32 = "rescuevision-onnx-fp32"
MODEL_NAME_INT8 = "rescuevision-onnx-int8"
MODEL_NAME_TFLITE = "rescuevision-tflite-int8"

# ── Inference defaults ────────────────────────────────────────────────────────
DEFAULT_IMGSZ = 640
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_MAX_DETECTIONS = 100
LETTERBOX_PAD_VALUE = 114  # grey padding for letterbox

# ── COCO person class ─────────────────────────────────────────────────────────
PERSON_CLASS_ID = 0
CLASS_NAMES: dict[int, str] = {0: "person"}

# ── Accepted upload types ──────────────────────────────────────────────────────
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# ── ONNX Runtime ──────────────────────────────────────────────────────────────
ORT_PROVIDERS = ["CPUExecutionProvider"]
ORT_INTRA_THREADS = 2
