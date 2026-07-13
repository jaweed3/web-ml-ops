from pathlib import Path

CONFIG_FILE_PATH = Path("configs/serve_config.yaml")

EXPERIMENT_NAME = "rescuevision-yolov8n"

ARTIFACTS_DIR = Path("artifacts")
MODEL_CACHE_DIR = Path("/tmp/rescuevision_model")

MODEL_NAME_FP32 = "rescuevision-onnx-fp32"
MODEL_NAME_INT8 = "rescuevision-onnx-int8"
MODEL_NAME_TFLITE = "rescuevision-tflite-int8"

DEFAULT_IMGSZ = 640
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_MAX_DETECTIONS = 100
LETTERBOX_PAD_VALUE = 114

PERSON_CLASS_ID = 0
CLASS_NAMES: dict[int, str] = {0: "person"}

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

ORT_PROVIDERS = ["CPUExecutionProvider"]
ORT_INTRA_THREADS = 2
