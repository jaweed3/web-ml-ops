import cv2
import numpy as np

from app.constant import DEFAULT_IMGSZ, LETTERBOX_PAD_VALUE
from app.monitoring.drift import compute_image_stats
from core.logger import get_logger

log = get_logger("components.preprocessor")


class ImagePreprocessor:
    def __init__(self, imgsz: int = DEFAULT_IMGSZ) -> None:
        self.imgsz = imgsz

    def run(self, image_bytes: bytes) -> tuple[np.ndarray, dict[str, float]]:
        img = self._decode(image_bytes)
        stats = compute_image_stats(img)
        img = self._letterbox(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        blob = np.ascontiguousarray(img)
        log.info("preprocess_ok", shape=list(blob.shape), dtype=str(blob.dtype))
        return blob, stats

    def _decode(self, data: bytes) -> np.ndarray:
        if not data:
            raise ValueError("Could not decode image — unsupported format or corrupt file")
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image — unsupported format or corrupt file")
        return img

    def _letterbox(self, img: np.ndarray) -> np.ndarray:
        """Resize with grey padding to preserve aspect ratio."""
        h, w = img.shape[:2]
        scale = self.imgsz / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = self.imgsz - new_h
        pad_w = self.imgsz - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=LETTERBOX_PAD_VALUE,
        )
        return img
