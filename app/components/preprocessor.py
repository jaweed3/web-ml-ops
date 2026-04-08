import cv2
import numpy as np

from app.constant import DEFAULT_IMGSZ, LETTERBOX_PAD_VALUE
from core.logger import get_logger

log = get_logger("components.preprocessor")


class ImagePreprocessor:
    """
    Converts raw image bytes into a normalized float32 tensor ready for
    ONNX Runtime inference.

    Steps:
    1. Decode image bytes (JPEG / PNG / WebP).
    2. Letterbox-resize to *imgsz* × *imgsz* preserving aspect ratio.
    3. BGR → RGB.
    4. Normalize pixel values to [0, 1].
    5. HWC → NCHW layout, add batch dim.

    The output shape is always ``(1, 3, imgsz, imgsz)`` regardless of the
    original image dimensions.
    """

    def __init__(self, imgsz: int = DEFAULT_IMGSZ) -> None:
        self.imgsz = imgsz

    # ── public ────────────────────────────────────────────────────────────────

    def run(self, image_bytes: bytes) -> np.ndarray:
        """
        Parameters
        ----------
        image_bytes : bytes
            Raw bytes of a JPEG, PNG, or WebP image.

        Returns
        -------
        np.ndarray
            Float32 tensor of shape ``(1, 3, imgsz, imgsz)``.

        Raises
        ------
        ValueError
            If the bytes cannot be decoded as an image.
        """
        img = self._decode(image_bytes)
        img = self._letterbox(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))          # HWC → CHW
        img = np.expand_dims(img, axis=0)            # CHW → NCHW
        blob = np.ascontiguousarray(img)
        log.info("preprocess_ok", shape=list(blob.shape), dtype=str(blob.dtype))
        return blob

    # ── private ───────────────────────────────────────────────────────────────

    def _decode(self, data: bytes) -> np.ndarray:
        if not data:
            raise ValueError("Could not decode image — unsupported format or corrupt file")
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(
                "Could not decode image — unsupported format or corrupt file"
            )
        return img

    def _letterbox(self, img: np.ndarray) -> np.ndarray:
        """Resize with grey padding to preserve aspect ratio."""
        h, w   = img.shape[:2]
        scale  = self.imgsz / max(h, w)
        new_h  = int(h * scale)
        new_w  = int(w * scale)
        img    = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h  = self.imgsz - new_h
        pad_w  = self.imgsz - new_w
        top    = pad_h // 2
        bottom = pad_h - top
        left   = pad_w // 2
        right  = pad_w - left

        img = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=LETTERBOX_PAD_VALUE,
        )
        return img
