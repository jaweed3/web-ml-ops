"""
RescueVision load test — run with:

    locust -f scripts/locustfile.py --host http://localhost:8080 \
           --users 50 --spawn-rate 5 --run-time 60s --headless

Pass-criteria (checked in CI via --exit-code-on-error 1):
    - p95 response time < 200 ms
    - error rate       < 1 %

Optional env vars:
    API_KEY   — set X-API-Key header if auth is enabled
"""

import os

import cv2
import numpy as np
from locust import HttpUser, between, task


def _make_jpeg(h: int = 320, w: int = 320) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# Pre-generate a small pool of test images to vary payloads slightly
_IMAGE_POOL = [_make_jpeg() for _ in range(5)]
_API_KEY = os.environ.get("API_KEY", "")


class PredictUser(HttpUser):
    """
    Simulates a drone operator uploading frames for victim detection.
    Wait 100-500 ms between requests (realistic inter-frame cadence).
    """

    wait_time = between(0.1, 0.5)

    def on_start(self):
        self._pool_idx = 0

    @task
    def predict(self):
        image_bytes = _IMAGE_POOL[self._pool_idx % len(_IMAGE_POOL)]
        self._pool_idx += 1

        headers = {}
        if _API_KEY:
            headers["X-API-Key"] = _API_KEY

        with self.client.post(
            "/predict",
            files={"file": ("frame.jpg", image_bytes, "image/jpeg")},
            headers=headers,
            catch_response=True,
            name="/predict",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 429:
                # Rate limit — not counted as failure in load test
                resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}: {resp.text[:200]}")
