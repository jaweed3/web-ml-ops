"""
Security layer tests — API key auth and Content-Length guard.

These tests run against the same mocked client fixture from conftest.py
(no real model, no DagsHub). Auth is toggled by patching the module-level
_API_KEY constant in app.dependencies.
"""

import pytest
from unittest.mock import patch


# ── API key auth ──────────────────────────────────────────────────────────────


def test_predict_passes_when_api_key_disabled(client, sample_image_bytes):
    """Default: API_KEY env not set → all requests pass through."""
    r = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert r.status_code == 200


def test_predict_401_when_api_key_required_and_missing(client, sample_image_bytes):
    with patch("app.dependencies._API_KEY", "secret-key"):
        r = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
    assert r.status_code == 401


def test_predict_401_when_api_key_wrong(client, sample_image_bytes):
    with patch("app.dependencies._API_KEY", "secret-key"):
        r = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            headers={"X-API-Key": "wrong-key"},
        )
    assert r.status_code == 401


def test_predict_200_when_api_key_correct(client, sample_image_bytes):
    with patch("app.dependencies._API_KEY", "secret-key"):
        r = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            headers={"X-API-Key": "secret-key"},
        )
    assert r.status_code == 200


# ── Content-Length guard ──────────────────────────────────────────────────────


def test_predict_422_when_content_length_exceeds_limit(client, sample_image_bytes):
    oversized = 11 * 1024 * 1024  # 11 MB declared
    r = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        headers={"Content-Length": str(oversized)},
    )
    assert r.status_code == 422


def test_predict_passes_when_content_length_within_limit(client, sample_image_bytes):
    r = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        headers={"Content-Length": str(len(sample_image_bytes))},
    )
    assert r.status_code == 200
