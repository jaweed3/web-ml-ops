"""
Security layer tests — API key auth, Content-Length guard, and rate limiting.

These tests run against the same mocked client fixture from conftest.py
(no real model, no DagsHub). Auth is toggled by patching the module-level
_API_KEY constant in app.dependencies.
"""

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


# ── Rate limiting ─────────────────────────────────────────────────────────────


def test_predict_429_when_rate_limit_exceeded(client, sample_image_bytes):
    """11th request from same IP within one minute must return 429."""
    from app.dependencies import limiter

    # Start from a clean counter so this test doesn't depend on run order.
    limiter._storage.reset()

    def _post():
        return client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )

    for _ in range(10):
        r = _post()
        assert r.status_code == 200, f"Unexpected failure before limit: {r.status_code}"

    over_limit = _post()
    assert over_limit.status_code == 429

    # Restore clean state for tests that run after this one
    limiter._storage.reset()
