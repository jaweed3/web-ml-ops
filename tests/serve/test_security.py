"""
Security layer tests — API key auth, Content-Length guard, and rate limiting.

These tests run against the same mocked client fixture from conftest.py
(no real model, no DagsHub). Auth is toggled by patching the module-level
_API_KEY constant in app.dependencies.
"""

from unittest.mock import patch


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


def test_model_info_401_when_api_key_required_and_missing(client):
    with patch("app.dependencies._API_KEY", "secret-key"):
        r = client.get("/model/info")
    assert r.status_code == 401


def test_model_info_200_when_api_key_correct(client):
    with patch("app.dependencies._API_KEY", "secret-key"):
        r = client.get("/model/info", headers={"X-API-Key": "secret-key"})
    assert r.status_code == 200


def test_model_version_401_when_api_key_required_and_missing(client):
    with patch("app.dependencies._API_KEY", "secret-key"):
        r = client.get("/model/version")
    assert r.status_code == 401


def test_model_version_200_when_api_key_correct(client):
    with patch("app.dependencies._API_KEY", "secret-key"):
        r = client.get("/model/version", headers={"X-API-Key": "secret-key"})
    assert r.status_code == 200


def test_health_passes_without_api_key(client):
    with patch("app.dependencies._API_KEY", "secret-key"):
        r = client.get("/health")
    assert r.status_code == 200


def test_ready_passes_without_api_key(client):
    with patch("app.dependencies._API_KEY", "secret-key"):
        r = client.get("/ready")
    assert r.status_code == 200


def test_predict_returns_generic_message_on_corrupt_image(client):
    r = client.post(
        "/predict",
        files={"file": ("bad.bin", b"\x00\x01\x02", "image/jpeg")},
    )
    assert r.status_code == 422
    assert "corrupt" in r.json()["detail"].lower()


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


def test_predict_429_when_rate_limit_exceeded(client, sample_image_bytes):
    """11th request from same IP within one minute must return 429."""
    from app.dependencies import limiter

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

    limiter._storage.reset()


def test_feedback_401_when_api_key_required_and_missing(client):
    with patch("app.dependencies._API_KEY", "secret-key"):
        r = client.post("/feedback", json={"request_id": "test", "detections": []})
    assert r.status_code == 401


def test_security_headers_present(client, sample_image_bytes):
    r = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"
    assert r.headers.get("cache-control") == "no-store"
