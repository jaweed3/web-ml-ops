def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_body_has_status_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_health_body_has_timestamp(client):
    body = client.get("/health").json()
    assert "timestamp" in body


def test_ready_returns_200_when_model_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200


def test_ready_body_has_status_ready(client):
    body = client.get("/ready").json()
    assert body["status"] == "ready"


def test_ready_body_has_model_version(client):
    body = client.get("/ready").json()
    assert "model_version" in body


def test_ready_body_has_loaded_at(client):
    body = client.get("/ready").json()
    assert "loaded_at" in body
