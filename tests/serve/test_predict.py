def test_predict_200_with_valid_jpeg(client, sample_image_bytes):
    r = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert r.status_code == 200


def test_predict_response_has_required_keys(client, sample_image_bytes):
    body = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    ).json()
    for key in ("detections", "inference_time_ms", "model_version", "request_id", "image_shape"):
        assert key in body, f"Missing key: {key}"


def test_predict_422_for_text_file(client):
    r = client.post(
        "/predict",
        files={"file": ("readme.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 422


def test_predict_422_for_corrupt_jpeg(client):
    r = client.post(
        "/predict",
        files={"file": ("bad.jpg", b"\xff\xd8corrupt_data", "image/jpeg")},
    )
    assert r.status_code == 422


def test_predict_422_for_empty_file(client):
    r = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert r.status_code == 422


def test_predict_response_has_x_request_id_header(client, sample_image_bytes):
    r = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert "x-request-id" in r.headers


def test_predict_detections_is_list(client, sample_image_bytes):
    body = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    ).json()
    assert isinstance(body["detections"], list)


def test_predict_image_shape_is_640(client, sample_image_bytes):
    body = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    ).json()
    assert body["image_shape"] == [640, 640]
