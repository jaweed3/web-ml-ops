import pytest
from pathlib import Path


@pytest.mark.parametrize("artifact", [
    "artifacts/model.onnx",
    "artifacts/model_int8.onnx",
    "artifacts/model_int8.tflite",
])
def test_artifact_exists(artifact):
    assert Path(artifact).exists(), f"Missing artifact: {artifact}"


def test_onnx_loadable():
    import onnxruntime as ort
    sess = ort.InferenceSession("artifacts/model.onnx")
    assert sess is not None


def test_onnx_int8_loadable():
    import onnxruntime as ort
    sess = ort.InferenceSession("artifacts/model_int8.onnx")
    assert sess is not None


def test_tflite_loadable():
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path="artifacts/model_int8.tflite")
    interp.allocate_tensors()
    assert interp is not None
