"""
Stage 3 — Export & quantization tests.

Requires: `make export` (or `dvc repro`) already run.
"""

from pathlib import Path

import numpy as np
import pytest

ARTIFACTS = {
    "onnx_fp32": Path("artifacts/model.onnx"),
    "onnx_int8": Path("artifacts/model_int8.onnx"),
    "tflite_int8": Path("artifacts/model_int8.tflite"),
}

# Expected YOLOv8n output shape for 640x640 input
EXPECTED_OUTPUT_SHAPE = (1, 5, 2100)
DUMMY_INPUT = np.zeros((1, 3, 320, 320), dtype=np.float32)

# ── Existence ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("key,path", ARTIFACTS.items())
def test_artifact_exists(key, path):
    assert path.exists(), f"Missing artifact [{key}]: {path}"


@pytest.mark.parametrize("key,path", ARTIFACTS.items())
def test_artifact_not_empty(key, path):
    size_mb = path.stat().st_size / 1_000_000
    assert size_mb > 0.5, f"Artifact [{key}] suspiciously small: {size_mb:.2f} MB"


# ── Loadability ───────────────────────────────────────────────────────────────


def test_onnx_fp32_loadable():
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(ARTIFACTS["onnx_fp32"]), providers=["CPUExecutionProvider"])
    assert sess is not None


def test_onnx_int8_loadable():
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(ARTIFACTS["onnx_int8"]), providers=["CPUExecutionProvider"])
    assert sess is not None


def test_tflite_loadable():
    tf = pytest.importorskip("tensorflow", reason="tensorflow not installed — skipping TFLite test")
    tflite_path = ARTIFACTS["tflite_int8"]
    if not tflite_path.exists():
        pytest.skip(
            "model_int8.tflite not found — TFLite export may have been skipped on this machine"
        )
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    assert interp is not None


# ── Input / output shapes ─────────────────────────────────────────────────────


def test_onnx_fp32_input_shape():
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(ARTIFACTS["onnx_fp32"]), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    # Shape may have dynamic dims (None) — check rank and known dims
    assert len(inp.shape) == 4, f"Expected 4-dim input, got {inp.shape}"
    assert inp.shape[1] == 3, f"Expected 3 channels, got {inp.shape[1]}"


def test_onnx_fp32_output_shape():
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(ARTIFACTS["onnx_fp32"]), providers=["CPUExecutionProvider"])
    outputs = sess.run(None, {sess.get_inputs()[0].name: DUMMY_INPUT})
    assert outputs[0].shape == EXPECTED_OUTPUT_SHAPE, (
        f"FP32 output shape {outputs[0].shape} != expected {EXPECTED_OUTPUT_SHAPE}"
    )


def test_onnx_int8_output_shape():
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(ARTIFACTS["onnx_int8"]), providers=["CPUExecutionProvider"])
    outputs = sess.run(None, {sess.get_inputs()[0].name: DUMMY_INPUT})
    assert outputs[0].shape == EXPECTED_OUTPUT_SHAPE, (
        f"INT8 output shape {outputs[0].shape} != expected {EXPECTED_OUTPUT_SHAPE}"
    )


def test_fp32_and_int8_output_shapes_match():
    """Both formats must produce the same output shape."""
    ort = pytest.importorskip("onnxruntime")
    fp32 = ort.InferenceSession(str(ARTIFACTS["onnx_fp32"]), providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(ARTIFACTS["onnx_int8"]), providers=["CPUExecutionProvider"])
    out_fp32 = fp32.run(None, {fp32.get_inputs()[0].name: DUMMY_INPUT})[0].shape
    out_int8 = int8.run(None, {int8.get_inputs()[0].name: DUMMY_INPUT})[0].shape
    assert out_fp32 == out_int8, f"Shape mismatch: FP32={out_fp32} INT8={out_int8}"


# ── Inference sanity ──────────────────────────────────────────────────────────


def test_onnx_fp32_output_is_finite():
    """Output should not contain NaN or Inf."""
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(ARTIFACTS["onnx_fp32"]), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: DUMMY_INPUT})[0]
    assert np.all(np.isfinite(out)), "FP32 output contains NaN or Inf"


def test_onnx_int8_output_is_finite():
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(ARTIFACTS["onnx_int8"]), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: DUMMY_INPUT})[0]
    assert np.all(np.isfinite(out)), "INT8 output contains NaN or Inf"


# ── Size relationship ─────────────────────────────────────────────────────────


def test_int8_smaller_than_fp32():
    fp32_mb = ARTIFACTS["onnx_fp32"].stat().st_size / 1_000_000
    int8_mb = ARTIFACTS["onnx_int8"].stat().st_size / 1_000_000
    assert int8_mb < fp32_mb, (
        f"INT8 ({int8_mb:.2f} MB) should be smaller than FP32 ({fp32_mb:.2f} MB)"
    )


def test_int8_compression_ratio_reasonable():
    """INT8 should be roughly 2-4x smaller than FP32 for YOLOv8n."""
    fp32_mb = ARTIFACTS["onnx_fp32"].stat().st_size / 1_000_000
    int8_mb = ARTIFACTS["onnx_int8"].stat().st_size / 1_000_000
    ratio = fp32_mb / int8_mb
    assert ratio >= 1.5, f"Compression ratio {ratio:.2f}x too low — quantization may have failed"
