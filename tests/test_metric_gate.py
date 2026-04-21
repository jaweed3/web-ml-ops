"""
Unit tests for metric_gate() in pipeline/stage5_register.py.

No MLflow, no filesystem — pure logic tests.
"""


from pipeline.stage5_register import DEGRADATION_THRESHOLD, metric_gate


def _report(fp32: float | None, int8: float | None) -> dict:
    """Build a minimal benchmark report dict."""
    results = []
    if fp32 is not None:
        results.append({"format": "onnx_fp32", "mAP50": fp32, "mean_latency_ms": 10})
    if int8 is not None:
        results.append({"format": "onnx_int8", "mAP50": int8, "mean_latency_ms": 8})
    return {"n_samples": 100, "results": results}


# ── Gate passes ───────────────────────────────────────────────────────────────


def test_gate_passes_when_ratio_above_threshold():
    assert metric_gate(_report(fp32=0.80, int8=0.78)) is True  # ratio=0.975


def test_gate_passes_at_exact_threshold():
    int8 = round(0.80 * DEGRADATION_THRESHOLD, 6)
    assert metric_gate(_report(fp32=0.80, int8=int8)) is True


def test_gate_passes_when_models_identical():
    assert metric_gate(_report(fp32=0.70, int8=0.70)) is True  # ratio=1.0


# ── Gate fails ────────────────────────────────────────────────────────────────


def test_gate_fails_when_ratio_below_threshold():
    assert metric_gate(_report(fp32=0.80, int8=0.50)) is False  # ratio=0.625


def test_gate_fails_just_below_threshold():
    int8 = 0.80 * (DEGRADATION_THRESHOLD - 0.001)
    assert metric_gate(_report(fp32=0.80, int8=int8)) is False


# ── Missing metrics → skip gate (return True) ─────────────────────────────────


def test_gate_skips_when_fp32_missing():
    """fp32 not in report → gate cannot evaluate → pass through."""
    assert metric_gate(_report(fp32=None, int8=0.50)) is True


def test_gate_skips_when_int8_missing():
    assert metric_gate(_report(fp32=0.80, int8=None)) is True


def test_gate_skips_when_both_missing():
    assert metric_gate(_report(fp32=None, int8=None)) is True


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_gate_passes_when_fp32_is_zero():
    """fp32=0 would cause division by zero — should default ratio to 1.0 and pass."""
    assert metric_gate(_report(fp32=0.0, int8=0.0)) is True


def test_gate_handles_empty_results_list():
    assert metric_gate({"n_samples": 0, "results": []}) is True


def test_gate_handles_extra_formats_in_report():
    """Extra formats (e.g. tflite) must not interfere with the FP32 vs INT8 gate."""
    report = {
        "n_samples": 100,
        "results": [
            {"format": "onnx_fp32", "mAP50": 0.80},
            {"format": "onnx_int8", "mAP50": 0.78},
            {"format": "tflite_int8", "mAP50": 0.75},
        ],
    }
    assert metric_gate(report) is True
