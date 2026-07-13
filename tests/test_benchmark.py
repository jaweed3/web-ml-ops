"""
Stage 4 — Benchmark tests.

Requires: `make benchmark` (or `dvc repro`) already run.
"""

import json
from pathlib import Path

import pytest

REPORT_PATH = Path("artifacts/benchmark_report.json")

# Adjust these based on your target hardware.
# These defaults assume a modern laptop CPU.

SLO_ONNX_FP32_MEAN_LATENCY_MS = 200  # FP32 must be under 200ms mean
SLO_ONNX_INT8_MEAN_LATENCY_MS = 100  # INT8 must be under 100ms mean
SLO_ONNX_INT8_P95_LATENCY_MS = 150  # INT8 p95 must be under 150ms
SLO_FP32_MAX_SIZE_MB = 15  # sanity cap for YOLOv8n FP32
SLO_INT8_MAX_SIZE_MB = 10  # sanity cap for YOLOv8n INT8


@pytest.fixture(scope="module")
def report() -> dict:
    assert REPORT_PATH.exists(), (
        f"Benchmark report not found at {REPORT_PATH} — run `make benchmark` first"
    )
    return json.loads(REPORT_PATH.read_text())


@pytest.fixture(scope="module")
def results_by_format(report) -> dict:
    return {r["format"]: r for r in report["results"]}


def test_report_exists():
    assert REPORT_PATH.exists()


def test_report_top_level_keys(report):
    for key in ("results", "n_samples"):
        assert key in report, f"Missing top-level key: {key}"


def test_report_n_samples_positive(report):
    assert report["n_samples"] > 0


def test_each_result_has_required_fields(report):
    required = {"format", "mean_latency_ms", "p95_latency_ms", "model_size_mb"}
    for r in report["results"]:
        missing = required - r.keys()
        assert not missing, f"Result missing fields {missing}: {r}"


def test_both_onnx_formats_present(results_by_format):
    for fmt in ("onnx_fp32", "onnx_int8"):
        assert fmt in results_by_format, f"Format '{fmt}' missing from benchmark report"


def test_int8_smaller_than_fp32(results_by_format):
    fp32_mb = results_by_format["onnx_fp32"]["model_size_mb"]
    int8_mb = results_by_format["onnx_int8"]["model_size_mb"]
    assert int8_mb < fp32_mb, f"INT8 ({int8_mb} MB) should be smaller than FP32 ({fp32_mb} MB)"


def test_fp32_size_within_sanity_cap(results_by_format):
    size = results_by_format["onnx_fp32"]["model_size_mb"]
    assert size < SLO_FP32_MAX_SIZE_MB, (
        f"FP32 model {size} MB exceeds sanity cap {SLO_FP32_MAX_SIZE_MB} MB"
    )


def test_int8_size_within_sanity_cap(results_by_format):
    size = results_by_format["onnx_int8"]["model_size_mb"]
    assert size < SLO_INT8_MAX_SIZE_MB, (
        f"INT8 model {size} MB exceeds sanity cap {SLO_INT8_MAX_SIZE_MB} MB"
    )


def test_fp32_mean_latency_within_slo(results_by_format):
    lat = results_by_format["onnx_fp32"]["mean_latency_ms"]
    assert lat < SLO_ONNX_FP32_MEAN_LATENCY_MS, (
        f"FP32 mean latency {lat} ms exceeds SLO {SLO_ONNX_FP32_MEAN_LATENCY_MS} ms"
    )


def test_int8_mean_latency_within_slo(results_by_format):
    lat = results_by_format["onnx_int8"]["mean_latency_ms"]
    assert lat < SLO_ONNX_INT8_MEAN_LATENCY_MS, (
        f"INT8 mean latency {lat} ms exceeds SLO {SLO_ONNX_INT8_MEAN_LATENCY_MS} ms"
    )


def test_int8_p95_latency_within_slo(results_by_format):
    lat = results_by_format["onnx_int8"]["p95_latency_ms"]
    assert lat < SLO_ONNX_INT8_P95_LATENCY_MS, (
        f"INT8 p95 latency {lat} ms exceeds SLO {SLO_ONNX_INT8_P95_LATENCY_MS} ms"
    )


def test_latency_values_are_positive(report):
    for r in report["results"]:
        assert r["mean_latency_ms"] > 0, f"Non-positive mean latency in {r['format']}"
        assert r["p95_latency_ms"] > 0, f"Non-positive p95 latency in {r['format']}"


def test_p95_greater_than_mean(report):
    for r in report["results"]:
        assert r["p95_latency_ms"] >= r["mean_latency_ms"], (
            f"p95 ({r['p95_latency_ms']}) < mean ({r['mean_latency_ms']}) for {r['format']} — "
            "this should not happen statistically"
        )
