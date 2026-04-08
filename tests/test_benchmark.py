import json
from pathlib import Path


def test_benchmark_report_exists():
    assert Path("artifacts/benchmark_report.json").exists()


def test_benchmark_report_schema():
    report = json.loads(Path("artifacts/benchmark_report.json").read_text())
    assert "results" in report
    assert "n_samples" in report
    for r in report["results"]:
        assert "mean_latency_ms" in r
        assert "model_size_mb" in r
        assert "format" in r


def test_int8_smaller_than_fp32():
    report = json.loads(Path("artifacts/benchmark_report.json").read_text())
    sizes = {r["format"]: r["model_size_mb"] for r in report["results"]}
    assert "onnx_int8" in sizes and "onnx_fp32" in sizes, (
        "Both onnx_fp32 and onnx_int8 must be present in benchmark report"
    )
    assert sizes["onnx_int8"] < sizes["onnx_fp32"], (
        f"INT8 ({sizes['onnx_int8']} MB) should be smaller than FP32 ({sizes['onnx_fp32']} MB)"
    )
