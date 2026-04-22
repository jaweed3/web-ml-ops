"""
Unit tests for pipeline/stage1_data.py::_validate_label_file.
Uses tmp_path — no real dataset needed.
"""

import pytest

from pipeline.stage1_data import _validate_label_file


def _write(tmp_path, content: str):
    f = tmp_path / "labels.txt"
    f.write_text(content)
    return f


def test_valid_single_annotation(tmp_path):
    f = _write(tmp_path, "0 0.5 0.5 0.2 0.3\n")
    _validate_label_file(f)  # must not raise


def test_valid_multiple_annotations(tmp_path):
    f = _write(tmp_path, "0 0.1 0.2 0.3 0.4\n1 0.9 0.8 0.1 0.1\n")
    _validate_label_file(f)


def test_valid_empty_file(tmp_path):
    f = _write(tmp_path, "")
    _validate_label_file(f)  # empty label = background image, valid


def test_valid_boundary_values(tmp_path):
    f = _write(tmp_path, "0 0.0 0.0 1.0 1.0\n")
    _validate_label_file(f)


def test_raises_on_wrong_field_count(tmp_path):
    f = _write(tmp_path, "0 0.5 0.5 0.2\n")  # only 4 fields
    with pytest.raises(ValueError, match="expected 5 fields"):
        _validate_label_file(f)


def test_raises_on_non_numeric_value(tmp_path):
    f = _write(tmp_path, "0 abc 0.5 0.2 0.3\n")
    with pytest.raises(ValueError, match="non-numeric"):
        _validate_label_file(f)


def test_raises_on_negative_class_id(tmp_path):
    f = _write(tmp_path, "-1 0.5 0.5 0.2 0.3\n")
    with pytest.raises(ValueError, match="class_id must be"):
        _validate_label_file(f)


def test_raises_on_coord_above_1(tmp_path):
    f = _write(tmp_path, "0 1.1 0.5 0.2 0.3\n")
    with pytest.raises(ValueError, match="out of range"):
        _validate_label_file(f)


def test_raises_on_coord_below_0(tmp_path):
    f = _write(tmp_path, "0 -0.1 0.5 0.2 0.3\n")
    with pytest.raises(ValueError, match="out of range"):
        _validate_label_file(f)


def test_raises_on_second_invalid_line(tmp_path):
    f = _write(tmp_path, "0 0.5 0.5 0.2 0.3\n0 0.5 0.5 0.2\n")
    with pytest.raises(ValueError, match="expected 5 fields"):
        _validate_label_file(f)
