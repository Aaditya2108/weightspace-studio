"""Tests for Safetensors checkpoint loader."""

from pathlib import Path
from typing import Dict
import numpy as np
import pytest

from weightspace.core.exceptions import CorruptedFileError
from weightspace.io.safetensors import SafetensorsData, load_safetensors


def test_load_safetensors_with_metadata(create_safetensors_file, sample_tensors):
    meta = {"format": "pt", "author": "test-suite", "version": "1.0"}
    path = create_safetensors_file(sample_tensors, filename="with_meta.safetensors", metadata=meta)

    result = load_safetensors(path)
    assert isinstance(result, SafetensorsData)
    assert set(result.tensors.keys()) == set(sample_tensors.keys())
    assert result.metadata == meta

    for k, expected_arr in sample_tensors.items():
        np.testing.assert_array_equal(result.tensors[k], expected_arr)


def test_load_safetensors_missing_metadata(create_safetensors_file, sample_tensors):
    """Verify loading files without metadata works and returns None or empty metadata."""
    path = create_safetensors_file(sample_tensors, filename="no_meta.safetensors", metadata=None)

    result = load_safetensors(path)
    assert isinstance(result, SafetensorsData)
    assert result.metadata is None or result.metadata == {}
    assert set(result.tensors.keys()) == set(sample_tensors.keys())


def test_load_safetensors_file_not_found(tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist.safetensors"
    with pytest.raises(FileNotFoundError):
        load_safetensors(nonexistent)


def test_load_safetensors_corrupted_file(corrupted_safetensors_file):
    """Verify corrupted files raise CorruptedFileError."""
    with pytest.raises(CorruptedFileError):
        load_safetensors(corrupted_safetensors_file)

