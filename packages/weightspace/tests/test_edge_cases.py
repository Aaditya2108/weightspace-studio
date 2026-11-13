"""Edge cases, failure modes, and optional dependency handling."""

from unittest.mock import patch
import numpy as np
import pytest
from pydantic import ValidationError

from weightspace.adapters.pytorch import PyTorchAdapter
from weightspace.core.descriptor import TensorDescriptor, compute_tensor_checksum
from weightspace.core.exceptions import (
    DuplicateTensorError,
    PyTorchNotAvailableError,
    TensorNotFoundError,
)
from weightspace.core.model import WeightModel
from weightspace.core.registry import TensorRegistry


def test_optional_pytorch_availability_when_missing():
    """Verify PyTorchAdapter behaves correctly when PyTorch is unavailable."""
    with patch.object(PyTorchAdapter, "is_available", return_value=False):
        assert PyTorchAdapter.is_available() is False

    # Simulate torch missing by mocking import inside load_into_module
    with patch("builtins.__import__", side_effect=ImportError("No module named 'torch'")):
        with pytest.raises(PyTorchNotAvailableError) as exc_info:
            dummy_model = WeightModel(tensors={})
            PyTorchAdapter.load_into_module(dummy_model, None)
        assert "PyTorch is not available" in str(exc_info.value)


def test_empty_weight_model():
    """Verify WeightModel handles empty tensor sets gracefully."""
    empty_model = WeightModel(tensors={}, metadata={"note": "empty"})
    assert len(empty_model.registry) == 0
    assert len(empty_model.registry.names()) == 0
    assert empty_model.metadata == {"note": "empty"}

    summary = empty_model.summary()
    assert "Tensors: 0" in summary
    assert "Total Parameters: 0" in summary


def test_invalid_tensor_descriptor_missing_fields():
    """Verify TensorDescriptor requires core fields."""
    with pytest.raises(ValidationError):
        # Missing shape, dtype, checksum
        TensorDescriptor(name="invalid")  # type: ignore

    with pytest.raises(ValidationError):
        # Missing checksum
        TensorDescriptor(name="valid_name", shape=(2, 2), dtype="float32")  # type: ignore


def test_non_contiguous_array_checksum():
    """Verify compute_tensor_checksum works on non-contiguous array slices."""
    base = np.arange(20, dtype=np.float32).reshape(4, 5)
    non_contig = base[:, ::2]  # strided, non-contiguous
    assert not non_contig.flags.c_contiguous

    checksum = compute_tensor_checksum(non_contig)
    assert isinstance(checksum, str)
    assert len(checksum) == 64

    # Verify checksum matches contiguous equivalent
    contig = np.ascontiguousarray(non_contig)
    assert checksum == compute_tensor_checksum(contig)


def test_duplicate_registration_in_model_init():
    """Verify registry rejects duplicate names."""
    registry = TensorRegistry(model_id="test-dup")
    desc1 = TensorDescriptor(name="w", shape=(2,), dtype="float32", checksum="1234")
    registry.register(desc1)

    with pytest.raises(DuplicateTensorError):
        registry.register(desc1)

