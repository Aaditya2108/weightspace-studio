"""Tests for TensorDescriptor and checksum computation."""

import numpy as np
import pytest
from pydantic import ValidationError

from weightspace.core.descriptor import TensorDescriptor, compute_tensor_checksum


def test_tensor_descriptor_immutability():
    """Verify that TensorDescriptor is frozen and cannot be modified."""
    desc = TensorDescriptor(
        name="layer.weight",
        shape=(10, 5),
        dtype="float32",
        checksum="abcd1234",
    )

    with pytest.raises(ValidationError):
        # In Pydantic v2 frozen models, attribute assignment raises ValidationError
        desc.name = "new_name"  # type: ignore

    with pytest.raises(ValidationError):
        desc.shape = (20, 5)  # type: ignore


def test_tensor_descriptor_extra_fields_forbidden():
    """Verify that unexpected extra fields are rejected."""
    with pytest.raises(ValidationError):
        TensorDescriptor(
            name="layer.weight",
            shape=(10, 5),
            dtype="float32",
            checksum="abcd1234",
            unexpected_field="disallowed",  # type: ignore
        )


def test_tensor_descriptor_unknown_properties_explicit():
    """Verify that unknown properties remain None and are not guessed."""
    desc = TensorDescriptor(
        name="layer.weight",
        shape=(10, 5),
        dtype="float32",
        checksum="abcd1234",
    )

    assert desc.role is None
    assert desc.module is None
    assert desc.trainable is None
    assert desc.layout is None
    assert desc.device == "cpu"


def test_tensor_descriptor_from_tensor():
    """Verify creation from numpy array."""
    arr = np.ones((4, 8), dtype=np.float32)
    desc = TensorDescriptor.from_tensor("fc.weight", arr)

    assert desc.name == "fc.weight"
    assert desc.shape == (4, 8)
    assert desc.dtype == "float32"
    assert desc.checksum == compute_tensor_checksum(arr)
    assert desc.role is None
    assert desc.trainable is None


def test_compute_tensor_checksum_deterministic():
    """Verify checksum is deterministic and sensitive to data changes."""
    arr1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr3 = np.array([[1.0, 2.0], [3.0, 4.0001]], dtype=np.float32)

    c1 = compute_tensor_checksum(arr1)
    c2 = compute_tensor_checksum(arr2)
    c3 = compute_tensor_checksum(arr3)

    assert c1 == c2
    assert c1 != c3
    assert len(c1) == 64  # Standard SHA-256 hex string


def test_compute_tensor_checksum_invalid_type():
    """Verify checksum raises TypeError on unsupported types."""
    with pytest.raises(TypeError):
        compute_tensor_checksum("not_a_tensor")

