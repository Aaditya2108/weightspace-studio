"""Tests for TensorDescriptor validation and checksum computation."""

import hashlib
import numpy as np
import pytest
from pydantic import ValidationError

from weightspace.core.descriptor import CHECKSUM_VERSION, TensorDescriptor, compute_tensor_checksum

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

VALID_HEX_64 = "a" * 64


def test_tensor_descriptor_immutability():
    """Verify that TensorDescriptor is frozen and cannot be modified."""
    desc = TensorDescriptor(
        name="layer.weight",
        shape=(10, 5),
        dtype="float32",
        checksum=VALID_HEX_64,
    )

    with pytest.raises(ValidationError):
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
            checksum=VALID_HEX_64,
            unexpected_field="disallowed",  # type: ignore
        )


def test_tensor_descriptor_validation_name():
    """Verify empty and whitespace-only tensor names are rejected."""
    with pytest.raises(ValidationError) as exc1:
        TensorDescriptor(name="", shape=(2, 2), dtype="float32", checksum=VALID_HEX_64)
    assert "name" in str(exc1.value)

    with pytest.raises(ValidationError) as exc2:
        TensorDescriptor(name="   ", shape=(2, 2), dtype="float32", checksum=VALID_HEX_64)
    assert "name" in str(exc2.value)


def test_tensor_descriptor_validation_shape():
    """Verify negative shape dimensions are rejected."""
    with pytest.raises(ValidationError) as exc:
        TensorDescriptor(name="w", shape=(-1, 10), dtype="float32", checksum=VALID_HEX_64)
    assert "Shape dimensions must be non-negative" in str(exc.value)

    # Valid scalar (empty tuple) or multi-dim shapes
    desc_scalar = TensorDescriptor(name="scalar", shape=(), dtype="float32", checksum=VALID_HEX_64)
    assert desc_scalar.shape == ()


def test_tensor_descriptor_validation_checksum():
    """Verify checksum must be a 64-character hexadecimal SHA-256 string."""
    # Too short
    with pytest.raises(ValidationError):
        TensorDescriptor(name="w", shape=(2, 2), dtype="float32", checksum="abcd1234")

    # Too long (65 chars)
    with pytest.raises(ValidationError):
        TensorDescriptor(name="w", shape=(2, 2), dtype="float32", checksum="a" * 65)

    # Non-hex characters
    with pytest.raises(ValidationError):
        TensorDescriptor(name="w", shape=(2, 2), dtype="float32", checksum="g" * 64)

    # Uppercase hex is valid and normalized to lowercase
    desc_upper = TensorDescriptor(name="w", shape=(2, 2), dtype="float32", checksum="A" * 64)
    assert desc_upper.checksum == "a" * 64


def test_tensor_descriptor_unknown_properties_explicit():
    """Verify that unknown properties remain None and are not guessed."""
    desc = TensorDescriptor(
        name="layer.weight",
        shape=(10, 5),
        dtype="float32",
        checksum=VALID_HEX_64,
    )

    assert desc.role is None
    assert desc.module is None
    assert desc.trainable is None
    assert desc.layout is None
    assert desc.device == "cpu"


def test_tensor_descriptor_from_tensor():
    """Verify creation from numpy array computes canonical checksum."""
    arr = np.ones((4, 8), dtype=np.float32)
    desc = TensorDescriptor.from_tensor("fc.weight", arr)

    assert desc.name == "fc.weight"
    assert desc.shape == (4, 8)
    assert desc.dtype == "float32"
    assert desc.checksum == compute_tensor_checksum(arr)
    assert len(desc.checksum) == 64
    assert desc.role is None
    assert desc.trainable is None


def test_compute_tensor_checksum_deterministic():
    """Verify checksum is deterministic across identical arrays."""
    arr1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr3 = np.array([[1.0, 2.0], [3.0, 4.0001]], dtype=np.float32)

    c1 = compute_tensor_checksum(arr1)
    c2 = compute_tensor_checksum(arr2)
    c3 = compute_tensor_checksum(arr3)

    assert c1 == c2
    assert c1 != c3
    assert len(c1) == 64


def test_checksum_different_shape_same_bytes():
    """Verify that identical byte content with different shapes produces different checksums."""
    arr_2d = np.zeros((4, 4), dtype=np.float32)
    arr_1d = np.zeros((16,), dtype=np.float32)

    assert arr_2d.tobytes() == arr_1d.tobytes()

    c_2d = compute_tensor_checksum(arr_2d)
    c_1d = compute_tensor_checksum(arr_1d)

    assert c_2d != c_1d


def test_checksum_different_dtype_same_bytes():
    """Verify that identical byte content with different dtypes produces different checksums."""
    arr_float = np.zeros((16,), dtype=np.float32)
    arr_int = np.zeros((16,), dtype=np.int32)

    assert arr_float.tobytes() == arr_int.tobytes()

    c_float = compute_tensor_checksum(arr_float)
    c_int = compute_tensor_checksum(arr_int)

    assert c_float != c_int


def test_checksum_different_endianness_same_bytes():
    """Verify that same shape and raw bytes with different endian interpretation produce different checksums."""
    raw = b"\x00\x00\x80\x3f\x00\x00\x00\x40"
    arr_le = np.frombuffer(raw, dtype="<f4")
    arr_be = np.frombuffer(raw, dtype=">f4")

    assert arr_le.shape == arr_be.shape
    assert arr_le.tobytes() == arr_be.tobytes()
    assert arr_le.dtype.str != arr_be.dtype.str

    c_le = compute_tensor_checksum(arr_le)
    c_be = compute_tensor_checksum(arr_be)

    assert c_le != c_be


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not available")
def test_checksum_pytorch_bfloat16():
    """Verify checksum computation for PyTorch bfloat16 tensors without .numpy()."""
    t = torch.tensor([1.0, 2.0, 3.5], dtype=torch.bfloat16)
    checksum = compute_tensor_checksum(t)
    assert isinstance(checksum, str)
    assert len(checksum) == 64

    # Verify descriptor construction succeeds with bfloat16
    desc = TensorDescriptor.from_tensor("layer.bf16", t)
    assert desc.dtype == "bfloat16"
    assert desc.checksum == checksum


def test_compute_tensor_checksum_invalid_type():
    """Verify checksum raises TypeError on unsupported types."""
    with pytest.raises(TypeError):
        compute_tensor_checksum("not_a_tensor")
