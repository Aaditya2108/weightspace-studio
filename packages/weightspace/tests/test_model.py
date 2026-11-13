"""Tests for WeightModel abstraction."""

from pathlib import Path
import numpy as np
import pytest

from weightspace.core.exceptions import TensorNotFoundError, UnsupportedFormatError
from weightspace.core.model import WeightModel


def test_model_load_safetensors(create_safetensors_file, sample_tensors):
    meta = {"architecture": "mlp", "version": "1.0"}
    path = create_safetensors_file(sample_tensors, filename="mlp.safetensors", metadata=meta)

    model = WeightModel.load(path)

    assert model.model_id is not None
    assert model.source_path == str(path)
    assert model.metadata == meta
    assert len(model.registry) == len(sample_tensors)

    for name in sample_tensors:
        desc = model.inspect(name)
        assert desc.name == name
        assert desc.shape == sample_tensors[name].shape
        assert desc.dtype == str(sample_tensors[name].dtype)


def test_multiple_models_isolation(create_safetensors_file, sample_tensors):
    """Verify that multiple WeightModels have isolated registries and state."""
    path1 = create_safetensors_file(sample_tensors, filename="m1.safetensors")
    subset = {"encoder.weight": sample_tensors["encoder.weight"]}
    path2 = create_safetensors_file(subset, filename="m2.safetensors")

    m1 = WeightModel.load(path1)
    m2 = WeightModel.load(path2)

    assert m1.model_id != m2.model_id
    assert m1.registry is not m2.registry
    assert m1.registry.model_id == m1.model_id
    assert m2.registry.model_id == m2.model_id

    assert len(m1.registry) == 4
    assert len(m2.registry) == 1
    assert "decoder.weight" in m1.registry
    assert "decoder.weight" not in m2.registry


def test_model_tensor_preservation_against_in_place_mutation(sample_tensors):
    """Verify original checkpoint tensors are preserved against in-place modifications."""
    model = WeightModel(tensors=sample_tensors)
    initial_desc = model.inspect("encoder.weight")

    # Get tensor and attempt to mutate it in place
    retrieved = model.get_tensor("encoder.weight")
    retrieved.fill(999.0)

    # Check that model's internal tensor was not mutated
    fresh_tensor = model.get_tensor("encoder.weight")
    assert not np.allclose(fresh_tensor, 999.0)
    np.testing.assert_array_equal(fresh_tensor, sample_tensors["encoder.weight"])

    # Check that descriptor and checksum remain valid
    assert model.inspect("encoder.weight").checksum == initial_desc.checksum


def test_model_unsupported_format(tmp_path: Path):
    bad_format = tmp_path / "model.pt"
    bad_format.write_bytes(b"dummy_bytes")

    with pytest.raises(UnsupportedFormatError) as exc_info:
        WeightModel.load(bad_format)
    assert "Milestone 1 only supports '.safetensors'" in str(exc_info.value)


def test_model_inspect_missing_tensor(sample_tensors):
    model = WeightModel(tensors=sample_tensors)
    with pytest.raises(TensorNotFoundError):
        model.inspect("nonexistent.weight")

    with pytest.raises(TensorNotFoundError):
        model.get_tensor("nonexistent.weight")


def test_model_summary(sample_tensors):
    model = WeightModel(tensors=sample_tensors, metadata={"key": "val"})
    summary = model.summary()

    assert model.model_id in summary
    assert "Tensors: 4" in summary
    assert "encoder.weight" in summary
    assert "key: val" in summary

