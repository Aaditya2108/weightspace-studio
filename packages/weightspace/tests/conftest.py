"""Pytest fixtures and helpers for WeightSpace tests."""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pytest
from safetensors.numpy import save_file


@pytest.fixture
def sample_tensors() -> Dict[str, np.ndarray]:
    """Provide a standard dictionary of reproducible numpy weight tensors."""
    rng = np.random.default_rng(42)
    return {
        "encoder.weight": rng.standard_normal((32, 16)).astype(np.float32),
        "encoder.bias": rng.standard_normal((32,)).astype(np.float32),
        "decoder.weight": rng.standard_normal((16, 32)).astype(np.float32),
        "decoder.bias": rng.standard_normal((16,)).astype(np.float32),
    }


@pytest.fixture
def create_safetensors_file(tmp_path: Path):
    """Factory fixture to create valid .safetensors files on demand."""
    def _create(
        tensors: Dict[str, np.ndarray],
        filename: str = "model.safetensors",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Path:
        out_path = tmp_path / filename
        save_file(tensors, str(out_path), metadata=metadata)
        return out_path

    return _create


@pytest.fixture
def corrupted_safetensors_file(tmp_path: Path) -> Path:
    """Create an invalid corrupted safetensors file."""
    bad_file = tmp_path / "corrupted.safetensors"
    bad_file.write_bytes(b"\x00\x01\x02\x03CORRUPTED_HEADER_DATA_123456789")
    return bad_file

