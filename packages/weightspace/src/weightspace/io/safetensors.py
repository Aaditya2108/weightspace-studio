"""Safetensors checkpoint loader for WeightSpace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Union
import numpy as np
from safetensors import safe_open

from ..core.exceptions import CorruptedFileError, ModelLoadError


@dataclass(frozen=True)
class SafetensorsData:
    """Container holding loaded tensors and optional metadata separately."""
    tensors: Mapping[str, np.ndarray]
    metadata: Optional[Dict[str, str]] = None


def load_safetensors(path: Union[str, Path]) -> SafetensorsData:
    """Load a safetensors checkpoint file.

    Returns a SafetensorsData instance separating raw tensor arrays from optional file metadata.
    Does not assume any metadata or state_dict wrapper exists.
    Original tensors are loaded as numpy arrays.

    Raises:
        FileNotFoundError: If the file does not exist.
        CorruptedFileError: If the file is malformed, truncated, or corrupted.
        ModelLoadError: If loading fails for another I/O or parsing reason.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {file_path}")

    tensors: Dict[str, np.ndarray] = {}
    metadata: Optional[Dict[str, str]] = None

    try:
        with safe_open(str(file_path), framework="numpy", device="cpu") as f:
            raw_meta = f.metadata()
            if raw_meta is not None:
                metadata = dict(raw_meta)

            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    except FileNotFoundError:
        raise
    except Exception as e:
        # safetensors C-extension or parser errors on corrupted headers/files
        raise CorruptedFileError(f"Failed to load safetensors file '{file_path}': {e}") from e

    return SafetensorsData(tensors=tensors, metadata=metadata)

