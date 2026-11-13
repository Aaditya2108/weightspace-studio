"""WeightModel core abstraction representing a neural network checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Union
import uuid
import numpy as np

from .descriptor import TensorDescriptor
from .exceptions import TensorNotFoundError, UnsupportedFormatError
from .registry import TensorRegistry
from ..io.safetensors import load_safetensors


class WeightModel:
    """Represents a loaded neural network checkpoint.

    Encapsulates:
    - Unique model identifier
    - Raw immutable tensor data
    - Per-model isolated thread-safe TensorRegistry
    - Optional checkpoint metadata
    - Tensor inspection and summary capabilities
    """

    def __init__(
        self,
        tensors: Mapping[str, np.ndarray],
        *,
        metadata: Optional[Dict[str, str]] = None,
        source_path: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self._model_id = model_id or str(uuid.uuid4())
        self._source_path = source_path
        self._metadata: Dict[str, str] = dict(metadata) if metadata else {}
        self._registry = TensorRegistry(model_id=self._model_id)

        # Store tensors preserving original arrays without in-place mutation
        # Make internal arrays read-only where possible to guarantee preservation
        self._tensors: Dict[str, np.ndarray] = {}
        for name, arr in tensors.items():
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)
            # Create a detached read-only copy for protection
            arr_copy = arr.copy()
            arr_copy.flags.writeable = False
            self._tensors[name] = arr_copy

            # Construct and register immutable descriptor
            # Do not guess role, module, or trainability when not in metadata
            descriptor = TensorDescriptor.from_tensor(
                name=name,
                tensor=arr_copy,
                role=None,
                module=None,
                layout="strided" if arr_copy.flags.c_contiguous else None,
                trainable=None,
            )
            self._registry.register(descriptor)

    @property
    def model_id(self) -> str:
        """Unique ID for this WeightModel instance."""
        return self._model_id

    @property
    def source_path(self) -> Optional[str]:
        """Source checkpoint path if loaded from file."""
        return self._source_path

    @property
    def metadata(self) -> Dict[str, str]:
        """Metadata associated with the checkpoint."""
        return dict(self._metadata)

    @property
    def registry(self) -> TensorRegistry:
        """The per-model thread-safe TensorRegistry."""
        return self._registry

    @classmethod
    def load(cls, path: Union[str, Path]) -> WeightModel:
        """Load a neural network checkpoint from disk.

        In Milestone 1, Safetensors (.safetensors) is the only supported format.
        """
        file_path = Path(path)
        ext = file_path.suffix.lower()
        if ext != ".safetensors":
            raise UnsupportedFormatError(
                f"Unsupported checkpoint format '{ext}'. Milestone 1 only supports '.safetensors'."
            )

        data = load_safetensors(file_path)
        return cls(
            tensors=data.tensors,
            metadata=data.metadata,
            source_path=str(file_path),
        )

    def inspect(self, target: str) -> TensorDescriptor:
        """Inspect a specific tensor by name, returning its immutable TensorDescriptor."""
        return self._registry.get(target)

    def get_tensor(self, name: str) -> np.ndarray:
        """Retrieve a copy of a tensor's data by name.

        Preserves original checkpoint tensors against external in-place modifications.
        """
        if name not in self._tensors:
            raise TensorNotFoundError(
                f"Tensor '{name}' not found in model '{self._model_id}'"
            )
        # Return a copy to prevent external mutation
        return self._tensors[name].copy()

    def summary(self) -> str:
        """Generate a structured textual overview of the model and its parameters."""
        total_elements = sum(t.size for t in self._tensors.values())
        total_bytes = sum(t.nbytes for t in self._tensors.values())
        size_mb = total_bytes / (1024 * 1024)

        lines = [
            f"WeightModel [ID: {self._model_id}]",
            f"Source: {self._source_path or '<in-memory>'}",
            f"Tensors: {len(self._tensors)}",
            f"Total Tensor Elements: {total_elements:,}",
            f"Estimated Size: {size_mb:.2f} MB",
        ]
        if self._metadata:
            lines.append("Metadata:")
            for k, v in self._metadata.items():
                lines.append(f"  {k}: {v}")

        lines.append("Tensor Registry:")
        for desc in self._registry.list():
            lines.append(f"  - {desc.name}: shape={desc.shape}, dtype={desc.dtype}, checksum={desc.checksum[:8]}...")

        return "\n".join(lines)
