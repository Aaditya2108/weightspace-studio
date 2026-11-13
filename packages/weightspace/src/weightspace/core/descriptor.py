"""TensorDescriptor and checksum computation for neural network parameters."""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


def compute_tensor_checksum(tensor: Any) -> str:
    """Compute a deterministic SHA-256 checksum over a tensor's contiguous byte buffer.

    Supports numpy ndarray and objects convertable or exposing contiguous bytes.
    Does not modify the input tensor.
    """
    if isinstance(tensor, np.ndarray):
        arr = np.ascontiguousarray(tensor)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    # Check for PyTorch tensor without strictly requiring torch import
    if hasattr(tensor, "detach") and hasattr(tensor, "cpu"):
        cpu_t = tensor.detach().cpu()
        if hasattr(cpu_t, "numpy"):
            arr = np.ascontiguousarray(cpu_t.numpy())
            return hashlib.sha256(arr.tobytes()).hexdigest()

    # Fallback if bytes() or buffer interface is supported
    if isinstance(tensor, (bytes, bytearray, memoryview)):
        return hashlib.sha256(tensor).hexdigest()

    raise TypeError(f"Unsupported tensor type for checksum calculation: {type(tensor)}")


TensorRole = Literal["weight", "bias", "buffer", "parameter", "unknown"]


class TensorDescriptor(BaseModel):
    """Immutable metadata descriptor for a single tensor.

    All attributes are frozen. Does not attempt to guess or infer unknown
    properties such as role, module, or trainability unless explicitly provided.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Unique hierarchical name of the tensor (e.g. 'layer.0.weight')")
    shape: Tuple[int, ...] = Field(..., description="Dimensions of the tensor")
    dtype: str = Field(..., description="Data type representation string (e.g. 'float32')")
    device: str = Field(default="cpu", description="Device location (e.g. 'cpu', 'cuda:0')")
    role: Optional[TensorRole] = Field(default=None, description="Explicit semantic role if known, else None")
    module: Optional[str] = Field(default=None, description="Parent module name if known, else None")
    layout: Optional[str] = Field(default=None, description="Memory layout if known (e.g. 'strided'), else None")
    trainable: Optional[bool] = Field(default=None, description="Trainability status if known, else None")
    checksum: str = Field(..., description="Cryptographic SHA-256 checksum of tensor bytes")

    @classmethod
    def from_tensor(
        cls,
        name: str,
        tensor: Any,
        *,
        role: Optional[TensorRole] = None,
        module: Optional[str] = None,
        layout: Optional[str] = None,
        trainable: Optional[bool] = None,
        device: str = "cpu",
    ) -> TensorDescriptor:
        """Construct a frozen TensorDescriptor from a tensor object and compute its checksum."""
        shape: Tuple[int, ...]
        dtype: str

        if isinstance(tensor, np.ndarray):
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype)
        elif hasattr(tensor, "shape") and hasattr(tensor, "dtype"):
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype).replace("torch.", "")
        else:
            raise TypeError(f"Cannot inspect tensor of type {type(tensor)}")

        checksum = compute_tensor_checksum(tensor)

        return cls(
            name=name,
            shape=shape,
            dtype=dtype,
            device=device,
            role=role,
            module=module,
            layout=layout,
            trainable=trainable,
            checksum=checksum,
        )

