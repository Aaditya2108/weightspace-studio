"""TensorDescriptor and checksum computation for neural network parameters.

Checksum Specification:
    Version: wsp-sha256-v1
    Format: SHA-256 digest computed over:
        1. Version identifier: "wsp-sha256-v1\\n" (ASCII)
        2. Canonical dtype string: "dtype:<name>[<byteorder_str>]\\n" (UTF-8)
        3. Canonical shape string: "shape:<dim0,dim1,...>\\n" (UTF-8)
        4. Contiguous tensor payload bytes.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

CHECKSUM_VERSION = "wsp-sha256-v1"
HEX_64_REGEX = re.compile(r"^[0-9a-fA-F]{64}$")

TensorRole = Literal["weight", "bias", "buffer", "parameter", "unknown"]


def compute_tensor_checksum(tensor: Any) -> str:
    """Compute a deterministic SHA-256 checksum over canonical dtype, shape, and bytes.

    Format Specification (wsp-sha256-v1):
        Header:
            wsp-sha256-v1\\n
            dtype:<canonical_dtype>\\n
            shape:<dim0,dim1,...>\\n
        Payload:
            contiguous raw bytes

    Ensures that identical byte content with different shapes, dtypes, or
    NumPy endianness interpretations produces distinct checksum digests.
    Supports PyTorch tensors (including bfloat16) without relying on .numpy().
    """
    canonical_dtype: str
    canonical_shape: Tuple[int, ...]
    raw_bytes: bytes

    if isinstance(tensor, np.ndarray):
        arr = np.ascontiguousarray(tensor)
        # Preserve full dtype specification including byte order (e.g. '<f4', '>f4', '|u1')
        canonical_dtype = f"{arr.dtype.name}[{arr.dtype.str}]"
        canonical_shape = tuple(arr.shape)
        raw_bytes = arr.tobytes()
    elif hasattr(tensor, "detach") and hasattr(tensor, "cpu"):
        cpu_t = tensor.detach().cpu().contiguous()
        dtype_name = str(cpu_t.dtype).replace("torch.", "")
        canonical_dtype = f"torch:{dtype_name}"
        canonical_shape = tuple(cpu_t.shape)

        if cpu_t.numel() == 0:
            raw_bytes = b""
        else:
            try:
                import torch
                # Safe byte reinterpretation that works for all PyTorch dtypes including bfloat16
                raw_bytes = bytes(cpu_t.reshape(-1).view(torch.uint8))
            except Exception:
                offset = cpu_t.storage_offset() * cpu_t.element_size()
                nbytes = cpu_t.numel() * cpu_t.element_size()
                raw_bytes = bytes(cpu_t.untyped_storage())[offset : offset + nbytes]
    else:
        raise TypeError(f"Unsupported tensor type for checksum calculation: {type(tensor)}")

    hasher = hashlib.sha256()
    hasher.update(f"{CHECKSUM_VERSION}\n".encode("ascii"))
    hasher.update(f"dtype:{canonical_dtype}\n".encode("utf-8"))
    shape_str = ",".join(str(d) for d in canonical_shape)
    hasher.update(f"shape:{shape_str}\n".encode("utf-8"))
    hasher.update(raw_bytes)
    return hasher.hexdigest().lower()


class TensorDescriptor(BaseModel):
    """Immutable metadata descriptor for a single tensor.

    All attributes are frozen. Does not attempt to guess or infer unknown
    properties such as role, module, or trainability unless explicitly provided.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1, description="Unique hierarchical name of the tensor")
    shape: Tuple[int, ...] = Field(..., description="Dimensions of the tensor")
    dtype: str = Field(..., min_length=1, description="Data type representation string (e.g. 'float32')")
    device: str = Field(default="cpu", description="Device location (e.g. 'cpu', 'cuda:0')")
    role: Optional[TensorRole] = Field(default=None, description="Explicit semantic role if known, else None")
    module: Optional[str] = Field(default=None, description="Parent module name if known, else None")
    layout: Optional[str] = Field(default=None, description="Memory layout if known (e.g. 'strided'), else None")
    trainable: Optional[bool] = Field(default=None, description="Trainability status if known, else None")
    checksum: str = Field(..., description="Cryptographic SHA-256 checksum (64-char hex)")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Tensor name cannot be empty or whitespace-only")
        return v

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, v: Tuple[int, ...]) -> Tuple[int, ...]:
        for dim in v:
            if dim < 0:
                raise ValueError(f"Shape dimensions must be non-negative, got {v}")
        return v

    @field_validator("checksum")
    @classmethod
    def validate_checksum(cls, v: str) -> str:
        if not HEX_64_REGEX.match(v):
            raise ValueError(
                f"Checksum must be a valid 64-character hexadecimal SHA-256 string, got '{v}'"
            )
        return v.lower()

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
            dtype = tensor.dtype.name
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
