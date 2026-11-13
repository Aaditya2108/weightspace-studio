"""WeightSpace: Universal framework for Weight-Space Programming (WSP)."""

from .adapters.pytorch import PyTorchAdapter
from .core.descriptor import TensorDescriptor, compute_tensor_checksum
from .core.exceptions import (
    AdapterError,
    CorruptedFileError,
    DescriptorValidationError,
    DuplicateTensorError,
    ModelLoadError,
    ModuleMismatchError,
    PyTorchNotAvailableError,
    TensorNotFoundError,
    UnsupportedFormatError,
    WeightSpaceError,
)
from .core.model import WeightModel
from .core.registry import TensorRegistry
from .io.safetensors import SafetensorsData, load_safetensors

__version__ = "0.1.0"

__all__ = [
    "AdapterError",
    "CorruptedFileError",
    "DescriptorValidationError",
    "DuplicateTensorError",
    "ModelLoadError",
    "ModuleMismatchError",
    "PyTorchAdapter",
    "PyTorchNotAvailableError",
    "SafetensorsData",
    "TensorDescriptor",
    "TensorNotFoundError",
    "TensorRegistry",
    "UnsupportedFormatError",
    "WeightModel",
    "WeightSpaceError",
    "compute_tensor_checksum",
    "load_safetensors",
]

