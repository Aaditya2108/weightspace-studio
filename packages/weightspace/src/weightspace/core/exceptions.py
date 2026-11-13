"""Domain exceptions for WeightSpace framework."""


class WeightSpaceError(Exception):
    """Base exception for all WeightSpace errors."""
    pass


class ModelLoadError(WeightSpaceError):
    """Raised when loading a model checkpoint fails."""
    pass


class CorruptedFileError(ModelLoadError):
    """Raised when a checkpoint file is corrupted or unreadable."""
    pass


class UnsupportedFormatError(ModelLoadError):
    """Raised when a checkpoint format is not supported."""
    pass


class TensorNotFoundError(WeightSpaceError):
    """Raised when a requested tensor cannot be found in the registry or model."""
    pass


class DuplicateTensorError(WeightSpaceError):
    """Raised when attempting to register a tensor name that already exists in the registry."""
    pass


class DescriptorValidationError(WeightSpaceError):
    """Raised when a TensorDescriptor fails validation."""
    pass


class AdapterError(WeightSpaceError):
    """Base exception for model adapter errors."""
    pass


class PyTorchNotAvailableError(AdapterError):
    """Raised when a PyTorch operation is requested but torch is not installed."""
    pass


class ModuleMismatchError(AdapterError):
    """Raised when weights do not match the target module structure."""
    pass

