"""PyTorch adapter for loading WeightModel parameters into explicit torch.nn.Modules."""

from __future__ import annotations

from typing import Any
from ..core.exceptions import ModuleMismatchError, PyTorchNotAvailableError
from ..core.model import WeightModel


class PyTorchAdapter:
    """Adapter bridging WeightSpace models with PyTorch.

    Does not guess or reconstruct architectures from raw state dicts. Instead,
    it loads tensor parameters directly into an explicitly supplied torch.nn.Module.
    Handles PyTorch lazily and gracefully when torch is not installed.
    """

    @staticmethod
    def is_available() -> bool:
        """Return True if PyTorch is installed and importable, False otherwise."""
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def load_into_module(
        cls,
        weight_model: WeightModel,
        module: Any,
        *,
        strict: bool = True,
    ) -> Any:
        """Load weights from a WeightModel into an explicitly supplied torch.nn.Module.

        Args:
            weight_model: Source WeightModel.
            module: Target torch.nn.Module instance.
            strict: If True, requires exact key and shape matching.

        Returns:
            The target module with updated weights.

        Raises:
            PyTorchNotAvailableError: If PyTorch is not installed.
            TypeError: If module is not a torch.nn.Module instance.
            ModuleMismatchError: If keys or tensor shapes mismatch under strict mode.
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise PyTorchNotAvailableError(
                "PyTorch is not available. Please install 'torch' to use PyTorchAdapter."
            ) from e

        if not isinstance(module, nn.Module):
            raise TypeError(
                f"Expected target to be a torch.nn.Module, got {type(module)}"
            )

        module_state = module.state_dict()
        new_state = {}

        model_tensor_names = set(weight_model.registry.names())
        module_param_names = set(module_state.keys())

        missing_keys = module_param_names - model_tensor_names
        unexpected_keys = model_tensor_names - module_param_names

        if strict and (missing_keys or unexpected_keys):
            raise ModuleMismatchError(
                f"Mismatch loading weights into module: "
                f"missing={sorted(missing_keys)}, unexpected={sorted(unexpected_keys)}"
            )

        for name, param in module_state.items():
            if name in model_tensor_names:
                arr = weight_model.get_tensor(name)
                if tuple(arr.shape) != tuple(param.shape):
                    raise ModuleMismatchError(
                        f"Shape mismatch for tensor '{name}': model has {tuple(arr.shape)}, "
                        f"module expects {tuple(param.shape)}"
                    )
                # Convert numpy array to torch tensor matching target device and dtype
                t = torch.from_numpy(arr).to(device=param.device, dtype=param.dtype)
                new_state[name] = t
            elif not strict:
                new_state[name] = param

        module.load_state_dict(new_state, strict=strict)
        return module

