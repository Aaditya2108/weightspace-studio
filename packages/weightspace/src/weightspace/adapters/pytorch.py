"""PyTorch adapter for loading WeightModel parameters into explicit torch.nn.Modules."""

from __future__ import annotations

from typing import Any, Dict, Set
from ..core.exceptions import ModuleMismatchError, PyTorchNotAvailableError
from ..core.model import WeightModel


class PyTorchAdapter:
    """Adapter bridging WeightSpace models with PyTorch.

    Does not guess or reconstruct architectures from raw state dicts. Instead,
    it loads tensor parameters directly into an explicitly supplied torch.nn.Module.
    Validates all matching keys and shapes before touching the target module to
    guarantee atomic updates and prevent partial state modification.
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

        All keys and tensor shapes are pre-validated before any parameters or buffers
        are modified on the target module. If validation fails, the module remains
        in its exact original state.

        Args:
            weight_model: Source WeightModel.
            module: Target torch.nn.Module instance.
            strict: If True, requires exact key and shape matching.

        Returns:
            The target module with updated weights.

        Raises:
            PyTorchNotAvailableError: If PyTorch is not installed.
            TypeError: If module is not a torch.nn.Module instance.
            ModuleMismatchError: If keys or tensor shapes mismatch.
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
        model_tensor_names: Set[str] = set(weight_model.registry.names())
        module_param_names: Set[str] = set(module_state.keys())

        missing_keys = module_param_names - model_tensor_names
        unexpected_keys = model_tensor_names - module_param_names

        # Key validation in strict mode
        if strict and (missing_keys or unexpected_keys):
            raise ModuleMismatchError(
                f"Mismatch loading weights into module: "
                f"missing={sorted(missing_keys)}, unexpected={sorted(unexpected_keys)}"
            )

        # Determine target keys to transfer
        target_keys = module_param_names if strict else (module_param_names & model_tensor_names)

        # Pre-validate all tensor shapes prior to making any modifications
        shape_mismatches = []
        for name in sorted(target_keys):
            desc = weight_model.inspect(name)
            expected_shape = tuple(module_state[name].shape)
            if tuple(desc.shape) != expected_shape:
                shape_mismatches.append(
                    f"tensor '{name}': model has {tuple(desc.shape)}, module expects {expected_shape}"
                )

        if shape_mismatches:
            raise ModuleMismatchError(
                "Shape mismatch during validation: " + "; ".join(shape_mismatches)
            )

        # Pre-validation passed: Stage all converted tensors
        staged_state: Dict[str, Any] = {}
        for name, param in module_state.items():
            if name in target_keys:
                arr = weight_model.get_tensor(name)
                t = torch.from_numpy(arr).to(device=param.device, dtype=param.dtype)
                staged_state[name] = t
            elif not strict:
                staged_state[name] = param

        # Snapshot original state to guarantee non-destructive atomicity on internal failure
        original_state = {k: v.clone() for k, v in module.state_dict().items()}
        try:
            module.load_state_dict(staged_state, strict=strict)
        except Exception:
            module.load_state_dict(original_state, strict=False)
            raise

        return module
