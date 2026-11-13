"""Thread-safe per-model TensorRegistry for managing TensorDescriptors."""

from __future__ import annotations

import threading
from typing import Dict, Iterator, List

from .descriptor import TensorDescriptor
from .exceptions import DuplicateTensorError, TensorNotFoundError


class TensorRegistry:
    """Thread-safe registry of TensorDescriptors belonging to a specific WeightModel.

    Guarantees isolation between models and prevents accidental duplicate
    registrations or inconsistent concurrent state.
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._lock = threading.RLock()
        self._descriptors: Dict[str, TensorDescriptor] = {}

    @property
    def model_id(self) -> str:
        """The identifier of the model this registry belongs to."""
        return self._model_id

    def register(self, descriptor: TensorDescriptor) -> None:
        """Register a new TensorDescriptor in this registry.

        Raises:
            DuplicateTensorError: If a tensor with the same name is already registered.
            TypeError: If descriptor is not an instance of TensorDescriptor.
        """
        if not isinstance(descriptor, TensorDescriptor):
            raise TypeError(f"Expected TensorDescriptor, got {type(descriptor)}")

        with self._lock:
            if descriptor.name in self._descriptors:
                raise DuplicateTensorError(
                    f"Tensor '{descriptor.name}' is already registered in model '{self._model_id}'"
                )
            self._descriptors[descriptor.name] = descriptor

    def get(self, name: str) -> TensorDescriptor:
        """Retrieve a TensorDescriptor by tensor name.

        Raises:
            TensorNotFoundError: If no tensor with the given name exists.
        """
        with self._lock:
            if name not in self._descriptors:
                raise TensorNotFoundError(
                    f"Tensor '{name}' not found in registry for model '{self._model_id}'"
                )
            return self._descriptors[name]

    def list(self) -> List[TensorDescriptor]:
        """Return a copy of all registered TensorDescriptors."""
        with self._lock:
            return list(self._descriptors.values())

    def names(self) -> List[str]:
        """Return a list of all registered tensor names."""
        with self._lock:
            return list(self._descriptors.keys())

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._descriptors

    def __len__(self) -> int:
        with self._lock:
            return len(self._descriptors)

    def __iter__(self) -> Iterator[TensorDescriptor]:
        with self._lock:
            return iter(list(self._descriptors.values()))

    def __repr__(self) -> str:
        return f"TensorRegistry(model_id='{self._model_id}', count={len(self)})"

