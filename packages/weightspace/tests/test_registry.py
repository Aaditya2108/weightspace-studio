"""Tests for per-model thread-safe TensorRegistry."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import pytest

from weightspace.core.descriptor import TensorDescriptor
from weightspace.core.exceptions import DuplicateTensorError, TensorNotFoundError
from weightspace.core.registry import TensorRegistry


def make_desc(name: str) -> TensorDescriptor:
    valid_checksum = hashlib.sha256(f"seed_{name}".encode("utf-8")).hexdigest()
    return TensorDescriptor(
        name=name,
        shape=(16, 16),
        dtype="float32",
        checksum=valid_checksum,
    )


def test_registry_basic_operations():
    registry = TensorRegistry(model_id="model-alpha")
    assert registry.model_id == "model-alpha"
    assert len(registry) == 0

    desc1 = make_desc("layer1.weight")
    desc2 = make_desc("layer1.bias")

    registry.register(desc1)
    registry.register(desc2)

    assert len(registry) == 2
    assert "layer1.weight" in registry
    assert "layer1.bias" in registry
    assert "nonexistent" not in registry

    assert registry.get("layer1.weight") == desc1
    assert registry.get("layer1.bias") == desc2

    names = registry.names()
    assert "layer1.weight" in names
    assert "layer1.bias" in names

    all_descs = registry.list()
    assert len(all_descs) == 2


def test_registry_not_found():
    registry = TensorRegistry(model_id="test-model")
    with pytest.raises(TensorNotFoundError) as exc_info:
        registry.get("missing.weight")
    assert "missing.weight" in str(exc_info.value)
    assert "test-model" in str(exc_info.value)


def test_registry_duplicate_registration():
    registry = TensorRegistry(model_id="test-model")
    desc = make_desc("shared.weight")
    registry.register(desc)

    with pytest.raises(DuplicateTensorError) as exc_info:
        registry.register(desc)
    assert "shared.weight" in str(exc_info.value)
    assert "already registered" in str(exc_info.value)


def test_registry_type_error():
    registry = TensorRegistry(model_id="test-model")
    with pytest.raises(TypeError):
        registry.register("not_a_descriptor")  # type: ignore


def test_registry_thread_safety():
    """Verify thread-safe concurrent registrations from multiple threads."""
    registry = TensorRegistry(model_id="concurrent-model")
    n_threads = 8
    items_per_thread = 50

    def register_batch(thread_idx: int):
        for i in range(items_per_thread):
            desc = make_desc(f"t_{thread_idx}_{i}")
            registry.register(desc)

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(register_batch, t) for t in range(n_threads)]
        for f in futures:
            f.result()

    assert len(registry) == n_threads * items_per_thread
    assert len(registry.list()) == n_threads * items_per_thread
