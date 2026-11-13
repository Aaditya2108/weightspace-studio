"""Tests for PyTorchAdapter."""

import numpy as np
import pytest

from weightspace.adapters.pytorch import PyTorchAdapter
from weightspace.core.exceptions import ModuleMismatchError
from weightspace.core.model import WeightModel

# Check if torch is available in runtime
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not available in environment")
def test_pytorch_adapter_successful_load():
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 2)

    net = SimpleNet()

    rng = np.random.default_rng(123)
    tensors = {
        "fc1.weight": rng.standard_normal((8, 4)).astype(np.float32),
        "fc1.bias": rng.standard_normal((8,)).astype(np.float32),
        "fc2.weight": rng.standard_normal((2, 8)).astype(np.float32),
        "fc2.bias": rng.standard_normal((2,)).astype(np.float32),
    }

    weight_model = WeightModel(tensors=tensors)
    loaded_module = PyTorchAdapter.load_into_module(weight_model, net, strict=True)

    assert loaded_module is net

    # Validate weights transferred accurately
    state = net.state_dict()
    for name, expected_arr in tensors.items():
        np.testing.assert_allclose(state[name].cpu().numpy(), expected_arr, atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not available in environment")
def test_pytorch_adapter_shape_mismatch():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 8)

    net = Net()
    bad_tensors = {
        "fc.weight": np.zeros((8, 10), dtype=np.float32),  # Expected (8, 4)
        "fc.bias": np.zeros((8,), dtype=np.float32),
    }

    weight_model = WeightModel(tensors=bad_tensors)
    with pytest.raises(ModuleMismatchError) as exc_info:
        PyTorchAdapter.load_into_module(weight_model, net, strict=True)
    assert "Shape mismatch" in str(exc_info.value)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not available in environment")
def test_pytorch_adapter_key_mismatch_strict():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 2)

    net = Net()
    incomplete_tensors = {
        "fc1.weight": np.zeros((8, 4), dtype=np.float32),
        "fc1.bias": np.zeros((8,), dtype=np.float32),
    }

    weight_model = WeightModel(tensors=incomplete_tensors)
    with pytest.raises(ModuleMismatchError) as exc_info:
        PyTorchAdapter.load_into_module(weight_model, net, strict=True)
    assert "Mismatch loading weights into module" in str(exc_info.value)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not available in environment")
def test_pytorch_adapter_non_strict():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 2)

    net = Net()
    initial_fc2_weight = net.fc2.weight.clone()

    partial_tensors = {
        "fc1.weight": np.ones((8, 4), dtype=np.float32),
        "fc1.bias": np.ones((8,), dtype=np.float32),
    }

    weight_model = WeightModel(tensors=partial_tensors)
    PyTorchAdapter.load_into_module(weight_model, net, strict=False)

    np.testing.assert_allclose(net.fc1.weight.cpu().detach().numpy(), np.ones((8, 4), dtype=np.float32))
    np.testing.assert_allclose(net.fc2.weight.cpu().detach().numpy(), initial_fc2_weight.cpu().detach().numpy())


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not available in environment")
def test_pytorch_adapter_type_error():
    tensors = {"w": np.zeros((2, 2), dtype=np.float32)}
    weight_model = WeightModel(tensors=tensors)
    with pytest.raises(TypeError):
        PyTorchAdapter.load_into_module(weight_model, "not_a_module")  # type: ignore

