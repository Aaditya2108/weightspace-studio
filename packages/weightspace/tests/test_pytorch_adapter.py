"""Tests for PyTorchAdapter."""

import numpy as np
import pytest

from weightspace.adapters.pytorch import PyTorchAdapter
from weightspace.core.exceptions import ModuleMismatchError
from weightspace.core.model import WeightModel

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

    state = net.state_dict()
    for name, expected_arr in tensors.items():
        np.testing.assert_allclose(state[name].cpu().numpy(), expected_arr, atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not available in environment")
def test_pytorch_adapter_shape_mismatch_preserves_target_state():
    """Regression test: Target module must remain completely untouched on shape validation failure."""
    class TwoLayerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 2)

    net = TwoLayerNet()

    # Snapshot exact original weights before attempt
    orig_fc1_weight = net.fc1.weight.clone()
    orig_fc1_bias = net.fc1.bias.clone()
    orig_fc2_weight = net.fc2.weight.clone()
    orig_fc2_bias = net.fc2.bias.clone()

    # Create model where fc1 has valid shapes, but fc2.weight has an invalid shape
    tensors = {
        "fc1.weight": np.ones((8, 4), dtype=np.float32) * 42.0,  # Valid shape
        "fc1.bias": np.ones((8,), dtype=np.float32) * 42.0,      # Valid shape
        "fc2.weight": np.ones((5, 5), dtype=np.float32),          # Invalid shape (expects (2, 8))
        "fc2.bias": np.ones((2,), dtype=np.float32) * 42.0,      # Valid shape
    }

    weight_model = WeightModel(tensors=tensors)

    with pytest.raises(ModuleMismatchError) as exc_info:
        PyTorchAdapter.load_into_module(weight_model, net, strict=True)

    assert "Shape mismatch" in str(exc_info.value)

    # CRITICAL: Verify fc1 was NOT partially modified, and all parameters match original exactly
    assert torch.equal(net.fc1.weight, orig_fc1_weight)
    assert torch.equal(net.fc1.bias, orig_fc1_bias)
    assert torch.equal(net.fc2.weight, orig_fc2_weight)
    assert torch.equal(net.fc2.bias, orig_fc2_bias)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not available in environment")
def test_pytorch_adapter_key_mismatch_strict_preserves_target_state():
    """Regression test: Target module must remain untouched on strict key mismatch."""
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 8)

    net = SimpleNet()
    orig_weight = net.fc.weight.clone()
    orig_bias = net.fc.bias.clone()

    # Missing fc.bias in checkpoint
    tensors = {
        "fc.weight": np.ones((8, 4), dtype=np.float32) * 7.0,
    }
    weight_model = WeightModel(tensors=tensors)

    with pytest.raises(ModuleMismatchError) as exc_info:
        PyTorchAdapter.load_into_module(weight_model, net, strict=True)

    assert "missing=['fc.bias']" in str(exc_info.value)
    assert torch.equal(net.fc.weight, orig_weight)
    assert torch.equal(net.fc.bias, orig_bias)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not available in environment")
def test_pytorch_adapter_non_strict():
    """Verify non-strict mode loads matching keys and leaves unreferenced keys untouched."""
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 2)

    net = Net()
    orig_fc2_weight = net.fc2.weight.clone()
    orig_fc2_bias = net.fc2.bias.clone()

    partial_tensors = {
        "fc1.weight": np.ones((8, 4), dtype=np.float32) * 5.0,
        "fc1.bias": np.ones((8,), dtype=np.float32) * 5.0,
    }

    weight_model = WeightModel(tensors=partial_tensors)
    PyTorchAdapter.load_into_module(weight_model, net, strict=False)

    np.testing.assert_allclose(net.fc1.weight.cpu().detach().numpy(), np.ones((8, 4), dtype=np.float32) * 5.0)
    np.testing.assert_allclose(net.fc1.bias.cpu().detach().numpy(), np.ones((8,), dtype=np.float32) * 5.0)
    assert torch.equal(net.fc2.weight, orig_fc2_weight)
    assert torch.equal(net.fc2.bias, orig_fc2_bias)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not available in environment")
def test_pytorch_adapter_type_error():
    tensors = {"w": np.zeros((2, 2), dtype=np.float32)}
    weight_model = WeightModel(tensors=tensors)
    with pytest.raises(TypeError):
        PyTorchAdapter.load_into_module(weight_model, "not_a_module")  # type: ignore
