# WeightSpace

WeightSpace is a developer framework for **Weight-Space Programming (WSP)**, enabling programmers to directly inspect, construct, manipulate, analyze, and execute neural network weights as first-class software artifacts.

## Installation

```bash
# Core package (NumPy, Pydantic, Safetensors)
pip install -e .

# With optional PyTorch integration and test suite
pip install -e ".[test,torch]"
```

## Running Tests

```bash
pytest -q
```

## Quick Start

```python
import weightspace as wsp

# Load a checkpoint
model = wsp.WeightModel.load("path/to/checkpoint.safetensors")

# Summary of model architecture and tensors
print(model.summary())

# Inspect a tensor descriptor (immutable, with wsp-sha256-v1 checksum)
desc = model.inspect("encoder.weight")
print(desc.shape, desc.dtype, desc.checksum)

# Optional PyTorch loading into explicit nn.Module
if wsp.PyTorchAdapter.is_available():
    import torch.nn as nn
    module = nn.Linear(16, 32)
    wsp.PyTorchAdapter.load_into_module(model, module, strict=False)
```

## Core Principles & Limitations

- **Storage**: In Milestone 1, checkpoint tensors are loaded as in-memory NumPy arrays. Copy-on-write and memory-mapping are planned for Milestone 2.
- **Immutability**: Tensor descriptors are frozen and cannot be modified. Original tensors are stored read-only, and accessor methods return defensive copies to prevent in-place mutation.
- **Checksums**: Uses `wsp-sha256-v1` which incorporates canonical dtype, NumPy byte-order, shape, and raw contiguous bytes. Supports PyTorch `bfloat16` without requiring NumPy conversion.
- **PyTorch Integration**: Optional and strictly pre-validated. Validation failures cannot leave the target module partially modified.
