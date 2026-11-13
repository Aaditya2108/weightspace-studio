# WeightSpace Architecture Overview

WeightSpace is a developer framework for **Weight-Space Programming (WSP)**, enabling neural network weights to be treated as first-class, programmable software artifacts.

## Milestone 1: WeightSpace Core

### 1. `packages/weightspace/` Structure
WeightSpace employs the modern Python `src` layout:

```
packages/weightspace/
├── pyproject.toml
├── src/
│   └── weightspace/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── descriptor.py     # Immutable TensorDescriptor & SHA-256 checksum
│       │   ├── registry.py       # Thread-safe per-model TensorRegistry
│       │   ├── model.py          # WeightModel abstraction & non-destructive preservation
│       │   └── exceptions.py     # Domain exceptions hierarchy
│       ├── io/
│       │   ├── __init__.py
│       │   └── safetensors.py    # Clean Safetensors checkpoint loader
│       └── adapters/
│           ├── __init__.py
│           └── pytorch.py        # PyTorchAdapter targeting explicit torch.nn.Modules
└── tests/
    ├── conftest.py
    ├── test_descriptor.py
    ├── test_registry.py
    ├── test_safetensors_io.py
    ├── test_model.py
    ├── test_pytorch_adapter.py
    └── test_edge_cases.py
```

### 2. Key Architectural Guarantees

1. **Immutable Tensor Descriptors**: `TensorDescriptor` models are frozen via Pydantic (`frozen=True`, `extra="forbid"`). Checksums are computed as deterministic SHA-256 digests over contiguous bytes.
2. **Explicit Representation of Unknowns**: Trainability, tensor roles, and module hierarchies are left as `None` unless explicitly present in metadata. No heuristic guessing.
3. **Per-Model Isolated Registries**: Each `WeightModel` maintains its own `TensorRegistry` guarded by a recursive lock (`threading.RLock`) to guarantee thread safety and prevent cross-model leakage or naming collisions.
4. **Preservation of Original Tensors**: Checkpoint tensors are held in read-only protected structures. `get_tensor()` returns defensive copies, ensuring external callers cannot inadvertently mutate internal checkpoint parameters in place.
5. **Clean I/O Separation**: The Safetensors loader produces `SafetensorsData`, separating pure tensor data mappings from file metadata without expecting proprietary envelope structures.
6. **Explicit PyTorch Integration**: Rather than guessing arbitrary network architectures from state dict keys, `PyTorchAdapter` transfers parameters into an explicitly provided target `torch.nn.Module`, supporting strict and non-strict matching.

