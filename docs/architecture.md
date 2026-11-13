# WeightSpace Architecture Overview

WeightSpace is an open-source framework and domain-specific environment for **Weight-Space Programming (WSP)**, enabling neural network weights to be directly inspected, constructed, manipulated, analyzed, and executed as first-class software artifacts.

---

## 1. Package Structure & `src` Layout

WeightSpace uses a standard `src` layout under `packages/weightspace/src/weightspace/` (as well as supporting repository-level editable installation):

```
weightspace-studio/
├── pyproject.toml                     # Root package configuration
├── .github/workflows/ci.yml           # Automated CI across Python versions
├── docs/
│   └── architecture.md                # Architecture specification
└── packages/weightspace/
    ├── pyproject.toml                 # Subpackage configuration
    ├── README.md                      # Package documentation
    ├── src/weightspace/
    │   ├── __init__.py
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── descriptor.py         # Immutable TensorDescriptor & wsp-sha256-v1
    │   │   ├── registry.py           # Thread-safe per-model TensorRegistry
    │   │   ├── model.py              # WeightModel abstraction & tensor preservation
    │   │   └── exceptions.py         # Domain exception hierarchy
    │   ├── io/
    │   │   ├── __init__.py
    │   │   └── safetensors.py        # SafetensorsData loader
    │   └── adapters/
    │       ├── __init__.py
    │       └── pytorch.py            # Pre-validating atomic PyTorchAdapter
    └── tests/
        ├── conftest.py
        ├── test_descriptor.py
        ├── test_registry.py
        ├── test_safetensors_io.py
        ├── test_model.py
        ├── test_pytorch_adapter.py
        └── test_edge_cases.py
```

---

## 2. Core Architectural Guarantees

### Immutable Tensor Descriptors & `wsp-sha256-v1` Checksum
Every tensor is described by an immutable, frozen Pydantic model (`TensorDescriptor` with `model_config = ConfigDict(frozen=True, extra="forbid")`).

**Checksum Specification (`wsp-sha256-v1`)**:
The cryptographic checksum is a deterministic 64-character lowercase hexadecimal SHA-256 digest computed over:
1. Version identifier: `wsp-sha256-v1\n` (ASCII)
2. Canonical dtype string: `dtype:<dtype_name>\n` (UTF-8)
3. Canonical shape string: `shape:<dim0,dim1,...>\n` (UTF-8)
4. Contiguous tensor payload bytes.

This ensures that identical byte content with differing shapes (e.g. `(4, 4)` vs `(16,)`) or differing dtypes (e.g. `float32` vs `int32`) produces strictly distinct cryptographic digests.

### Explicit Representation of Unknown Properties
To prevent brittle heuristics, semantic roles (`role`), parent modules (`module`), and parameter trainability (`trainable`) default to `None` unless explicitly declared in checkpoint metadata. Descriptors also reject empty tensor names and negative shape dimensions.

### Isolated, Thread-Safe Per-Model Registries
Each `WeightModel` instantiates an independent `TensorRegistry` guarded by a recursive lock (`threading.RLock`). This guarantees thread safety during concurrent reads/registrations and prevents cross-model naming collisions.

### Non-Destructive Storage & In-Place Protection
Tensors within `WeightModel` are marked non-writable (`flags.writeable = False`). Access via `get_tensor(name)` returns detached copies, guaranteeing that external callers cannot inadvertently mutate internal checkpoint weights.

### Clean I/O Separation
The Safetensors loader produces a clean `SafetensorsData(tensors, metadata)` data structure. It separates raw tensor arrays from optional file metadata without assuming any proprietary envelope or container dictionary.

### Atomic PyTorch Integration
`PyTorchAdapter.load_into_module(weight_model, module, strict=True)` loads weights directly into an explicitly supplied `torch.nn.Module`.
- **Pre-Validation**: All matching keys and tensor shapes are validated before modifying the target module.
- **Atomicity**: Validation failures (key mismatch or shape mismatch) leave the target module 100% untouched.
- **Strict / Non-Strict Modes**: Strict mode requires exact key matching; non-strict mode transfers matching keys while preserving unreferenced parameters.
- **Optional Dependency**: PyTorch is imported lazily. If PyTorch is not installed, `PyTorchAdapter.is_available()` returns `False` and operations raise `PyTorchNotAvailableError`.

---

## 3. Storage Limitations (Milestone 1)

In Milestone 1, checkpoint tensors are loaded as in-memory NumPy arrays. While this provides universal compatibility and high speed for standard models, it has known boundaries:
- **Host Memory Consumption**: The entire checkpoint is resident in host RAM.
- **Copy Overhead**: `get_tensor()` returns defensive NumPy array copies to enforce immutability.
- **Roadmap (Milestone 2)**: Milestone 2 will introduce memory-mapped streaming (`mmap`) and copy-on-write (CoW) tensor views to handle multi-gigabyte models without full memory replication.

---

## 4. Installation, Testing, and Usage

### Installation

```bash
# 1. Base install (NumPy, Pydantic, Safetensors)
pip install -e .

# 2. Base install with tests and PyTorch support
pip install -e ".[test,torch]"
```

### Running Tests

```bash
pytest -q
```

### Basic Usage

```python
import weightspace as wsp

# 1. Load a Safetensors checkpoint
model = wsp.WeightModel.load("model.safetensors")

# 2. Inspect model overview
print(model.summary())

# 3. Inspect a specific tensor descriptor
desc = model.inspect("layers.0.weight")
print(desc.name, desc.shape, desc.dtype, desc.checksum)

# 4. (Optional) Load parameters into a PyTorch module
if wsp.PyTorchAdapter.is_available():
    import torch.nn as nn
    my_module = nn.Sequential(...)
    wsp.PyTorchAdapter.load_into_module(model, my_module, strict=True)
```
