# Milestone 2 — Weight Editing Engine

Status: Ready for implementation planning
Target branch: `feat/milestone-2-editing-engine` (create from `main` after this specification is reviewed)
Prerequisite: Milestone 1 merged; current main includes the reviewed WeightModel, TensorDescriptor, TensorRegistry, Safetensors loader, and PyTorch adapter.

## 1. Objective

Build the first safe, usable weight-space editing engine on top of the existing WeightSpace Core. A developer must be able to select tensors, apply deterministic mathematical transformations, inspect the candidate state, compare it with the original checkpoint, revert changes, and export a new Safetensors checkpoint.

The engine must never mutate the original `WeightModel`. Editing happens in an isolated `EditSession` using copy-on-write overlays and immutable snapshots.

Milestone 2 is a Python-framework milestone. Do not build the Visual IDE, WSP parser/compiler, AI agent, model evaluation framework, training loop, or architecture reconstruction in this milestone.

## 2. Design Principles and Non-Goals

### Required principles
- Original checkpoint tensors are immutable from the editor's perspective.
- The editor must not duplicate the entire model when an edit session starts.
- Only edited tensors are stored in the session overlay.
- Operations are deterministic, shape-preserving, and non-mutating.
- A failed operation or transaction must not partially publish changes.
- Snapshots must be cheap: copy the overlay mapping, not all underlying tensor data.
- Export must create a separate checkpoint and must not overwrite an existing file unless explicitly requested.
- Operations must validate targets, shapes, dtypes, values, and numerical results before publishing changes.
- Every committed edit must have an inspectable operation record.

### Out of scope
- Changing tensor shape, tensor rank, module topology, or model architecture.
- Reconstructing a PyTorch module from checkpoint names.
- Direct in-place mutation of `WeightModel`.
- Gradient descent, optimizer steps, or training.
- GPU-native editing and non-NumPy backends. Preserve a backend abstraction seam, but implement NumPy only in this milestone.
- DSL parsing, compiler IR, GUI, web API, LLM agent, and automatic claims that an edit improves model quality.

## 3. Current Core Integration

Use the merged Milestone 1 implementation as the source of truth. Do not rewrite `WeightModel`, `TensorDescriptor`, `TensorRegistry`, or the Safetensors loader without a concrete compatibility reason.

Current constraints to respect:
- `WeightModel` stores NumPy arrays and returns defensive copies from `get_tensor(name)`.
- Internal model arrays are read-only.
- `TensorDescriptor` is frozen and uses the versioned `wsp-sha256-v1` checksum.
- `TensorRegistry` is isolated per model and thread-safe.
- PyTorch is optional and must remain lazily imported.
- The repository supports Python 3.10+ and root-level editable installation.

Recommended integration: add a private internal read-only tensor-view accessor to `WeightModel` for the engine only, such as `_get_tensor_view(name)`. It must return the existing internal array with `writeable=False`, never a mutable reference. Keep the existing public `get_tensor()` defensive-copy behaviour unchanged. If a private view accessor is not added, document and test the extra copy cost of using `get_tensor()`.

## 4. Package Layout

Add the following modules under `packages/weightspace/src/weightspace/`:

```text
core/
  selection.py                 # TensorSelection and deterministic selector resolution
operations/
  __init__.py
  base.py                       # Operation protocol/base and shared validation
  arithmetic.py                 # scale, add, subtract, multiply
  normalization.py              # normalize
  masking.py                    # mask
  slicing.py                    # replace_slice
  registry.py                   # operation registry (only if useful; do not over-engineer)
engine/
  __init__.py
  editor.py                     # EditSession and overlay management
  transaction.py                # atomic transaction context manager
  snapshot.py                   # immutable session snapshots
  diff.py                       # tensor diff records and summaries
io/
  safetensors_exporter.py       # atomic Safetensors export
core/exceptions.py              # extend existing domain exception hierarchy
__init__.py                     # export stable public API
```

Tests belong under `packages/weightspace/tests/`, grouped by feature. Update root and package docs where required; keep the two pyproject files consistent.

## 5. Core Concepts and Public API

### 5.1 EditSession

`EditSession` represents an isolated candidate edit state derived from one `WeightModel`.

Proposed usage:

```python
import weightspace as wsp

model = wsp.WeightModel.load('model.safetensors')
session = wsp.EditSession(model)

session.scale('encoder.weight', factor=0.95)
session.add('encoder.bias', value=0.01)
session.normalize('decoder.weight', method='l2', axis=1)
session.clamp('decoder.weight', min_value=-2.0, max_value=2.0)

preview = session.get_tensor('encoder.weight')
report = session.diff()
session.export('modified_model.safetensors')
```

Required public methods:
- `select(pattern: str) -> TensorSelection`
- `inspect(target: str) -> TensorDescriptor`
- `get_tensor(target: str) -> numpy.ndarray` (always return a defensive copy)
- `scale(target, factor)`
- `add(target, value)`
- `subtract(target, value)`
- `multiply(target, value)`
- `normalize(target, method='l2', axis=None, eps=1e-12)`
- `clamp(target, min_value=None, max_value=None)`
- `mask(target, mask, fill_value=0)`
- `replace_slice(target, index, values, broadcast=False)`
- `transaction(name: str)` context manager
- `snapshot(name: str | None = None) -> ModelSnapshot`
- `restore(snapshot: ModelSnapshot)`
- `rollback()` to discard all session edits and return to the base model state
- `diff() -> ModelDiff`
- `export(path, overwrite=False) -> ExportReport`
- read-only properties: `model_id`, `base_model`, `edited_tensor_names`, `operation_history`

Do not add a `commit()` method that mutates the source `WeightModel`. A successful transaction publishes changes only into the `EditSession` overlay. Export creates a new checkpoint.

### 5.2 TensorSelection

`session.select(pattern)` uses case-sensitive shell-style glob matching (`fnmatchcase`) against registered tensor names. It returns an immutable `TensorSelection` containing sorted, unique tensor names and the originating `model_id`.

Rules:
- Exact string targets in edit methods refer to exactly one tensor name; do not silently interpret a typo as an empty wildcard selection.
- A `TensorSelection` may be passed as the target to any operation to apply that operation to all selected tensors.
- A sequence of exact tensor names may optionally be supported; resolve and validate the entire sequence before changing anything.
- Empty selections raise a dedicated `TensorSelectionError`.
- A selection created for another model/session must be rejected.
- Apply selected tensors in deterministic lexicographic order.

Example:

```python
weights = session.select('encoder.*.weight')
session.scale(weights, factor=0.9)
```

### 5.3 Tensor and dtype support

Milestone 2 implements a NumPy backend. The first operation set supports real floating-point tensors only (`float16`, `float32`, `float64`, and any additional NumPy floating dtype that can be handled safely).

Integer, boolean, complex, object, string, and unsupported/quantized arrays must remain loadable and exportable when unedited, but arithmetic editing operations must reject them with a clear `UnsupportedTensorDtypeError`. Do not silently cast integer tensors to float or truncate floating values into integer tensors.

All edit operations preserve the target tensor's original shape and dtype. Do not expose dtype conversion in this milestone.

### 5.4 Operation operand and broadcasting policy

Arithmetic operands may be real finite scalars or NumPy arrays. Array operands must be broadcastable to the target tensor shape using NumPy broadcasting rules. Validate broadcasting before publishing the operation.

All operations must:
- leave their input arrays unchanged;
- return a new array with exactly the original target shape and dtype;
- reject invalid operands and incompatible shapes with domain-specific errors;
- reject non-finite scalar/array operands;
- reject results containing NaN or infinity;
- never silently overflow to infinity;
- preserve the base tensor if validation or computation fails.

Use a deliberate accumulator dtype for numerical stability (at least float32 for float16 input; float64 may be used for reductions). Cast the final result back to the original dtype only after checking that the cast is valid and finite. Document any rounding that is inherent to the target dtype.

## 6. Operation Semantics

Implement these operations with explicit, tested mathematical definitions.

### 6.1 scale

`scale(x, factor)` computes `x * factor` element-wise. `factor` must be a finite real scalar. Negative and zero factors are allowed.

### 6.2 add

`add(x, value)` computes `x + value` element-wise, using the scalar/array broadcasting rules above.

### 6.3 subtract

`subtract(x, value)` computes `x - value` element-wise.

### 6.4 multiply

`multiply(x, value)` computes `x * value` element-wise.

### 6.5 normalize

Support two methods initially:
- `method='l2'`: divide by the L2 norm along `axis`. When `axis=None`, normalize the whole tensor. For a zero-norm slice, return zeros for that slice rather than NaN.
- `method='zscore'`: `(x - mean) / std` along `axis`. When a slice has standard deviation less than or equal to `eps`, return zeros for that slice.

`eps` must be finite and strictly positive. Accept `axis=None`, one integer axis, or a tuple of axes; validate axis ranges and duplicates. Keep dimensions during reduction so broadcasting is deterministic. Do not add L1 or min-max normalization unless it is fully specified and tested.

### 6.6 clamp

`clamp(x, min_value=None, max_value=None)` clips values to the supplied inclusive bounds. At least one bound is required. If both are supplied, require `min_value <= max_value`. Bounds must be finite real scalars.

### 6.7 mask

`mask(x, mask, fill_value=0)` returns a tensor where positions with `mask == True` are replaced by `fill_value`; positions with `False` retain their original value. The mask must have boolean dtype and be broadcastable to `x.shape`. `fill_value` must be finite and safely representable in the target dtype.

### 6.8 replace_slice

`replace_slice(x, index, values, broadcast=False)` replaces a selected region without changing the full tensor shape.

Rules:
- `index` is a tuple of basic Python `slice` objects and/or integer indices; support normal NumPy indexing semantics.
- Reject advanced/fancy index arrays and zero slice steps.
- Resolve the selected region and its resulting shape before publishing.
- By default, `values` must have exactly the selected region's shape.
- With `broadcast=True`, permit NumPy broadcasting only into that selected region.
- Values must be real, finite, and safely representable in the target dtype.
- Do not permit this method to resize or reshape the full tensor.

## 7. Copy-on-Write Overlay

`EditSession` holds:
- an immutable reference to the base `WeightModel`;
- an overlay mapping `{tensor_name: read_only_numpy_array}` containing only tensors edited in the session;
- operation history for successfully committed edits;
- a lock protecting session state;
- snapshot metadata.

Read resolution order:
1. If a tensor exists in the current transaction's staged overlay, read that version.
2. Else if it exists in the session overlay, read that version.
3. Else read a read-only view from the base `WeightModel`.

On first edit of a tensor, compute a new result from the base read-only view and store only that result in the overlay. Subsequent edits must compute a new result from the latest overlay value; do not mutate arrays already referenced by snapshots.

Every array stored in the committed overlay or a snapshot must be marked `writeable=False`. Operation implementations must be pure: they return new arrays and never mutate an input.

Starting a session and taking a snapshot must not copy every tensor. The `edited_tensor_names` property should expose only names present in the overlay.

Do not add a public mutable tensor reference. `EditSession.get_tensor()` must return a copy, just like `WeightModel.get_tensor()`.

## 8. Transactions and Atomicity

Support explicit transaction blocks:

```python
with session.transaction('prepare_encoder') as edit:
    edit.scale('encoder.weight', factor=0.95)
    edit.clamp('encoder.weight', min_value=-3.0, max_value=3.0)
    edit.add('encoder.bias', value=0.02)
```

Transaction behaviour:
- Entering a transaction creates a staged overlay mapping that initially references the current immutable overlay arrays; it must not copy tensor payloads.
- Every operation inside the transaction reads its latest staged value.
- When the block exits successfully, validate the completed staged state and atomically publish the staged mapping to the session overlay. Record the operations as committed history.
- If any operation or the transaction body raises an exception, discard the entire staged overlay and operation records. The pre-transaction session state remains unchanged.
- A transaction affecting multiple target tensors is all-or-nothing.
- Nested transactions are not supported in Milestone 2; raise a clear `TransactionError` if attempted.
- A single operation invoked outside an explicit transaction must execute as its own implicit transaction, so it is also atomic.
- Protect transaction state with a session `RLock`; do not hold locks while doing unrelated I/O such as checkpoint export.

Clarify in docs: transaction commit means publishing a set of edits into the session overlay, not changing the original `WeightModel`.

## 9. Snapshots and Rollback

`ModelSnapshot` is an immutable, session-bound snapshot containing:
- snapshot ID (UUID);
- source `model_id`;
- optional user-provided name;
- creation timestamp in UTC;
- immutable overlay mapping (mapping copy, shared read-only arrays);
- operation-history length or immutable history prefix.

`session.snapshot(name=None)` returns a snapshot cheaply without copying the base model or tensor payloads.

`session.restore(snapshot)` replaces the current overlay and visible operation history with the snapshot state. Reject snapshots from a different source model. Prefer rejecting snapshots from a different session even if they share the same model, unless an explicit cross-session restore contract is implemented and tested.

`session.rollback()` clears the overlay and all current session edits, returning to the base model state. It must not change the base model. Document the difference between transaction rollback (automatic on exception) and full-session rollback.

## 10. Operation History and Provenance

Add an immutable `OperationRecord` for each operation published by a successful transaction. Include:
- sequential index;
- transaction ID and optional transaction name;
- operation name;
- resolved target tensor names;
- JSON-serializable parameters;
- before and after checksums for each changed tensor;
- timestamp in UTC.

Do not store full tensor payloads in the operation history. For a mask or large array operand, record operand shape, dtype, and checksum instead of embedding its contents. Keep history deterministic and suitable for future export metadata, but do not add a full experiment database.

Failed transactions must not appear in committed operation history.

## 11. Tensor Diffs

Expose `session.diff()` returning an immutable `ModelDiff` with per-tensor `TensorDiff` entries for tensors changed relative to the original model.

Each tensor diff should include:
- tensor name;
- original and current shape (identical in this milestone);
- original and current dtype;
- original and current `wsp-sha256-v1` checksum;
- changed element count (exact comparison);
- total element count;
- changed ratio;
- maximum absolute difference;
- mean absolute difference;
- L2 norm of the difference.

Rules:
- Only compare edited tensors by default; unchanged tensors need not be scanned.
- Use exact element comparison for `changed_count`; treat corresponding NaNs consistently if encountered in original data, but edited results must be finite.
- Compute numeric difference statistics using a safe accumulator dtype to avoid overflow for float16 inputs.
- Do not return full difference arrays by default. An optional helper may return a defensive copy for one named tensor if needed.
- The diff summary should include total edited tensors and total changed elements.

## 12. Safetensors Export

Implement `safetensors_exporter.py` and `session.export(path, overwrite=False)`.

Requirements:
- Export every tensor: overlay version for edited tensors and original base version for untouched tensors.
- Preserve all source file metadata when possible.
- Add namespaced provenance metadata such as `weightspace.format_version`, `weightspace.source_model_id`, and `weightspace.checksum_version` without overwriting unrelated user metadata.
- If a reserved WeightSpace metadata key already exists, define deterministic behaviour (prefer replace only the WeightSpace-owned namespace and document it).
- Safetensors metadata values must be strings.
- Ensure each exported array is contiguous and supported by the NumPy Safetensors backend.
- Reject unsupported dtypes with a clear export error rather than silently converting data.
- By default, refuse to overwrite an existing destination.
- For `overwrite=True`, write to a temporary file in the destination directory, flush/close it, then atomically replace the target with `os.replace`.
- Clean up temporary files on failure.
- Never modify the source checkpoint.
- Reload the exported checkpoint using the existing loader and verify tensor names, shapes, dtypes, checksums, and metadata before reporting success.

`ExportReport` should include destination path, number of tensors, number of edited tensors, total tensor bytes, and relevant provenance/checksum version.

## 13. Exceptions

Extend `core/exceptions.py` with a small, meaningful hierarchy. Suggested types:
- `EditingError(WeightSpaceError)`
- `TensorSelectionError(EditingError)`
- `UnsupportedTensorDtypeError(EditingError)`
- `OperationValidationError(EditingError)`
- `TransactionError(EditingError)`
- `SnapshotError(EditingError)`
- `ExportError(ModelLoadError)` or a dedicated export branch under `WeightSpaceError`.

Use these consistently. Do not leak low-level NumPy broadcast/index errors without context; chain them as causes where appropriate.

## 14. Public Imports

Export stable public objects from `weightspace.__init__` after implementation:
- `EditSession`
- `TensorSelection`
- `ModelSnapshot`
- `OperationRecord`
- `TensorDiff` and `ModelDiff`
- `ExportReport`
- the new public exception types.

Keep internal operation implementation classes private unless a concrete user-facing use case requires exposing them.

## 15. Testing Requirements

Add comprehensive tests. At minimum:

### Selection
- exact name selection;
- glob selection;
- deterministic sorted results;
- empty selection error;
- selection from another model/session rejected.

### Operations
- correct numerical results for scale, add, subtract, multiply;
- scalar and broadcastable array operands;
- incompatible shapes rejected;
- non-finite operands rejected;
- overflow/non-finite results rejected;
- integer/bool/complex tensor edits rejected;
- input arrays are never mutated;
- output shape and dtype preserved;
- L2 normalization across whole tensor and axes;
- zero-vector L2 normalization produces zeros;
- z-score normalization and constant-slice behaviour;
- invalid axes and invalid epsilon rejected;
- clamp lower-only, upper-only, both bounds, and invalid ranges;
- mask semantics, bool validation, broadcasting, fill value validation;
- slice replacement, scalar index, broadcast opt-in, shape mismatch, invalid index, and full shape preservation;
- multi-target operations succeed for all selected targets or publish none.

### Copy-on-write and original preservation
- starting an edit session creates an empty overlay;
- editing one tensor adds only that tensor to the overlay;
- base model tensors and checksums remain unchanged after edits;
- `get_tensor()` returns a defensive copy;
- untouched tensors are not duplicated into the overlay;
- overlay arrays are read-only.

### Transactions
- successful transaction publishes all edits;
- exception in a transaction discards every staged edit;
- failure on the second target does not publish the first target's result;
- failed single operation leaves session state unchanged;
- nested transaction rejected;
- operation history includes committed operations only.

### Snapshots and rollback
- snapshot restores a previous session state;
- snapshots share immutable arrays and do not mutate when later edits occur;
- cross-model and cross-session snapshots rejected according to the documented rule;
- full-session rollback clears edits but preserves the source model.

### Diffs
- correct changed count and ratio;
- correct max/mean absolute and L2 difference;
- no entry for untouched tensors;
- zero-change result for exact no-op transformations.

### Export
- export and reload preserves all tensor names, shapes, dtypes, and checksums;
- source metadata preserved;
- WeightSpace provenance metadata added;
- destination not overwritten by default;
- explicit overwrite works;
- failed export cleans up temporary files and leaves the original destination intact.

Use small deterministic test arrays. Do not download real model checkpoints in unit tests. Keep test execution practical on CI.

## 16. Performance and Safety Requirements

- Never create a full duplicate of the model just to open a session or take a snapshot.
- Store only edited tensors in the overlay.
- Avoid repeated defensive copies internally; use a private read-only base view for pure engine operations where safe.
- Use `np.broadcast_to` for operand validation, but do not materialize broadcast arrays unnecessarily.
- Check finite values and numeric dtype semantics before publishing.
- Bound error messages; never print entire large tensor values.
- Do not execute checkpoint-supplied code or pickle files.
- Do not add network access, telemetry, or automatic upload.

## 17. Documentation and Examples

Update `packages/weightspace/README.md` and `docs/architecture.md` with:
- edit session lifecycle;
- operation semantics and dtype policy;
- copy-on-write guarantees and limitations;
- transaction and snapshot examples;
- diff usage;
- safe export and overwrite behaviour;
- limitations: NumPy-backed host-memory editing, supported real floating dtypes, shape-preserving operations only, no architecture validation or model-quality evaluation.

Add at least one runnable example under `examples/` that:
1. creates a small deterministic NumPy tensor mapping or tiny Safetensors checkpoint;
2. loads it through `WeightModel`;
3. opens an `EditSession`;
4. scales and clamps a selected tensor inside a transaction;
5. creates a snapshot and demonstrates restore;
6. prints a diff;
7. exports a new Safetensors file and reloads it.

Do not make MNIST training or model benchmarking part of Milestone 2. Those belong in a later end-to-end demonstration milestone.

## 18. Acceptance Criteria

Milestone 2 is complete only when all of the following are true:

1. A developer can open an `EditSession` from an existing `WeightModel`.
2. Exact and glob tensor selection work deterministically.
3. All eight operations in Section 6 work with the documented semantics.
4. All operations preserve tensor shape and dtype.
5. Original model arrays and checksums remain unchanged.
6. The overlay contains only edited tensors.
7. Single operations and explicit multi-operation transactions are atomic.
8. Snapshots and rollback work without copying the entire model.
9. Operation history records committed changes only.
10. Diff reports exact changed counts and numerical summaries.
11. Export writes a new Safetensors checkpoint with preserved metadata and WeightSpace provenance.
12. Exported checkpoints reload correctly and pass checksum verification.
13. All new unit tests pass on supported CI environments, including the no-PyTorch job.
14. Root editable installation and package-level editable installation continue to work.
15. Documentation and the example are complete.

## 19. Suggested Implementation Sequence

Implement in small commits, in this order:

1. Review the existing merged code and produce a brief implementation plan. Do not code before the plan is reviewed.
2. Add domain exceptions and `TensorSelection` with tests.
3. Implement pure operation functions and tests, independent of `EditSession`.
4. Implement immutable overlay storage and `EditSession` read/inspect/select methods.
5. Add atomic single-operation application and explicit transactions.
6. Add snapshots, restore, and full-session rollback.
7. Add operation history and tensor diff reports.
8. Add atomic Safetensors export and reload verification.
9. Add integration example, update docs, run complete CI.

Suggested commit prefixes: `feat(operations): ...`, `feat(engine): ...`, `feat(io): ...`, `test(engine): ...`, `docs(milestone-2): ...`.

## 20. Instruction to Antigravity

Read this document and the current merged `main` implementation completely. First inspect the existing APIs and identify integration risks. Then prepare a detailed Milestone 2 implementation plan with proposed files, class interfaces, operation semantics, test strategy, and any necessary adjustments to this specification.

Do not start implementation until the plan has been reviewed and approved. Do not modify `main`; use a feature branch created from the current `main`. Do not begin the WSP DSL, compiler, visual IDE, AI agent, or model evaluation work in Milestone 2.

At the end of the implementation, report changed files, commit SHAs, test results, CI matrix results, memory/copy-on-write behaviour, known limitations, and exact commands to run the tests and example.