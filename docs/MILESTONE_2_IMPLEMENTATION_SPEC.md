# Milestone 2 — Weight Editing Engine

Status: Final revised specification submitted for approval — implementation not yet authorized
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

Milestone 2 edits exactly these NumPy target dtypes: `np.float16`, `np.float32`, and `np.float64`. Do not silently broaden the editing policy to other floating formats. In particular, bfloat16 support in the Milestone 1 checksum/adapter path does not imply bfloat16 editing support in this milestone.

Integer, unsigned integer, boolean, complex, object, string, quantized, and otherwise unsupported tensors remain loadable and exportable when unedited, but edit operations reject them with `UnsupportedTensorDtypeError`. Never silently cast an integer target to floating point or truncate floating results into an integer target.

All edit operations preserve the target tensor's exact shape and dtype. Dtype conversion is out of scope.

### 5.4 Operand validation, broadcasting, and arithmetic safety

- Real scalar operands may be Python integers/floats and NumPy integer/floating scalar types. Reject Python `bool` and `np.bool_` explicitly, even though Python treats `bool` as a subclass of `int`.
- Array operands may use real integer, unsigned-integer, or floating dtypes. Reject boolean, complex, object, string, and other non-real arrays. Validate that every value is finite after safe conversion to the computation representation.
- Target tensors must themselves contain only finite values before an edit is attempted. An edit must not use an operation to hide or silently repair NaN or infinity already present in the target.
- Validate array broadcasting with `np.broadcast_shapes` before computation. Use broadcast views where possible; do not materialize a broadcasted operand unnecessarily.
- Convert operands and target values to a float64 computation representation for arithmetic and validation. This gives float16/float32 operations additional headroom. Float64 operations remain limited by float64's representable range; reject a mathematically unrepresentable result rather than storing infinity.
- Run arithmetic under controlled NumPy floating-point error handling. Check the accumulator result for NaN/infinity, cast to the original target dtype, then check the cast result again for NaN/infinity before publishing it.
- Overflow at either stage is an `OperationValidationError`. Never clip or saturate an overflowing result implicitly.
- Finite underflow is accepted: a value that rounds to zero or a finite subnormal in the accumulator or target dtype is treated as normal IEEE floating-point rounding. Underflow alone is not an error. NaN and infinity are always rejected; underflow does not mean NaN.
- Document the precision and rounding inherent in the original target dtype. Array integer operands larger than float64's exact-integer range are interpreted through the documented float64 computation representation; no arbitrary-precision integer arithmetic is promised.
- `mask` is the exception to the real-array operand rule for its mask argument: the mask must have boolean dtype. Its fill value follows the real finite scalar policy.
- `replace_slice` values follow the real finite scalar/array policy and must be safely convertible to the target dtype.

Every operation must leave inputs unchanged, return a new array with the exact original shape and dtype, and publish nothing until all validation succeeds. Chain low-level NumPy errors as causes of the appropriate domain exception.


## 6. Operation Semantics

Implement these operations with explicit, tested mathematical definitions. All scalar bounds, factors, fill values, and `eps` values use the finite real scalar policy in Section 5.4; booleans are not accepted as numeric scalars.

### 6.1 scale

`scale(x, factor)` computes `x * factor` element-wise. `factor` must be a finite real scalar. Negative and zero factors are allowed.

### 6.2 add

`add(x, value)` computes `x + value` element-wise, using the scalar/array broadcasting rules in Section 5.4.

### 6.3 subtract

`subtract(x, value)` computes `x - value` element-wise.

### 6.4 multiply

`multiply(x, value)` computes `x * value` element-wise.

For all four arithmetic operations, compute in the float64 computation representation, reject any non-finite accumulator result, cast back to the target dtype, and reject any non-finite cast result. Finite underflow to zero or a finite subnormal is allowed and is documented as target-dtype rounding.

### 6.5 normalize

Support two methods initially:

- `method='l2'`: normalize along `axis`. When `axis=None`, treat the entire tensor as one reduction group. For each reduction group, use a scaled sum-of-squares calculation: let `s = max(abs(x))`; for nonzero groups compute `u = x / s`, then `q = sqrt(sum(u * u))` with `keepdims=True`, and compute the normalized values as `u / q`. This avoids forming an overflowing raw norm. A zero group returns zeros. Apply `eps` without first forming a potentially overflowing norm: a group is treated as degenerate when its true norm is less than or equal to `eps`, tested using the scaled quantities without an unsafe product.
- `method='zscore'`: compute `(x - mean) / std` along `axis`, but do not directly subtract or square unscaled extreme values. For each reduction group, let `s = max(abs(x))`; use zeros for an all-zero group, otherwise compute `u = x / s`, the mean of `u` in float64, then a two-pass centered variance of `u` in float64. Compute z-scores as `(u - mean(u)) / std(u)`; the scale cancels algebraically. The true standard deviation is `s * std(u)`, which is safe to compare with `eps` because `std(u)` is bounded for `u` in [-1, 1]. A group with standard deviation less than or equal to `eps` returns zeros.

Use float64 reductions with `keepdims=True`. The scaled z-score calculation avoids overflow in mean/variance from very large magnitudes and avoids the cancellation introduced by subtracting two large unscaled values. Do not use a naïve `sum(x * x)` for L2 normalization or a naïve unscaled `mean(x * x) - mean(x)**2` variance calculation.

`eps` must be finite, real, non-boolean, and strictly positive. Accept `axis=None`, one integer axis, or a tuple of axes; validate axis ranges and duplicates. Empty tensors must return an empty result with unchanged shape and dtype after argument/axis validation. Do not add L1 or min-max normalization unless it is fully specified and tested.

After normalization, cast to the original target dtype and enforce the post-cast finiteness check from Section 5.4.

### 6.6 clamp

`clamp(x, min_value=None, max_value=None)` clips values to the supplied inclusive bounds. At least one bound is required. If both are supplied, require `min_value <= max_value`. Bounds must be finite, non-boolean real scalars. The result must pass the common post-cast validation.

### 6.7 mask

`mask(x, mask, fill_value=0)` returns a tensor where positions with `mask == True` are replaced by `fill_value`; positions with `False` retain their original value. The mask must have boolean dtype and be broadcastable to `x.shape`. `fill_value` must be a finite, non-boolean real scalar safely representable in the target dtype. Validate the complete result before publishing it.

### 6.8 replace_slice

`replace_slice(x, index, values, broadcast=False)` replaces a selected region without changing the full tensor shape.

Rules:
- `index` is a tuple of basic Python `slice` objects and/or integer indices; support normal NumPy basic-indexing semantics.
- Reject advanced/fancy index arrays, boolean indices, and zero slice steps.
- Resolve the selected region and its resulting shape before publishing.
- By default, `values` must have exactly the selected region's shape.
- With `broadcast=True`, permit NumPy broadcasting only into that selected region.
- Values must be real, finite, non-boolean, and safely convertible to the target dtype.
- Validate the full replacement result, including post-cast finiteness, before publishing.
- Do not permit this method to resize or reshape the full tensor.


## 7. Copy-on-Write Overlay

`EditSession` holds:
- an immutable reference to the base `WeightModel`;
- an overlay mapping `{tensor_name: read_only_numpy_array}` containing only tensors edited in the session;
- operation history for successfully committed edits;
- a session `threading.RLock` protecting session state;
- snapshot metadata and a unique `session_id`.

Read resolution order:
1. If a tensor exists in the current transaction's staged overlay, read that version.
2. Else if it exists in the session overlay, read that version.
3. Else read a read-only view from the base `WeightModel`.

On first edit of a tensor, compute a new result from the base read-only view and store only that result in the overlay. Subsequent edits must compute a new result from the latest overlay value; never mutate arrays already referenced by snapshots or prior transaction state.

**Owning-array immutability is mandatory.** Before an array is inserted into the committed overlay or staged overlay, its owning ndarray must be made non-writeable with `flags.writeable = False`. Do this before creating or exposing any view. Snapshot and private accessor views must be backed by these non-writeable owners. Do not expose an owning array whose writeable flag is false but whose owner can be recovered and made writable through a public reference. Internal code must treat published arrays as immutable and replace them with newly allocated results on every edit.

`WeightModel._get_tensor_view(name)` returns a distinct ndarray view of the internal base tensor with `writeable=False`; the base owning array itself is already non-writeable. Public `WeightModel.get_tensor()` and `EditSession.get_tensor()` continue to return independent defensive copies. A caller must not be able to set a returned view's writeable flag back to true because its base owner is also non-writeable.

Starting a session and taking a snapshot must not copy every tensor. The `edited_tensor_names` property exposes only names present in the overlay. Do not add a public mutable tensor reference.


## 8. Transactions and Atomicity

Support explicit transaction blocks:

```python
with session.transaction('prepare_encoder') as edit:
    edit.scale('encoder.weight', factor=0.95)
    edit.clamp('encoder.weight', min_value=-3.0, max_value=3.0)
    edit.add('encoder.bias', value=0.02)
```

Transaction behaviour:
- Entering a transaction acquires the session `threading.RLock` and holds it continuously until the context exits. Record the owner thread using `threading.get_ident()`.
- The lock is re-entrant so operations made by the owning thread inside the transaction can safely enter session methods. Other threads attempting edits or transactions block until the owner releases the lock; they cannot join or modify the active transaction.
- Nested transactions are not supported; an attempt by the owner thread to start one raises `TransactionError`.
- Entering a transaction creates a staged overlay mapping that initially references the current immutable overlay arrays; it must not copy tensor payloads.
- Every operation inside the transaction reads its latest staged value.
- Each multi-target operation calculates and validates every target into a temporary working dictionary first. Only after all targets succeed is that operation's complete set of results added to the staged overlay. If one target fails, none of that operation's target results are published, even if the caller catches the exception inside the transaction body.
- When the block exits successfully, atomically publish the staged mapping and append its operation records to visible history under the same lock.
- If an exception escapes the transaction body, discard the entire staged overlay and all staged operation records. The pre-transaction session state remains unchanged.
- A single operation invoked outside an explicit transaction executes as its own implicit transaction and is atomic.
- A failed operation caught inside an explicit transaction leaves that individual operation's targets untouched; other successful operations in the same transaction may still commit if the transaction body then exits normally.
- Transaction commit means publishing edits into the `EditSession` overlay, not changing the original `WeightModel`.

Do not hold the session lock during checkpoint serialization, reload verification, or other unrelated export I/O. Export first captures a stable committed state under lock after confirming there is no active transaction, then performs filesystem I/O without holding the lock.


## 9. Snapshots and Rollback

Use this concrete immutable snapshot schema:

```python
@dataclass(frozen=True)
class ModelSnapshot:
    snapshot_id: str
    session_id: str
    model_id: str
    name: str | None
    created_at: datetime
    overlay: Mapping[str, np.ndarray]
    history: tuple["OperationRecord", ...]
    history_length: int
```

Requirements:
- `created_at` is timezone-aware UTC (created with `datetime.now(timezone.utc)`).
- `history` is the immutable visible operation-history prefix at snapshot creation; `history_length == len(history)` is validated.
- `overlay` is a defensive copy of the mapping wrapped in `MappingProxyType`. Its values are separate read-only views backed by the corresponding internal owning overlay arrays, which are themselves non-writeable.
- Snapshot construction never copies tensor payloads and never exposes the session's owning ndarray objects.
- Recursively freeze nested collection values in operation records as specified in Section 10.

`session.snapshot(name=None)` returns a snapshot cheaply without copying the base model or tensor payloads.

`session.restore(snapshot)` performs all checks before changing session state:
1. Reject if a transaction is active with `TransactionError`.
2. Require `snapshot.model_id == self.model_id`; otherwise raise `SnapshotError`.
3. Require `snapshot.session_id == self.session_id`; otherwise raise `SnapshotError`.
4. Validate the snapshot's mapping, read-only arrays, and `history_length == len(history)`.

After successful validation, restore replaces the session overlay with a new mapping copied from `snapshot.overlay` and replaces visible operation history with exactly `snapshot.history`. It does not append a restore operation record. This makes a snapshot reusable even after restoring an older snapshot and then creating a different edit branch. Future operation indices continue from the restored history length.

`session.rollback()` rejects execution during an active transaction, clears the overlay and visible operation history, and returns the session to the base model state. It never changes the base model. Distinguish transaction rollback (discarding one failed staged transaction) from full-session rollback (clearing all committed session edits and records).


## 10. Operation History and Provenance

Use this immutable record schema:

```python
@dataclass(frozen=True)
class OperationRecord:
    index: int
    transaction_id: str
    transaction_name: str | None
    operation: str
    targets: tuple[str, ...]
    parameters: Mapping[str, object]
    before_checksums: Mapping[str, str]
    after_checksums: Mapping[str, str]
    created_at: datetime
```

Contract:
- `index` is a zero-based sequential position in the currently visible history.
- All operation records from one successful explicit transaction share its transaction ID and optional name. A standalone implicit operation receives its own transaction ID.
- `targets` is the resolved, unique, lexicographically sorted tuple of tensor names changed by that operation.
- `before_checksums` and `after_checksums` contain the `wsp-sha256-v1` checksum for each target immediately before and after that operation in transaction execution order.
- `created_at` is timezone-aware UTC.
- Parameters are JSON-compatible values but are recursively frozen before exposure: mappings become defensive `MappingProxyType` mappings; lists and tuples become tuples; sets are converted to deterministic sorted tuples; primitives remain immutable. Any nested mapping is recursively frozen.
- A large ndarray operand is not stored in history. Store a frozen summary containing `type="ndarray"`, tuple `shape`, `dtype`, and `wsp-sha256-v1` checksum. Do not store the operand payload.
- Convert immutable parameter structures into ordinary JSON-compatible dict/list structures only when serializing provenance; never expose mutable internal record values.
- `operation_history` returns a tuple of immutable records. Failed transactions never appear in committed history.

Append operation records only in the same locked publication step that swaps the committed overlay. If either publication or history construction fails, preserve the prior overlay and prior history.


## 11. Tensor Diffs

Use these frozen schemas:

```python
@dataclass(frozen=True)
class TensorDiff:
    name: str
    original_shape: tuple[int, ...]
    current_shape: tuple[int, ...]
    original_dtype: str
    current_dtype: str
    original_checksum: str
    current_checksum: str
    changed_count: int
    total_elements: int
    changed_ratio: float
    max_abs_diff: float
    mean_abs_diff: float
    l2_diff: float


@dataclass(frozen=True)
class ModelDiff:
    model_id: str
    total_tensors: int
    edited_tensors_count: int
    total_changed_elements: int
    tensor_diffs: Mapping[str, TensorDiff]
```

Rules:
- `original_shape == current_shape` and `original_dtype == current_dtype` are invariants of all Milestone 2 operations. Store both explicitly to make the invariant auditable.
- `tensor_diffs` is a defensive mapping copy wrapped in `MappingProxyType`.
- Include every tensor in the session overlay, including a tensor whose net change is zero after a no-op edit; never include tensors that were never edited and are not in the overlay.
- `total_tensors` is the number of tensors in the base model; `edited_tensors_count` is the number of overlay tensors represented in `tensor_diffs`; `total_changed_elements` is the sum of their exact changed counts.
- Compute `changed_count` using exact element comparison. Base/current tensors must be finite for numeric metrics; if either contains NaN or infinity, raise `DiffMetricError` with the tensor name rather than returning misleading values. Edited results are already required to be finite.
- For empty tensors, set `total_elements=0`, `changed_count=0`, `changed_ratio=0.0`, and all three absolute-difference metrics to `0.0`.
- For nonempty tensors, avoid direct `orig - current` subtraction at extreme magnitudes. Let `m = max(max(abs(orig)), max(abs(current)))`; when `m == 0`, all difference metrics are zero. Otherwise compute normalized values `orig/m` and `current/m`, then calculate normalized absolute differences in float64. Compute the normalized maximum and mean from those bounded values, and the normalized L2 norm with a scaled sum-of-squares (LASSQ-style) algorithm.
- Rescale each metric by `m` only after checking that the product is representable in float64 (for a nonzero normalized metric, check against `np.finfo(np.float64).max / m` before multiplication). If any requested metric is mathematically outside float64's finite range, raise `DiffMetricError` naming the tensor and metric. Never return infinity or silently clamp a metric.
- Finite underflow in a reported metric is accepted as float64 representation; non-finite or unrepresentable metrics are not.
- Do not return full difference arrays by default. An optional helper may return a defensive copy for one named tensor if needed.

This policy allows exact element counts even when a numeric summary cannot be represented; in that case `session.diff()` raises `DiffMetricError` rather than returning a partially valid `ModelDiff`.


## 12. Safetensors Export

Implement `safetensors_exporter.py` and `session.export(path, overwrite=False)`.

Requirements:
- Export every tensor: overlay version for edited tensors and original base version for untouched tensors.
- Preserve all source file metadata when possible.
- Add namespaced provenance metadata such as `weightspace.format_version`, `weightspace.source_model_id`, and `weightspace.checksum_version` without overwriting unrelated user metadata. Replace only keys in the reserved WeightSpace-owned namespace using deterministic values. Safetensors metadata values must be strings.
- Ensure each exported array is contiguous and supported by the NumPy Safetensors backend. Reject unsupported dtypes with a clear `ExportError`; never silently convert tensor data.
- Capture a consistent committed session state under the session lock. Reject export while a transaction is active. Release the lock before serialization, verification, fsync, or filesystem publishing.
- Create a uniquely named temporary file in the destination directory using exclusive creation (for example `tempfile.mkstemp`). The temporary file must be on the same filesystem as the destination.
- Write and close the Safetensors temporary file, then flush it to stable storage using `os.fsync` on its file descriptor where supported. If writing or file fsync fails, delete the temporary file and leave any existing destination untouched.
- **Verify before publishing:** reload the temporary file using the existing `load_safetensors(temp_path)` loader. Verify the exact tensor-name set, shapes, dtypes, `wsp-sha256-v1` checksums, and expected metadata. A verification failure deletes only the temporary file and raises `ExportError`; it must never remove or modify the existing destination.
- `overwrite=False`: an upfront existence check may provide a fast error, but it is not the race-safety mechanism. Publish atomically without replacement:
  - On POSIX, use `os.link(temp_path, destination)` to atomically create the destination directory entry only if it does not already exist, then unlink the temporary name. Do not fall back to a check-then-`os.replace` sequence if hard links are unsupported; raise `ExportError` and leave the destination untouched.
  - On Windows, call the wide-character `MoveFileExW(temp_path, destination, 0)` API with flags zero (do not include `MOVEFILE_REPLACE_EXISTING`). Configure ctypes argument and return types and capture `GetLastError` on failure. The same-directory temporary path keeps the move on the same volume. A destination that appears during export must cause a no-overwrite failure.
  - A destination-exists race is reported as `ExportError("Destination file already exists: ...")`; other OS errors are wrapped with their cause.
- `overwrite=True`: only after verification and file fsync, publish with `os.replace(temp_path, destination)`. Because the temporary file is in the same directory, the replace is on the same volume. Never replace before verification.
- Clean up any remaining temporary path in a `finally` block. On success, ensure the temporary name is gone (POSIX hard-link path unlinks it; Windows move consumes it; replace consumes it).
- Where supported, fsync the parent directory after successful publication on POSIX. Treat unsupported directory fsync as best-effort and document it separately from atomic visibility. File fsync before publication is required where supported; atomic rename/link and crash durability are distinct guarantees.
- Never modify the source checkpoint.

Add an immutable `ExportReport` containing destination path, tensor count, edited tensor count, total tensor bytes, and the provenance/checksum version. Return it only after successful publication and required verification.

## 13. Exceptions

Extend `core/exceptions.py` with a small, meaningful hierarchy. Suggested types:
- `EditingError(WeightSpaceError)`
- `TensorSelectionError(EditingError)`
- `UnsupportedTensorDtypeError(EditingError)`
- `OperationValidationError(EditingError)`
- `TransactionError(EditingError)`
- `SnapshotError(EditingError)`
- `DiffMetricError(EditingError)` for non-finite source values or mathematically unrepresentable diff metrics.
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

Add comprehensive deterministic tests. At minimum:

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
- NaN and infinity operands rejected;
- Python bool and `np.bool_` scalar operands rejected;
- boolean/complex/object array operands rejected where not explicitly allowed;
- overflow/non-finite accumulator results rejected;
- post-cast float16/float32 overflow rejected;
- finite underflow to zero/subnormal accepted and NaN/infinity rejected;
- integer/bool/complex target edits rejected;
- input arrays are never mutated;
- output shape and dtype preserved;
- L2 normalization for large and tiny magnitudes, along whole tensor and axes;
- zero-vector L2 normalization produces zeros;
- z-score normalization for large offsets, extreme magnitudes, small variance, and constant slices;
- invalid axes and invalid epsilon rejected;
- clamp lower-only, upper-only, both bounds, invalid ranges, and boolean bounds;
- mask semantics, boolean mask validation, broadcasting, fill value validation;
- slice replacement, scalar index, broadcast opt-in, shape mismatch, fancy and boolean index rejection, zero-step rejection, invalid index, and full shape preservation;
- multi-target operations succeed for all selected targets or publish none, including when a failed-operation exception is caught inside an explicit transaction.

### Copy-on-write and original preservation
- starting an edit session creates an empty overlay;
- editing one tensor adds only that tensor to the overlay;
- base model tensors and checksums remain unchanged after edits;
- `get_tensor()` returns a defensive copy;
- untouched tensors are not duplicated into the overlay;
- owning overlay arrays have `writeable=False` before views are exposed;
- callers cannot set snapshot or private tensor views writable because their base owners are also non-writeable;
- snapshot mappings and nested history parameters reject mutation.

### Transactions
- successful transaction publishes all edits;
- exception in a transaction discards every staged edit;
- failure on a later target does not publish earlier target results from the same operation;
- caught multi-target operation failure leaves that operation's staged targets unchanged;
- failed single operation leaves session state unchanged;
- nested transaction rejected;
- second thread blocks until the owner transaction exits and cannot join the owner's transaction;
- operation history includes committed operations only and is published atomically with the overlay;
- snapshot, restore, rollback, diff, and export reject execution during an active transaction.

### Snapshots and rollback
- snapshot restores a previous session state;
- snapshots share immutable arrays and do not mutate when later edits occur;
- snapshot includes a frozen operation-history prefix and consistent history length;
- restore truncates/replaces visible history to exactly the snapshot prefix and appends no restore record;
- a snapshot remains reusable after restoring an older snapshot and creating a new edit branch;
- cross-model and cross-session snapshots rejected;
- full-session rollback clears edits and visible history but preserves the source model.

### Operation history
- all OperationRecord fields populated correctly;
- sequential indices and transaction IDs/names correct;
- before/after checksums match each operation's staged state;
- nested mappings/tuples are immutable;
- ndarray operand summaries include shape, dtype, checksum, and no payload;
- timestamps are timezone-aware UTC.

### Diffs
- correct changed count and ratio;
- correct max/mean/L2 difference for ordinary values;
- extreme float64 pairs that would overflow direct subtraction are handled by scaled metrics when representable;
- an unrepresentable metric raises DiffMetricError, never infinity;
- corresponding non-finite base values produce DiffMetricError;
- empty tensors produce zero metrics;
- no entry for never-edited tensors;
- zero-change result for exact no-op transformations;
- mapping and dataclass fields are immutable.

### Export
- export and reload preserves all tensor names, shapes, dtypes, checksums, and metadata;
- WeightSpace provenance metadata added deterministically;
- temporary file is reloaded and verified before destination publication;
- verification failure leaves an existing destination byte-for-byte intact;
- destination not overwritten by default, including a destination created after the initial existence check;
- POSIX no-overwrite uses atomic link creation; existing destination race is rejected;
- Windows MoveFileExW no-overwrite tests cover Unicode/non-ASCII paths, existing destination, destination-appears-during-export, API failure/GetLastError propagation, and temporary-file cleanup;
- file fsync occurs before publish where supported; parent-directory fsync is best-effort where supported;
- explicit overwrite works atomically after verification;
- failed write, fsync, verification, or publish cleans up temporary files and leaves an existing destination intact;
- a failed export never unlinks a prior destination.

Use small deterministic test arrays. Do not download real model checkpoints in unit tests. Keep test execution practical on CI.


## 16. Performance and Safety Requirements

- Never create a full duplicate of the model just to open a session or take a snapshot.
- Store only edited tensors in the overlay.
- Avoid repeated defensive copies internally; use a private read-only base view for pure engine operations where safe.
- Mark every internal owning overlay ndarray non-writeable before it is published or used as the base of an exposed view.
- Use `np.broadcast_to` or equivalent broadcast views for validation; do not materialize broadcast arrays unnecessarily.
- Use stable scaled reductions for normalization and diff metrics. Never rely on direct squaring or subtraction of unscaled extreme float64 values.
- Validate finite values, numeric dtype semantics, broadcasting, and post-cast results before publishing an edit.
- Reject unrepresentable diff metrics with `DiffMetricError`; do not return NaN/infinity or silently cap them.
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
10. Diff reports exact changed counts and numerically stable summaries, or raises DiffMetricError when a requested metric is not representable.
11. Export writes a new Safetensors checkpoint with preserved metadata and WeightSpace provenance.
12. Exported checkpoints are reloaded and verified before publication; no-overwrite publishing is race-safe, overwrite is atomic, and failed verification never damages an existing destination.
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