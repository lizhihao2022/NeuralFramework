# NeuralFramework Design Specification (Agent Contract)

Version: 1.1  
Audience: Codex/Agent contributors  
Scope: PDE-centric learning framework (one-step, rollout, SR, diffusion; grid/point/graph modalities)

This document defines **non-negotiable constraints** for the codebase.  
All changes MUST comply with this spec unless the spec itself is updated.

---

## 0. Terminology

- **BNC**: tensor layout `(B, N, C)`
  - `B`: batch size
  - `N`: number of spatial samples (flattened)
  - `C`: channels/features (history packed into channels)
- **Spatial shape**: original grid/geometry shape (e.g., `(H, W)` or `(D, H, W)`), stored as metadata.
- **Coords**: geometric coordinates, logically per-sample; physically may be shared across batch.
- **Geom**: geometry metadata dictionary (mesh/graph/masks/etc.).
- **Task**: one-step / rollout / SR / diffusion, each with an explicit contract.

---

## 1. Core Design Principles (MUST)

### 1.1 Canonical layout for learning signals
All processed learning signals MUST be **BNC**:
- dataset outputs: `x`, `y` are BNC after collation
- trainer forward/loss: consumes BNC
- metrics input: consumes BNC (may unflatten internally)

Any other layout (e.g., `BCHW`, `BHWC`) is allowed ONLY **inside a model** or inside metric/visualization utilities, and MUST be converted back to BNC at the boundary.

### 1.2 Explicit contracts over implicit inference
- No “guessing” semantics by slicing channels (e.g., assuming coords are last channels).
- No reliance on heuristics such as `y.ndim` to select task logic.
- All feature packing and task semantics MUST be declared via `meta`.

### 1.3 Minimal coupling
- Trainer MUST NOT contain dataset-/model-specific slicing or reshaping logic (except standardized canonicalization utilities in §3).
- Dataset MUST NOT embed model-specific hacks.
- Metrics/visualization MUST be pure functions (no side effects).

### 1.4 Type safety (Pylance strict)
All code MUST be Pylance-compliant under strict settings:
- no implicit `Any` propagation
- no missing methods / protocol violations
- no unsafe `Optional` access
- no dynamic attribute injection

### 1.5 Documentation rules
- All docstrings/comments MUST be **English**.
- Docstrings MUST be **concise** (purpose + args/returns only).
- Inline comments MUST be rare and only for non-obvious reasoning.

---

## 2. Data Contract (MUST)

### 2.1 Dataset output format (per batch)
Each dataset batch MUST be a dictionary:

```python
from typing import Any, TypedDict
import torch

class Batch(TypedDict):
    x: torch.Tensor                     # (B, N, Cx)
    y: torch.Tensor                     # (B, N, Cy)
    coords: torch.Tensor | None         # see §2.4
    geom: dict[str, Any] | None         # see §2.5
    meta: dict[str, Any]                # required, see §2.2
````

Single-sample `__getitem__` may omit the batch dimension, but the collate MUST output:

* `x` as `(B, N, Cx)`
* `y` as `(B, N, Cy)`

### 2.2 Required metadata fields

`meta` MUST include:

* `layout: Literal["BNC"]` (always `"BNC"`)
* `spatial_shape: tuple[int, ...] | None`

  * grid: `(H,)`, `(H, W)`, `(D, H, W)`
  * point/graph: may be `None` if not applicable
* `in_t: int`, `out_t: int`
* `field_channels_x: int` (physical channels per timestep, excluding coords unless appended)
* `field_channels_y: int`
* `feature_packing: dict[str, Any]` (see §2.3)
* `coords_shared: bool` (see §2.4)

### 2.3 Feature packing rules (history into C)

Inputs MUST pack temporal history into channels:

* If the field has `F = field_channels_x` channels and history length is `in_t`,
  then the history-packed field part has `F * in_t` channels.

`meta["feature_packing"]` MUST encode:

* `field_dim: int` (F)
* `history: int` (in_t)
* `coords_in_x: bool` (default False, see §2.4)
* `order: list[str]` describing channel composition, e.g.:

  * `["field_history"]`
  * `["field_history", "extra_features"]`
  * `["field_history", "coords"]` (only if `coords_in_x = True`)

### 2.4 Coordinates handling (UPDATED)

Coords are logically per-sample but may be physically shared to reduce memory.

Allowed `coords` shapes:

* `(N, d)` shared across batch
* `(1, N, d)` shared across batch (explicit shared batch dimension)
* `(B, N, d)` per-sample coords

Rules:

* Default: `coords_in_x = False` and `coords` provided separately.
* If a dataset appends coords into `x`, it MUST set `coords_in_x = True` in `feature_packing`.
* Trainer and models MUST NOT assume coords are appended unless `coords_in_x = True`.

`meta["coords_shared"]` MUST be:

* `True` if coords are shared (`(N,d)` or `(1,N,d)`)
* `False` if coords are per-sample (`(B,N,d)`)

### 2.5 Geometry (`geom`) contract

`geom` is optional and modality-dependent (grid/mesh/graph). It MUST be a plain dictionary.

Allowed contents (examples):

* masks, boundary flags, land/sea masks
* mesh connectivity
* graph `edge_index` / `edge_attr`
* per-node/element attributes

Constraints:

* Keys MUST be strings.
* Values SHOULD be tensors or JSON-serializable objects.
* Datasets MUST document `geom` fields concisely in a docstring.

---

## 3. Canonicalization Utilities (MUST)

Centralized utilities MUST exist (no ad-hoc reshapes scattered across trainers/models).

### 3.1 Flatten / unflatten

Provide:

* `flatten_grid(u: Tensor, spatial_shape) -> Tensor` to produce BNC
* `unflatten_grid(u_bnc: Tensor, spatial_shape) -> Tensor` to restore grid layout for models/metrics/visualization

Constraints:

* Flattening order MUST be consistent (row-major/C-order).
* Unflatten MUST be the exact inverse.

### 3.2 Coords canonicalization (UPDATED)

Provide a single utility:

```python
def canonicalize_coords(
    coords: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor | None:
    """Normalize coords to shape (B, N, d) using expand when shared."""
```

Rules:

* `coords is None` -> `None`
* `(N, d)` -> `(1, N, d)` -> expand to `(B, N, d)`
* `(1, N, d)` -> expand to `(B, N, d)`
* `(B, N, d)` -> unchanged
* otherwise -> raise `ValueError` with clear expected/actual shapes

Important:

* Use `expand` (not `repeat`) for shared coords to avoid memory blow-up.

### 3.3 Model internal layout conversions

If a model needs `BCHW/BHWC`, it MUST:

* use shared utilities to unflatten/reshape
* validate `N == prod(spatial_shape)` when `spatial_shape` is not None
* return BNC after computation

---

## 4. Model Contract (MUST)

### 4.1 Forward signature (no signature introspection)

All models MUST implement:

```python
def forward(
    self,
    x: torch.Tensor,                           # (B, N, Cx)
    *,
    coords: torch.Tensor | None = None,         # canonicalized to (B, N, d) by Trainer
    geom: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> torch.Tensor:                             # (B, N, Cy)
    ...
```

Constraints:

* `coords`, `geom`, `meta` MUST be keyword-only.
* Models MUST return BNC.
* Models MUST NOT rely on `inspect.signature`-based injection logic.

### 4.2 Config-driven behavior

* All optional behavior (e.g., `use_coords`) MUST be explicit via config.
* If `meta["feature_packing"]["coords_in_x"] == True`, models MUST NOT append coords again.

### 4.3 Internal reshape policy

Grid-based models (UNet/FNO/MG-style) MUST:

* read `spatial_shape` from `meta`
* validate compatibility
* unflatten -> compute -> flatten back

---

## 5. Task Semantics (MUST at boundary)

The trainer MUST separate task modes by an explicit flag, e.g. `data.task`.

### 5.1 One-step

* Input: `x` (B, N, Cx)
* Target: `y` (B, N, Cy)
* Output: `y_hat` (B, N, Cy)

### 5.2 Rollout (autoregressive)

Rollout MUST be defined by:

* `in_t`, `out_t`, and `rollout_steps` in config/meta
* explicit history update policy using `feature_packing`

Constraints:

* Rollout MUST NOT assume coords are embedded in `x`.
* History update MUST slice field-history channels using `field_dim` and `history`.
* Coords/extra features MUST remain static unless declared otherwise.

### 5.3 Super-resolution (SR)

SR MUST define:

* `spatial_shape_in`, `spatial_shape_out` in `meta`
* mapping strategy handled by task/model (not by Trainer heuristics)

---

## 6. Trainer Contract (MUST)

### 6.1 Responsibilities

Trainer MUST handle:

* DDP setup
* optimizer/scheduler
* checkpointing/logging
* calling model with standardized batch dict
* loss and metrics aggregation

Trainer MUST NOT:

* reshape `x/y` into BCHW/BHWC for model convenience
* contain dataset-specific slicing rules
* embed visualization

### 6.2 Batch-to-model call

Trainer MUST:

1. canonicalize `coords` via `canonicalize_coords`
2. call model as:

```python
coords = canonicalize_coords(batch["coords"], batch["x"].shape[0])
y_hat = model(batch["x"], coords=coords, geom=batch["geom"], meta=batch["meta"])
```

All shape mismatches MUST raise `ValueError` with:

* actual shape
* expected shape
* dataset/model name if available

---

## 7. Metrics and Visualization (MUST)

### 7.1 Metrics API

Metrics MUST accept:

* `pred: (B, N, C)`
* `target: (B, N, C)`
* `meta`

If spatial restoration is needed, metrics MUST use `unflatten_grid` with `meta["spatial_shape"]`.

### 7.2 No training-step visualization

Visualization utilities MUST be called outside the training step and MUST not mutate tensors/meta.

---

## 8. Code Quality and Pylance Rules (MUST)

### 8.1 Type hints everywhere

* All public functions/methods MUST be fully annotated.
* Avoid `Any`; allow it only for boundary dicts (`geom`, `meta`) and keep it localized.
* Prefer `TypedDict`, `Protocol`, `dataclass` for stable interfaces.

### 8.2 Disallowed patterns

* monkey patching
* dynamic attribute injection
* signature introspection to decide call arguments
* ambiguous containers mixing unrelated types without typing

### 8.3 Error handling

* No bare `except:`
* No silent fallbacks for shape/layout mismatches

---

## 9. Documentation Rules (MUST)

* English only.
* Concise docstrings (purpose + args/returns).
* Minimal inline comments.

---

## 10. Change Management (MUST)

If a change affects configs/checkpoints:

* update templates
* add a short migration note in `docs/MIGRATION.md`
* add at least a minimal smoke test:

  * 1 epoch on tiny subset
  * shape utility inverse test (flatten/unflatten)

---

## Appendix A: Recommended File Additions (SHOULD)

* `core/types.py` for `Batch` typing
* `core/shapes.py` for flatten/unflatten + canonicalize_coords
* `tasks/` for one-step / rollout / SR / diffusion task logic
* `tests/test_shapes.py` for shape invariants
