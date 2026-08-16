# Analysis tools

The `red_patterns` package provides notebook, CLI, and sweep helpers around
the CUDA simulation.  Initial phi distributions are implemented in
`red_patterns.phi`.

## Initial phi flow

Every entry point produces the same validated Pydantic payload, constructs a
`PhiField`, computes a `PhiResult`, and can write a CUDA-compatible HDF5 file:

```text
CLI / Marimo UI / PhiSweep row
        ↓
PHI_PARAMS_ADAPTER.validate_python(payload)
        ↓
concrete PhiParams model
        ↓
phi_field_from_params(...)
        ↓
concrete PhiField → PhiResult → /phi/values in HDF5
```

`PhiType` is the distribution identifier.  `PHI_FIELD_TYPES` maps each enum
member to its `PhiField` subclass, and each subclass declares its matching
`params_model`.  The Pydantic discriminated union uses `phi_type` to select the
concrete parameter model and rejects parameters belonging to a different type.

All phi payloads use the canonical shared names:

```python
{
    "phi_type": ...,
    "psi_avg": ...,
    "N": ...,
    "wing_z": ...,
    "wing_r": ...,
    "rho_center": ...,
    "rho_span": ...,
    "dz": ...,
}
```

## Example: perturbed smooth homogeneous phi

The type is represented by:

```python
PhiType.PERTURBED_SMOOTH_HOMOGENEOUS
```

Its schema is `PerturbedSmoothHomogeneousPhiParams`, which inherits the smooth
homogeneous `rho_range` parameter and adds:

```python
{
    "rho_range": ...,
    "seed": ...,
    "amplitude": ...,
}
```

The corresponding compute class is `PerturbedSmoothHomogeneousPhi`.  Its
`build(rho, z)` method first creates the smooth homogeneous field, then calls
`perturb_phi_z(...)` with `wing_z`, `seed`, and `amplitude`.  The normal compute
pipeline applies wings and normalization; export stores `rho_range`, `seed`,
and `amplitude` as type-specific HDF5 metadata.

### CLI

`build_export_parser()` adds the shared arguments and asks every registered
field class for its type-specific arguments.  The smooth parent registers
`--rho-range`; the perturbed type registers `--seed` and `--amplitude`.

```bash
uv run analysis/phi_init.py export \
  --output initial_phi.h5 \
  --phi-type perturbed_smooth_homogeneous \
  --psi-avg 0.02 \
  --N 512 \
  --wing-z 32 \
  --wing-r 32 \
  --rho-range 5 \
  --seed 7 \
  --amplitude 0.001
```

`validate_export_namespace()` converts the `argparse.Namespace` to a payload,
validates it with `PHI_PARAMS_ADAPTER`, and calls `phi_field_from_params()`.
For this command, the resulting object is a
`PerturbedSmoothHomogeneousPhi`.

### Marimo UI

`make_phi_ui()` returns one outer `mo.ui.dictionary` with three nested pieces:

```python
{
    "common": mo.ui.dictionary(...),
    "phi_type": mo.ui.tabs(...),
    "variants": mo.ui.dictionary(...),
}
```

`common` holds `psi_avg`, `N`, `wing_z`, and `wing_r`.  `variants` contains one
registered `mo.ui.dictionary` per phi type.  For the perturbed type,
`PerturbedSmoothHomogeneousPhi.make_ui_controls()` supplies controls for
`rho_range`, `seed`, and `amplitude`.

Inactive variant dictionaries remain registered so switching types preserves
their values.  `phi_field_from_ui()` merges the common values with only the
selected variant's values, derives `dz` from `N`, validates the payload, and
returns the selected field class.

### Sweep generation

`PhiSweep` stores sequences of common and type-specific values.  Its `rows()`
method builds the common Cartesian product, resolves each selected `PhiType`
through `PHI_FIELD_TYPES`, and asks the class for `sweep_param_names()`.

```python
PerturbedSmoothHomogeneousPhi.sweep_param_names()
# ("rho_range", "seed", "amplitude")
```

Those sequences form the type-specific Cartesian product and are merged with
each common row.  The resulting dictionaries can be passed directly to
`PHI_PARAMS_ADAPTER.validate_python()` and produce
`PerturbedSmoothHomogeneousPhiParams` instances.
