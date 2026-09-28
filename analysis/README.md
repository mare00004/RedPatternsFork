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
    "phi_type": mo.ui.dropdown(...),
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

## Single-mode smooth homogeneous phi

`single_mode_smooth_homogeneous` uses the same cosine-tapered radial profile
as `smooth_homogeneous`, with one longitudinal active-domain cosine mode:

\[
\varphi(\rho,z)=\varphi_{\rm smooth}(\rho,z)
\left[1+\epsilon\cos(m\pi x)\right],
\qquad x=\frac{z-z_{\rm active,0}}{z_{\rm active,end}-z_{\rm active,0}}.
\]

Its additional parameters are `amplitude` (`\epsilon`) and `mode_number`
(`m`); it accepts nonnegative mode numbers. The normal wing and normalization
pipeline remains in effect.

```bash
uv run analysis/phi_init.py export \
  --output initial_phi.h5 \
  --phi-type single_mode_smooth_homogeneous \
  --psi-avg 0.02 \
  --rho-range 5 \
  --amplitude 0.001 \
  --mode-number 7
```

## Linear Gradient Diagonal phi

`linear_full_ridge` (displayed as **Linear Gradient Diagonal**) is the
equilibrium state of the CUDA `LINEAR_FULL` gradient. It places one nonzero
rho bin in every z column along the mirrored cell-center line, so rho decreases
as z increases. It shifts the line by the configured `rho_center` and selects
the nearest rho bin. The usual wing masking and `psi_avg` normalization still
apply.

```bash
uv run analysis/phi_init.py export \
  --output initial_phi.h5 \
  --phi-type linear_full_ridge \
  --psi-avg 0.02
```

## Single-mode Linear Gradient Diagonal phi

`single_mode_linear_full_ridge` starts from `linear_full_ridge` and applies
the same active-domain longitudinal cosine perturbation as
`single_mode_smooth_homogeneous`:

\[
\varphi(\rho,z)=\varphi_{\rm ridge}(\rho,z)
\left[1+\epsilon\cos(m\pi x)\right].
\]

It accepts `amplitude` (`\epsilon`) and nonnegative `mode_number` (`m`), then
uses the standard wing masking and `psi_avg` normalization pipeline.

```bash
uv run analysis/phi_init.py export \
  --output initial_phi.h5 \
  --phi-type single_mode_linear_full_ridge \
  --psi-avg 0.02 \
  --amplitude 0.001 \
  --mode-number 7
```

## Perturbed Gaussian Linear Gradient Diagonal phi

`perturbed_linear_full_gaussian_ridge` maps an initially Gaussian rho
distribution onto the thin `LINEAR_FULL` neutral-buoyancy diagonal. Its ridge
weights follow that Gaussian, then a seeded finite longitudinal displacement
is applied:

\[
T(z)=z+\xi(z),\qquad
\varphi_{\rm pert}(\rho,T(z))=\frac{\varphi_0(\rho,z)}{1+\xi'(z)},
\qquad
\xi(z)=\epsilon\sum_{n=n_{\min}}^{n_{\max}}b_n\sin(2\pi nz/L).
\]

Here `epsilon` is a displacement in metres, (L=N\,dz), and the seeded
coefficients (b_n) are standard normal. The implementation conservatively
remaps each rho row through the monotone map, preserving each density class's
total phi and keeping phi nonnegative. It rejects configurations where
\(1+\xi'(z)\leq0\), because the map would fold over.

```bash
uv run analysis/phi_init.py export \
  --output initial_phi.h5 \
  --phi-type perturbed_linear_full_gaussian_ridge \
  --psi-avg 0.02 \
  --gaussian-mu 1100 \
  --gaussian-sigma 4 \
  --epsilon 1e-6 \
  --seed 0 \
  --mode-min 1 \
  --mode-max 32
```

## Perturbed Linear Gradient Diagonal phi

`perturbed_linear_full_ridge` applies the same finite, seeded conservative
longitudinal displacement to the constant `linear_full_ridge`. It preserves
each rho row's total phi, remains nonnegative for a monotone displacement map,
and is available in the Workbench phi picker.

```bash
uv run analysis/phi_init.py export \
  --output initial_phi.h5 \
  --phi-type perturbed_linear_full_ridge \
  --psi-avg 0.02 \
  --epsilon 1e-6 \
  --seed 0 \
  --mode-min 1 \
  --mode-max 32
```
