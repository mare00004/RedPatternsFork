# Analysis tools

The `red_patterns` package provides notebook, CLI, and sweep helpers around
the CUDA simulation.  Initial phi distributions are implemented in
`red_patterns.phi`.

## Local Taylor runs without buoyancy

Use `uv run marimo edit analysis/workbench.py` to generate and run initial
conditions. Select **Taylor** and gradient **zero**. In the current simulator,
`ZERO` sets **alpha = 0**, removing the entire buoyancy contribution
`-alpha * (rho - p)` from the face velocity. It also stores a constant physical
Percoll density `p = P0`; it does not merely flatten the profile while retaining
sedimentation. The interaction contribution `-beta * I_face` remains active.
The equivalent CLI options are `--use-taylor --gradient=zero` with your NU/MU.
Rebuild your selected binary after source changes; the saved
`/config/model` attribute `alpha` must be zero.

For a dispersion measurement about a spatially uniform state, set **wing_z = 0**
and use **Smooth Homogeneous** as the unperturbed initial condition. Run it,
then run **Perturbed Smooth Homogeneous** with a small amplitude (for example
`1e-4`) and otherwise identical parameters, including NU/MU, grid, and mean
concentration. Save both interaction runs' `run.h5` files and load them in
`analysis/analyze_dct_dispersion.py`. The workbench's automatically generated
no-interaction comparison is not the equilibrium reference: it changes NU/MU.

Check that the reference stays stationary, fit only the early linear regime,
and check stability of the rates as you reduce perturbation amplitude and time
step. Cosine fits describe a dispersion relation when the observed cosines
evolve independently; inspect residuals and mode mixing from the finite-domain
stencils. `analysis/analyze_linear_growth.py` provides joint dynamic-mode
analysis when several spatial cosines mix.

## Shared-window wavelength and paired-norm phase maps

```bash
uv run marimo edit analysis/analyze_finite_time_stability.py
```

Select the sweep directory containing `runs.jsonl` and `results/`. This notebook
matches Taylor `LINEAR_FULL_RIDGE` / `PERTURBED_LINEAR_FULL_RIDGE` runs by their
shared settings, including exact NU/MU, and provides two Altair phase-space maps:

- The fastest resolved growing wavelength from log-amplitude fits of the
  **perturbed field's own DCT-II coefficients**, with wavelength `2*N*dz/m`.
- The finite-time rate of the **time-matched paired difference**,
  `D(t) = sqrt(sum_m |DCT_m(psi_perturbed(t) - psi_unperturbed(t))|^2)`,
  `lambda_eff = (ln D(end) - ln D(start)) / (end - start)`.

One time range slider selects the linear window for every loaded pair; leave its
start at zero for `lambda_eff(0, end)`. One inclusive mode range controls both
maps and the norm, defaulting to the seeded band (1–100 for the example sweep).
Endpoints come from saved times common to all loaded pairs. Each pair must have
identical perturbed/reference saved-time and spatial grids; no interpolation is
performed. Missing or incompatible pairs remain in the availability table.
HDF5 loading and DCT computation are cached between slider changes.

Click a cell in either Altair heatmap to select a ν, μ pair and highlight it in
both maps; double-click to clear the selection. The run selector lists matching
seeds when a cell contains multiple runs. Grey cells are unresolved or unavailable;
numeric cell edges lie halfway between samples within each row.
The coordinate selector offers the original `(ν, μ)` view, `(ν², −μ)`, and
the default `(|ν|/|ν_c(μ)|, −μ)` view. For the saved Taylor model, optimizing
over continuous wavenumber gives `ν_c² = C*(-μ)`, where
`C = 4*alpha*(30/L)/(beta*mean_psi)`. This is a straight boundary in the squared
view and the vertical line at 1 in the normalized view. Dashed lines show this
continuum prediction using each loaded pair's saved parameters and initial
reference mean. Numeric axis spacing preserves the transformed boundary shape.
Transformed views restrict to ν < 0 and μ < 0; normalization also requires
valid saved parameters with positive alpha, beta, and mean psi. Omitted rows
remain in the table and CSV, which include the transformed coordinates.
These predictions do not impose the selected discrete DCT mode band.
Inspect individual pairs using the full-time logarithmic norm plot with window
markers and endpoint secant, modal fit table, and selected-mode log-amplitude
plot. Download the phase-map summary and selected-window norm as CSV. Refresh
saved results to rescan an ongoing sweep.

The first map uses raw perturbed coefficients, so background evolution can
affect it. The second cancels the evolving reference. Its zero crossing is a
finite-time, selected-band diagnostic, not automatically the asymptotic
instability boundary: a growing mode can coexist with a decaying total norm.
Compare nearby windows and bands. Unresolved modal fits, zero norm endpoints,
and norms below numerical precision are flagged rather than labelled stable.

To exercise all cells with an analytic synthetic sweep:

```bash
uv run python analysis/analyze_finite_time_stability.py
```

## Cosine eigenmode dispersion relation

For a complete paired Taylor `LINEAR_FULL` sweep, use the compact browser:

```bash
uv run marimo edit analysis/analyze_stability_sweep.py
```

Select the directory containing `runs.jsonl` and `results/<run_id>/run.h5`.
The browser matches `perturbed_linear_full_ridge` runs to `linear_full_ridge`
references using identical shared simulation and initial-grid settings, including
exact NU/MU values. Missing files and ambiguous references remain visible.
Choose a pair for shared-scale ψ heatmaps, a perturbation heatmap, modal rates
and wavelengths, the saved-parameter continuum prediction, cutoff diagnostics,
and individual log-amplitude fits with residuals. Pair CSV/NPZ downloads preserve
the selected window, modes, coefficients, fit masks, and metadata.

The optional **Whole-sweep growth map** analyzes each available pair sequentially
using a common fit endpoint (default 100 s) and mode range (default 1–100).
It plots the maximum resolved rate and continuum boundary, with a downloadable
summary table. Unresolved fits or reference drift are not labelled stable;
decay only describes the selected modes. Check the linear window individually,
especially far from the boundary. **Refresh results** rescans an ongoing sweep.
The synthetic checkbox supplies analytic validation data; script mode enables it
automatically (`uv run python analysis/analyze_stability_sweep.py`).

```bash
uv run marimo edit analysis/analyze_dct_dispersion.py
```

Select an unperturbed equilibrium and its perturbed `run.h5` under the same
dynamics. The notebook displays their saved ψ heatmaps side by side with a
shared color scale. Select the end of the linear regime with the saved-time
slider, then choose an inclusive DCT-II mode range. It subtracts the first
equilibrium ψ profile, held fixed, and fits `ln|c_m| = b_m + gamma_m (t-t0)`
by unweighted linear least squares using the original saved times through the
selected endpoint. There is no additive offset in the coefficient model.
Negative coefficients are supported; modes with resolved sign changes are
rejected. Samples at or below the precision floor are excluded without adding
epsilon, and at least five samples must remain. Fit errors and R² refer to log
amplitude, and downloads include the above-floor sample masks. This assumes
directly excited exponential modes; accumulated signals such as
`A (exp(gamma t) - 1)` require the offset-capable fits in
`analyze_linear_growth.py` instead.
For independently evolving cosine eigenmodes, these rates give the perturbation
dispersion relation, plotted against both wavenumber and wavelength.

For saved Taylor runs, the wavenumber plot overlays the continuum prediction
`sigma(k) = -beta * AVG_PSI * NU * k^2 + beta * AVG_PSI * MU * k^4`.
Coefficients come from the run metadata; `AVG_PSI` is the spatial mean of the
first equilibrium ψ profile over the full stored grid, in fraction units.
The caption shows these values, which are also saved in the NPZ metadata.
For `LINEAR_FULL`, the overlay automatically includes the buoyancy shift
`-alpha*g`, with saved `alpha` and `g = 30/(N*dz)` matching the simulator's
full-domain density gradient. This is the continuum prediction for
displacements of a constant-weight equilibrium ridge; the caption and NPZ
metadata include `alpha`, `g`, and the decay rate `alpha*g`.
For other gradients, the curve remains the uniform-state prediction without
buoyancy. The overlay is omitted for the synthetic validation pair and
convolution runs.

The full stored grid defines `L = N * dz`, `k_m = pi*m/L`, and the DCT-II cosine
wavelength `lambda_m = 2*L/m`. Mode 0 has no finite wavelength. The notebook
reports the fastest **measured** growing wavelength among resolved selected
modes. Short-wave cutoffs require adjacent resolved modes to cross from growth
to decay as k increases; their zero crossing is interpolated linearly in k.
It reports every bracket and retains long-wave band edges for multiple unstable
bands. A missing bracket leaves the cutoff unresolved, without extrapolation.
Unexcited, flat, poorly fitted, or unresolved modes do not determine these
features. A selected-mode explorer plots log magnitudes, fitted lines, and
log residuals, marking excluded samples and the precision floor.
Tables and downloads include modal rates, fit diagnostics, wavelengths,
crossings, and an NPZ of coefficients, fits, and selection metadata.

The synthetic validation pair excites independent cosines with known rates,
a maximum at mode 5, and a short-wave cutoff between modes 7 and 8. It can be
enabled interactively and is used automatically when running the notebook as
a script:

```bash
uv run python analysis/analyze_dct_dispersion.py
```

Invoking `python` keeps the repository's installed `red_patterns` package
available; invoking this file directly with `uv run` uses its isolated script
dependencies, which do not include the local package.

## Equilibrium dynamic modes and linear growth rates

Launch the marimo notebook from the repository root:

```bash
uv run marimo edit analysis/analyze_linear_growth.py
```

Select a **perturbed equilibrium** `run.h5` and an **unperturbed equilibrium**
`run.h5` under the same dynamics. The notebook checks model type, gradient,
interaction parameters (Taylor NU/MU or the saved convolution kernel), and
spatial grids. The initial phi fields should differ by the perturbation.
The workbench's interaction/no-interaction pair uses different dynamics and
is unsuitable for this equilibrium analysis.

Both heatmaps use the shared RBC colormap with the same physical color scale.
The reference panel shows the first saved equilibrium profile held fixed over
the perturbed run's times. The notebook reports measured equilibrium psi drift
(and first-to-last phi drift when saved), warning if it exceeds 1e-6 relative
norm. Verify stationarity before interpreting the modes. The equilibrium run
can have a different cadence or duration. The saved-time slider marks the end
of the linear regime and includes that frame in analysis.
The notebook preserves float64 precision and computes
`delta_psi(z,t) = psi_perturbed(z,t) - psi_equilibrium(z,t0)`, followed by
an orthonormal DCT-II for every selected saved frame and spatial mode,
including mode 0 and wing points.

Use **DCT mode range to analyze** to choose the inclusive mode interval, for
example `[5, 32]` if those are the modes you perturbed. The table, growth-rate
plot, individual-mode inspection slider, and ESPRIT analysis use only
those modes, preserving their original mode numbers. A single-mode interval is
also supported. The amplitude plot compares the selected-mode norm with the
full-field norm. The heatmaps and full stored DCT arrays retain the complete
field. The range defaults to all modes.

Two distinct growth-rate plots are shown: diagnostic exponential-fit rates
versus spatial DCT mode number, and ESPRIT rates versus extracted component
number. They are labelled separately. Each signed
coefficient is fitted as `A_m * exp(gamma_m * (t - t0)) + C_m`, using the original
saved times, including irregular saves. The constant offset accommodates a
difference signal starting at zero. Positive `gamma_m` means a growing
exponential component; negative means a decaying component approaching the
constant. The mode selector shows the measured coefficient, its fit, and the
residual; the summary plot shows growth rate versus DCT mode number.

The rate table includes relative fit error, R-squared, approximate rate standard
error, and status. Constant/weak, nearly linear, uncertain, or search-limit fits
have no reported rate. Other poor fits are marked explicitly. `ok` requires
relative error at most 10% and rate curvature resolved relative to the fit
uncertainty; it does not establish independent modal evolution. The search
allows `abs(gamma_m * window_duration) <= 50`. The standard error assumes
uncorrelated residuals. A cosine coefficient mixing several eigenmodes or
oscillating need not have one well-defined growth rate.

Main downloads provide the selected window/settings as JSON, the rate table as
`dct_mode_growth_rates.csv`, and raw time, difference/DCT arrays, fitted traces,
rates, and statuses as `dct_mode_growth_analysis.npz`. No interpolation is used
for individual-mode fitting. These downloads work independently of the SVD.
Both analysis archives record the inclusive mode bounds and `selected_modes`.
In the individual-mode archive, `selected_dct_vectors` and
`mode_fitted_coefficients` have one column per entry in `selected_modes`, in
that order; the rate/status arrays use the same ordering. `dct_vectors` still
contains all original spatial modes.

**Compute ESPRIT growth rates (Hankel / SVD)** is enabled by default and can be
disabled while inspecting DCT coefficient fits. ESPRIT extracts the shared
temporal exponents in `c_m(t) ≈ sum_q b_mq exp((gamma_q + i omega_q) (t-t0))`.
One component can mix several spatial DCT modes; its growth rate describes the
entire observed pattern. Component numbers are sorting labels, not DCT mode
numbers, and can change with the fit window or rank. A component's “dominant DCT mode” does
not assign that rate to one cosine mode. Choose delay count and SVD rank to
inspect the singular values, shift residual, and global exponential
reconstruction. Growth rates
are `log(abs(lambda)) / saved_dt`, in inverse seconds; oscillatory components
also report angular frequency. The fit requires uniform spatial coordinates
and checks matching spatial grids and model parameters, including interaction
coefficients/kernel. Different initial conditions are expected; changing the
operator between runs is rejected.
The simulation's elapsed-time saving threshold can produce intervals differing
by one solver step (for example, 1.00 and 1.01 seconds). For irregular saved
times, the notebook linearly interpolates a copy of the DCT vectors onto a
uniform grid with the same sample count and endpoints before forming the Hankel
matrix. It reports the original interval range and interpolation. This is an
approximation, particularly for fast oscillations or large sampling gaps.
Uniformly sampled data is used directly.
Changing rank reuses the existing Hankel SVD. Hankel matrices over 512 MiB are
blocked before allocation; reduce the delay count or selected time window.

Read the ESPRIT growth plot as follows: positive growth means an increasing
envelope, negative means a decaying envelope, and near zero means an
approximately steady envelope. An envelope at `gamma = 0.02 s^-1` doubles in
34.7 s. Nonzero `omega` adds oscillation with period `2*pi/abs(omega)`;
conjugate exponents with equal growth and opposite frequency together represent
one real oscillating pattern. The table reports frequencies and periods.

The adjacent DCT-content heatmap plots `abs(b_mq)^2 / sum_m abs(b_mq)^2`,
normalized separately for each component over the selected modes. It connects
each ESPRIT rate to its observed spatial cosine content. Normalization removes
overall excitation strength, and the map omits complex phase; it is not a
measure of reliability or an additive energy partition across components.
Complex fitted amplitudes are retained in the archive. These are fitted
coefficient vectors for the extracted poles, not full phi eigenvectors.

The **dynamic-mode explorer** lets you select an ESPRIT component and a time.
It displays its normalized complex spatial psi structure (real and imaginary
quadratures), DCT weights, the fitted contribution at that time, and the
growing envelope. Conjugate partners are included together in the real
contribution; choosing either member displays the same pattern. Each recovered
structure has one shared exponent: `delta_psi ≈ sum_q a_q*u_q(z)*exp(s_q*(t-t0))`.
Shapes have unit Euclidean norm, with their largest spatial entry real and
positive; the complex initial amplitude retains excitation and phase.

The wavelength table reports the strongest DCT contributions with
`wavelength = 2*M*dz/m`; mode 0 is constant with infinite wavelength.
The entire stored z grid, including wings, sets this length. Per-component CSV
exports contain normalized complex DCT and psi structures, wavelengths, and
the selected-time contribution. The inverse DCT zero pads outside the selected
mode range, so widening that range may recover additional structure. These
are observable psi projections of dynamic modes; psi alone cannot identify the
full phi eigenfunctions. Repeated eigenvalues only identify their combined
excited projection in a single trajectory.

Check stability of the extracted exponents and DCT content across nearby
endpoints, delay counts, ranks, and observed DCT ranges. Low reconstruction and
shift residuals support the exponential model, but high retained singular-value
energy alone does not establish reliable equilibrium eigenvalues. Unexcited
or unobserved dynamics cannot be recovered.

ESPRIT downloads provide the selected endpoint/settings as JSON,
component rates as `hankel_component_growth_rates.csv`, and the difference,
DCT vectors, singular values, exponents, and fitted arrays as
`hankel_growth_analysis.npz` with embedded metadata. With `numpy.load`, access
`metadata_json` as a string using `.item()`; no pickle is required.
`time`, `delta_psi`, and `dct_vectors` preserve the original saved observations;
`hankel_time` and `hankel_vectors` contain the uniform input used by the SVD.
The reconstruction uses `hankel_time`. Metadata records whether interpolation
was applied and the interval deviation; `saved_dt_s` is the Hankel time step.
The `esprit_dct_weights` array uses rows ordered by `selected_modes` and columns
ordered by the exponents, matching the plotted components. `spatial_amplitudes`
preserves their complex fitted coefficient vectors. `dynamic_dct_shapes` and
`dynamic_psi_shapes` have components as columns; `dynamic_initial_amplitudes`
contains their complex initial amplitudes. `dynamic_time_factors` uses
`hankel_time`. The spatial reconstruction is
`(dynamic_time_factors * dynamic_initial_amplitudes) @ dynamic_psi_shapes.T`.
`dynamic_conjugate_partners` contains zero-based partner indices, or -1 if
unpaired. `equilibrium_psi_profile` stores the fixed reference.

Both methods estimate observed difference-signal rates. The reference is held
fixed. A biased equilibrium profile or a conserved direction can contribute a
constant component. Interpreting rates as eigenvalues
of the perturbed equilibrium's phi linearization requires an appropriate
reference and modes observable through psi. Check stability across endpoints,
delay counts, and ranks. A synthetic validation pair is available in the UI
and used automatically in script mode. It mixes DCT modes 3, 5, and 7 in two
growing dynamic structures with shared rates +0.12 and +0.06 s^-1; ESPRIT rank 2
recovers both despite mixed individual DCT traces:


```bash
uv run analysis/analyze_linear_growth.py
```

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
\xi(z)=\epsilon\sum_{n=n_{\min}}^{n_{\max}}b_n\sin(\pi nz/L).
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

Mode numbers now match the DCT-II cosine indices: for a constant marginal
`C`, `delta_psi = -C * xi'` excites `cos(n*pi*z/L)` to first order.
Set **mode_min = mode_max = 50** to excite cosine mode 50. The displacement
uses sines so both boundaries stay fixed and each density class is conserved.
Finite amplitudes can also generate harmonics. Set both wing sizes to zero
for comparison with full-domain cosine modes. The Gaussian-weighted variant
uses the same displacement basis, but its nonuniform marginal also mixes modes.

Previously the displacement used `sin(2*pi*n*z/L)`, so index 25 excited
DCT mode 50. Regenerate initial phi files to use the new convention; new
exports record `displacement_basis = sin(n*pi*z/L)`. To reproduce an old
single-mode displacement with the same seed and epsilon, double its index.

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
