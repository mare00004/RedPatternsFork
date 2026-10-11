# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo>=0.23.6", "numpy", "scipy", "h5py", "matplotlib", "pandas"]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="wide")


with app.setup:
    import io
    import json
    from pathlib import Path

    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.fft import dct, idct

    from red_patterns import get_rbc_cmap, plot_psi_arrays
    from red_patterns.linear_growth import (
        estimate_growth,
        fit_dct_mode,
        hankel_subspace,
        prepare_uniform_observations,
        recover_dynamic_modes,
        uniform_step,
    )

    REPO_ROOT = Path(__file__).resolve().parent.parent

    def read_psi_run(path):
        # Read float64 directly: RunData currently casts saved data to float32,
        # which can erase small perturbations before subtraction.
        with h5py.File(path, "r") as handle:
            result = {
                "path": str(Path(path).resolve()),
                "psi": np.asarray(handle["fields/psi"], dtype=np.float64),
                "time": np.asarray(handle["time"], dtype=np.float64),
                "z": np.asarray(handle["coords/z"], dtype=np.float64),
                "model": dict(handle["config/model"].attrs),
                "run": dict(handle["config/run"].attrs),
            }
            initial_key = "config/initial_conditions/phi_values"
            result["phi_initial"] = (
                np.asarray(handle[initial_key]) if initial_key in handle
                else np.asarray(handle["fields/phi"][0])
                if "fields/phi" in handle and handle["fields/phi"].shape[0] > 0
                else None
            )
            result["rho"] = np.asarray(handle["coords/rho"]) if "coords/rho" in handle else None
            result["phi_endpoint_relative_drift"] = None
            if "fields/phi" in handle and handle["fields/phi"].shape[0] > 1:
                _phi_first = np.asarray(handle["fields/phi"][0], dtype=float)
                result["phi_endpoint_relative_drift"] = float(
                    np.linalg.norm(np.asarray(handle["fields/phi"][-1]) - _phi_first)
                    / max(np.linalg.norm(_phi_first), np.finfo(float).tiny)
                )
            taylor_key = "config/model/variant/tayl"
            kernel_key = "config/model/variant/conv/kernel_values"
            result["kernel"] = None
            if taylor_key in handle:
                attrs = handle[taylor_key].attrs
                result["variant"] = {"NU": float(attrs["NU"]), "MU": float(attrs["MU"])}
                result["zero_interaction"] = float(attrs["NU"]) == 0 and float(attrs["MU"]) == 0
                result["dynamics_label"] = f"Taylor: ν={float(attrs['NU']):.4g}, μ={float(attrs['MU']):.4g}"
            elif kernel_key in handle:
                result["kernel"] = np.asarray(handle[kernel_key])
                result["variant"] = {key: value for key, value in handle["config/model/variant/conv"].attrs.items() if key != "kernelSource"}
                result["zero_interaction"] = bool(np.all(result["kernel"] == 0))
                result["dynamics_label"] = "Convolution: saved zero kernel" if result["zero_interaction"] else "Convolution: saved nonzero kernel"
            else:
                result["variant"] = (
                    {key: value for key, value in handle["config/model/variant/conv"].attrs.items() if key != "kernelSource"}
                    if "config/model/variant/conv" in handle else {}
                )
                result["zero_interaction"] = None
                result["dynamics_label"] = "Convolution: kernel values unavailable"
        return result

    def compare_dynamics(perturbed, equilibrium):
        differences, missing = [], []
        for key in ("gradientType", "alpha", "beta", "modelType"):
            if key not in perturbed["model"] or key not in equilibrium["model"]:
                missing.append(key)
            elif perturbed["model"][key] != equilibrium["model"][key]:
                differences.append(key)
        for key in set(perturbed["variant"]) | set(equilibrium["variant"]):
            if key not in perturbed["variant"] or key not in equilibrium["variant"]:
                missing.append(key)
            elif perturbed["variant"][key] != equilibrium["variant"][key]:
                differences.append(key)
        if perturbed["model"].get("modelType") in ("CONV", b"CONV"):
            if perturbed["kernel"] is None or equilibrium["kernel"] is None:
                missing.append("convolution kernel values")
            elif not np.array_equal(perturbed["kernel"], equilibrium["kernel"]):
                differences.append("convolution kernel values")
        return differences, missing

    def demo_pair():
        demo_time = np.linspace(0, 20, 121)
        demo_z = (np.arange(32) + 0.5) * 1e-5
        coefficients = np.zeros((len(demo_time), len(demo_z)))
        # Two growing dynamic modes, each mixing several spatial cosines.
        shape_a = np.zeros(len(demo_z))
        shape_b = np.zeros(len(demo_z))
        shape_a[[3, 5, 7]] = [1, 0.6, -0.3]
        shape_b[[3, 5, 7]] = [0.3, -0.5, 1]
        coefficients = (
            1e-4 * np.exp(0.12 * demo_time[:, None]) * shape_a
            + 7e-5 * np.exp(0.06 * demo_time[:, None]) * shape_b
        )
        baseline = np.full_like(coefficients, 0.02)
        common = {"time": demo_time, "z": demo_z,
            "model": {"modelType": "TAYL", "gradientType": "LINEAR_FULL", "alpha": 1.0, "beta": 1.0},
            "variant": {"NU": 1.0, "MU": 1.0}, "kernel": None,
            "run": {}, "phi_initial": None, "rho": None, "phi_endpoint_relative_drift": None}
        return (
            dict(common, path="Synthetic perturbed equilibrium", psi=baseline + idct(coefficients, type=2, norm="ortho", axis=1), zero_interaction=False, dynamics_label="Known shared rates: +0.12 and +0.06 s⁻¹"),
            dict(common, path="Synthetic equilibrium", psi=baseline, zero_interaction=False, dynamics_label="Stationary equilibrium under the same dynamics"),
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Dynamic modes and growth around an equilibrium

    Select a **perturbed equilibrium** `run.h5` and an **unperturbed equilibrium**
    `run.h5` with the same dynamics: gradient, interaction parameters/kernel,
    and spatial grids. The initial $\phi$ fields should differ by your perturbation.
    A workbench interaction/no-interaction pair changes the operator and is not
    the equilibrium/perturbed pair needed here.
    Mark the end of the linear regime using the time slider; every saved frame
    from the first frame **through that endpoint** enters the analysis.

    The notebook computes $\delta\psi(z,t)=\psi_{\rm perturbed}(z,t)-\psi_{\rm equilibrium}(z,t_0)$,
    an orthonormal spatial DCT-II, and two distinct analyses: **diagnostic
    exponential fits versus DCT mode number**, and **ESPRIT growth rates versus
    extracted component number**.
    The full signed DCT is retained; choose an **inclusive mode range** for
    analysis below. Mode 0 and wing points remain available. Individual-mode fits use the original physical times,
    including irregular saves. ESPRIT uses Hankel shift invariance below to
    extract dynamic components that can mix several DCT modes. The dynamic-mode
    explorer reconstructs each component's spatial $\psi$ structure and shows its
    shared growth rate, oscillation, and wavelength content. The equilibrium's
    first saved profile is held fixed; the equilibrium run can have a different
    saving cadence or duration.
    """)
    return


@app.cell
def _():
    perturbed_picker = mo.ui.file_browser(
        initial_path=REPO_ROOT / "data", filetypes=[".h5"], multiple=False,
        label="Perturbed equilibrium run.h5",
    )
    equilibrium_picker = mo.ui.file_browser(
        initial_path=REPO_ROOT / "data", filetypes=[".h5"], multiple=False,
        label="Unperturbed equilibrium run.h5 (same dynamics)",
    )
    demo_toggle = mo.ui.checkbox(value=mo.app_meta().mode == "script", label="Use synthetic validation pair")
    mo.vstack([mo.hstack([perturbed_picker, equilibrium_picker], widths="equal"), demo_toggle])
    return equilibrium_picker, demo_toggle, perturbed_picker


@app.cell
def _(equilibrium_picker, demo_toggle, perturbed_picker):
    mo.stop(
        not demo_toggle.value and (not perturbed_picker.value or not equilibrium_picker.value),
        mo.md("Select both HDF5 files above, or enable the synthetic validation pair."),
    )
    _load_error = None
    if demo_toggle.value:
        perturbed_run, equilibrium_run = demo_pair()
    else:
        _perturbed_path = perturbed_picker.path(0)
        _equilibrium_path = equilibrium_picker.path(0)
        mo.stop(_perturbed_path.resolve() == _equilibrium_path.resolve(), mo.callout("Select two different runs.", kind="warn"))
        try:
            perturbed_run = read_psi_run(_perturbed_path)
            equilibrium_run = read_psi_run(_equilibrium_path)
        except (OSError, KeyError, ValueError) as _exc:
            _load_error = str(_exc)
    mo.stop(_load_error is not None, mo.callout(f"Could not load the selected runs: {_load_error}", kind="warn"))
    return equilibrium_run, perturbed_run


@app.cell
def _(equilibrium_run, demo_toggle, perturbed_run):
    time = perturbed_run["time"]
    z = perturbed_run["z"]
    perturbed_psi = perturbed_run["psi"]
    _eq_saved = equilibrium_run["psi"]
    _eq_time = equilibrium_run["time"]
    mo.stop(
        time.ndim != 1 or z.ndim != 1 or time.size < 3 or z.size < 2
        or perturbed_psi.shape != (time.size, z.size)
        or _eq_time.ndim != 1 or _eq_time.size == 0
        or _eq_saved.shape != (_eq_time.size, z.size)
        or not np.array_equal(z, equilibrium_run["z"]),
        mo.callout("Runs must have matching z grids and ψ arrays of shape (time, z). The perturbed run needs at least three frames; the equilibrium needs at least one.", kind="warn"),
    )
    mo.stop(
        not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0)
        or not np.all(np.isfinite(_eq_time)) or not np.all(np.diff(_eq_time) > 0)
        or not np.all(np.isfinite(z)) or not np.all(np.diff(z) > 0)
        or not np.all(np.isfinite(perturbed_psi)) or not np.all(np.isfinite(_eq_saved)),
        mo.callout("Time and z must increase strictly; all selected data must be finite.", kind="warn"),
    )
    mo.stop(
        not np.allclose(np.diff(z), np.mean(np.diff(z)), rtol=1e-7, atol=1e-14),
        mo.callout("The spatial DCT requires a uniform z grid.", kind="warn"),
    )
    _differences, _missing = compare_dynamics(perturbed_run, equilibrium_run)
    mo.stop(bool(_differences), mo.callout(
        "These runs use different dynamics: " + ", ".join(_differences)
        + ". Select an unperturbed equilibrium and its perturbed run with identical model parameters, including interaction strength/kernel.", kind="warn"))
    _rho_p, _rho_e = perturbed_run["rho"], equilibrium_run["rho"]
    mo.stop(_rho_p is not None and _rho_e is not None and not np.array_equal(_rho_p, _rho_e),
        mo.callout("The rho grids must match for the same dynamical system.", kind="warn"))
    _notes = []
    if _missing:
        _notes.append("Missing metadata; equality of these dynamics settings could not be checked: " + ", ".join(sorted(_missing)))
    if perturbed_run["run"].get("DT") != equilibrium_run["run"].get("DT"):
        _notes.append("Solver DT differs; discretization errors can differ even with identical physical model parameters.")
    _phi_p, _phi_e = perturbed_run["phi_initial"], equilibrium_run["phi_initial"]
    _phi_summary = ""
    if _phi_p is not None and _phi_e is not None:
        mo.stop(_phi_p.shape != _phi_e.shape, mo.callout("Initial φ grids must have the same shape.", kind="warn"))
        _perturbation = np.linalg.norm(_phi_p - _phi_e) / max(np.linalg.norm(_phi_e), np.finfo(float).tiny)
        _phi_summary = f" Initial φ perturbation relative norm: `{_perturbation:.3e}`."
        if np.array_equal(_phi_p, _phi_e):
            _notes.append("The saved initial φ fields are identical. Check that the perturbed file actually includes your perturbation.")
    elif not demo_toggle.value:
        _notes.append("Initial φ is unavailable in at least one file; the initial perturbation cannot be checked.")
    _relative_drift = np.max(np.linalg.norm(_eq_saved - _eq_saved[0], axis=1)) / max(np.linalg.norm(_eq_saved[0]), np.finfo(float).tiny)
    if _relative_drift > 1e-6:
        _notes.append("The chosen equilibrium ψ changes during its run. A relative drift above 10⁻⁶ suggests it is not stationary: verify this equilibrium before interpreting eigenmodes. Subtraction still uses its first profile held fixed.")
    _phi_drift = equilibrium_run["phi_endpoint_relative_drift"]
    if _phi_drift is not None and _phi_drift > 1e-6:
        _notes.append(f"Equilibrium φ differs between its first and last saved fields (relative norm {_phi_drift:.3e}); verify stationarity of φ as well as ψ.")
    equilibrium_psi = np.broadcast_to(_eq_saved[0], perturbed_psi.shape)
    _summary = mo.md(
        f"**Perturbed:** `{perturbed_run['path']}`  \n{perturbed_run['dynamics_label']}  \n"
        f"**Equilibrium:** `{equilibrium_run['path']}`  \n{equilibrium_run['dynamics_label']}  \n"
        f"`{time.size}` perturbed frames × `{z.size}` spatial points. "
        f"Maximum equilibrium ψ drift relative to its first profile: `{_relative_drift:.3e}`."
        + _phi_summary + " Subtraction uses the fixed first equilibrium profile, independent of its saved times."
    )
    mo.vstack([_summary] + [mo.callout(_note, kind="warn") for _note in _notes])
    return equilibrium_psi, perturbed_psi, time, z


@app.cell
def _(time):
    end_time_slider = mo.ui.slider(
        steps=time.tolist(), value=float(time[max(2, (len(time) - 1) // 3)]),
        label="Visually selected end of the linear regime, t [s]",
        show_value=True, full_width=True, debounce=True,
    )
    end_time_slider
    return (end_time_slider,)


@app.cell
def _(end_time_slider, time):
    end_index = int(np.searchsorted(time, end_time_slider.value))
    end_time = float(time[end_index])
    mo.md(f"Selected endpoint: **t = {end_time:.8g} s** (saved frame `{end_index}`); **{end_index + 1} frames** included from t = {time[0]:.8g} s.")
    return end_index, end_time


@app.cell
def _(equilibrium_psi, end_time, perturbed_psi, time, z):
    _figure, _axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True, constrained_layout=True)
    for _axis, _psi, _title in zip(_axes, (perturbed_psi, equilibrium_psi), ("Perturbed equilibrium", "Equilibrium ψ(z, t₀), held fixed")):
        # Only the display is downsampled; the DCT/Hankel uses every saved frame.
        _ti = np.unique(np.linspace(0, len(time) - 1, min(1500, len(time)), dtype=int))
        _zi = np.unique(np.linspace(0, len(z) - 1, min(1000, len(z)), dtype=int))
        plot_psi_arrays(
            _psi[np.ix_(_ti, _zi)], time[_ti], z[_zi], ax=_axis,
            cmap=get_rbc_cmap(), vmin=0.01, vmax=100, title=_title,
            add_colorbar=False,
        )
        _axis.axvline(end_time, color="#00e5ff", linewidth=2, linestyle="--", label="Linear-regime end")
        _axis.axvspan(end_time, time[-1], color="black", alpha=0.12)
        _axis.legend(loc="upper right")
    _figure.colorbar(_axes[0].images[0], ax=list(_axes), label=r"$\psi\;[\%]$")
    _output = mo.as_html(_figure)
    plt.close(_figure)
    _output
    return


@app.cell
def _(equilibrium_psi, end_index, perturbed_psi, time):
    mo.stop(end_index < 2, mo.callout("Select at least three saved frames for shift invariance.", kind="warn"))
    linear_time = time[:end_index + 1]
    delta_psi = perturbed_psi[:end_index + 1] - equilibrium_psi[:end_index + 1]
    dct_vectors = dct(delta_psi, type=2, norm="ortho", axis=1)
    mo.md(f"Original δψ and DCT arrays: **N = {dct_vectors.shape[0]}**, **M = {dct_vectors.shape[1]}**, modes `0 … {dct_vectors.shape[1] - 1}`.")
    return dct_vectors, delta_psi, linear_time


@app.cell
def _(dct_vectors):
    mode_range = mo.ui.range_slider(
        start=0, stop=dct_vectors.shape[1] - 1, step=1,
        value=[0, dct_vectors.shape[1] - 1],
        label="DCT mode range to analyze [m_min, m_max], both endpoints included",
        show_value=True, full_width=True, debounce=True,
    )
    mode_range
    return (mode_range,)


@app.cell
def _(dct_vectors, mode_range):
    mode_min, mode_max = map(int, mode_range.value)
    selected_modes = np.arange(mode_min, mode_max + 1)
    selected_dct_vectors = dct_vectors[:, mode_min:mode_max + 1]
    mo.md(f"Analyzing **{len(selected_modes)} DCT modes**, from **m = {mode_min}** through **m = {mode_max}**, inclusive. "
        "The heatmaps and full DCT arrays retain the complete spatial field.")
    return mode_max, mode_min, selected_dct_vectors, selected_modes


@app.cell
def _(dct_vectors, delta_psi, end_time, linear_time, selected_dct_vectors, z):
    _figure, _axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    _limit = max(float(np.max(np.abs(delta_psi))), np.finfo(float).tiny)
    _image = _axes[0].imshow(
        delta_psi.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-_limit, vmax=_limit,
        extent=(linear_time[0], end_time, 100 * z[0], 100 * z[-1]),
    )
    _axes[0].set(xlabel="t [s]", ylabel="z [cm]", title="δψ over the selected linear regime")
    _figure.colorbar(_image, ax=_axes[0], label="δψ [fraction]")
    _axes[1].plot(linear_time, np.linalg.norm(dct_vectors, axis=1), label="All DCT modes / full δψ norm")
    _axes[1].plot(linear_time, np.linalg.norm(selected_dct_vectors, axis=1), "--", label="Selected DCT modes")
    _axes[1].set(xlabel="t [s]", ylabel="Coefficient-vector norm", title="Difference-signal amplitude")
    _axes[1].legend()
    _axes[1].grid(alpha=0.25)
    _output = mo.as_html(_figure)
    plt.close(_figure)
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Diagnostic: single-exponential fits to individual DCT coefficients

    For each spatial cosine mode $m$, fit its **signed** DCT coefficient over
    the selected time window:

    $$c_m(t)=[\operatorname{DCT\!\!\!-II}(\delta\psi(t,:))]_m
    \approx A_m e^{\gamma_m(t-t_0)}+C_m.$$

    **$\gamma_m$ is the reported rate**, in s⁻¹. Positive means that the
    exponential component grows; negative means that it decays towards the
    fitted constant $C_m$. For example, $\gamma_m=0.02$ gives an e-folding time
    of 50 s. The offset allows $c_m(t_0)=0$ and avoids taking a logarithm of
    zero. Mode 0 represents the spatial mean; larger $m$ describe finer cosine
    patterns. Modes are defined on the entire stored z grid, including wings.

    A small fit error means one exponential plus a constant describes that
    coefficient well. Flat, negligible, or nearly linear traces may not
    determine a rate; these rows are marked **unresolved**. A poor fit can
    indicate several growth rates, oscillations, or a window outside the linear
    regime. A fitted DCT-mode rate is an observed rate; it is an equilibrium
    eigenvalue only when that spatial mode evolves independently.
    """)
    return


@app.cell
def _(equilibrium_psi, dct_vectors, demo_toggle, perturbed_psi, linear_time, selected_dct_vectors, selected_modes):
    mo.stop(len(linear_time) < 5, mo.callout("Select at least five saved frames to fit individual DCT-mode rates.", kind="warn"))
    # Roundoff in subtracting two O(psi) fields can dominate unexcited modes.
    coefficient_floor = 100 * np.finfo(float).eps * max(
        float(np.max(np.abs(perturbed_psi[:len(linear_time)]))),
        float(np.max(np.abs(equilibrium_psi[:len(linear_time)])))
    ) * np.sqrt(dct_vectors.shape[1])
    mode_fits = [fit_dct_mode(selected_dct_vectors[:, _column], linear_time, amplitude_floor=coefficient_floor)
        for _column in range(len(selected_modes))]
    mode_rate_table = pd.DataFrame({
        "DCT_mode_m": selected_modes,
        "growth_rate_s^-1": [_fit.rate for _fit in mode_fits],
        "fit_rate_std_s^-1": [_fit.rate_std for _fit in mode_fits],
        "relative_fit_error": [_fit.relative_rmse for _fit in mode_fits],
        "R_squared": [_fit.r_squared for _fit in mode_fits],
        "max_abs_coefficient": [_fit.max_abs_coefficient for _fit in mode_fits],
        "status": [_fit.status for _fit in mode_fits],
    })
    _demo_hint = [mo.callout("Synthetic validation: DCT modes 3, 5, and 7 mix two dynamic patterns with shared rates +0.12 and +0.06 s⁻¹. Their individual DCT fits need not equal either rate. ESPRIT with rank 2 should recover both patterns.", kind="info")] if demo_toggle.value else []
    _resolved_count = int(np.count_nonzero(mode_rate_table["status"] == "ok"))
    _window_hint = [mo.callout("No mode has a resolved single-exponential fit over this window. Inspect the selected coefficient and try an earlier endpoint; saturation, oscillations, or mixed dynamics can prevent one-rate fits.", kind="warn")] if _resolved_count == 0 else []
    mo.vstack(_demo_hint + _window_hint + [
        mo.md(f"**{_resolved_count} of {len(mode_fits)} DCT modes have a resolved fit over this window.**"),
        mo.md("**Each row is one spatial DCT mode.** Read `growth_rate_s^-1` together with `status` and `relative_fit_error`. "
            "Blank rates mean the data did not determine a rate. `ok` means relative fit error ≤ 10% and resolved curvature; "
            "it does not prove independent modal evolution. The reported standard error assumes uncorrelated fit residuals."),
        mo.ui.table(mode_rate_table, selection=None, show_download=True, label="Diagnostic DCT coefficient-fit rates; these are not ESPRIT results"),
    ])
    return coefficient_floor, mode_fits, mode_rate_table


@app.cell
def _(mode_max, mode_min, selected_dct_vectors, selected_modes):
    mode_selector = mo.ui.slider(start=mode_min, stop=mode_max, step=1,
        value=int(selected_modes[np.argmax(np.max(np.abs(selected_dct_vectors), axis=0))]),
        label="DCT mode m to inspect (mode 0 is the spatial mean)", show_value=True, debounce=True)
    mode_selector
    return (mode_selector,)


@app.cell
def _(dct_vectors, linear_time, mode_fits, mode_min, mode_selector):
    _m = int(mode_selector.value)
    _fit = mode_fits[_m - mode_min]
    _figure, _axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    _axes[0].plot(linear_time, dct_vectors[:, _m], label=f"Measured DCT coefficient, m={_m}")
    _rate_label = f"γ = {_fit.rate:.5g} s⁻¹" if np.isfinite(_fit.rate) else "Rate unresolved"
    _axes[0].plot(linear_time, _fit.fitted, "--", label=f"Exponential + offset fit; {_rate_label}")
    _axes[0].set(xlabel="t [s]", ylabel=f"Signed coefficient c{_m}(t)", title=f"Does one exponential describe DCT mode {_m}?")
    _axes[0].legend(fontsize=9)
    _axes[0].grid(alpha=0.25)
    _axes[1].plot(linear_time, dct_vectors[:, _m] - _fit.fitted)
    _axes[1].axhline(0, color="grey", linewidth=1)
    _axes[1].set(xlabel="t [s]", ylabel="Measured − fitted coefficient", title="Residual: smaller is better")
    _axes[1].grid(alpha=0.25)
    _figure.suptitle(f"Mode {_m}: {_fit.status}; relative fit error = {_fit.relative_rmse:.3g}")
    _output = mo.as_html(_figure)
    plt.close(_figure)
    _output
    return


@app.cell
def _(mode_max, mode_min, mode_rate_table):
    _figure, _axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    _ok = mode_rate_table["status"] == "ok"
    _poor = np.isfinite(mode_rate_table["growth_rate_s^-1"]) & ~_ok
    _axis.scatter(mode_rate_table.loc[_ok, "DCT_mode_m"], mode_rate_table.loc[_ok, "growth_rate_s^-1"], label="Resolved fit")
    _axis.scatter(mode_rate_table.loc[_poor, "DCT_mode_m"], mode_rate_table.loc[_poor, "growth_rate_s^-1"], marker="x", color="tab:orange", label="Poor single-exponential fit")
    _axis.axhline(0, color="grey", linewidth=1)
    _axis.set_xlim(mode_min - 0.5, mode_max + 0.5)
    _axis.set(xlabel="Spatial DCT mode number m", ylabel="Coefficient-fit rate γₘ [s⁻¹]", title="Diagnostic DCT coefficient fits — not ESPRIT eigenvalues")
    _axis.legend()
    _axis.grid(alpha=0.25)
    _output = mo.as_html(_figure)
    plt.close(_figure)
    _output
    return


@app.cell
def _(equilibrium_run, coefficient_floor, dct_vectors, delta_psi, end_index, end_time, perturbed_run, linear_time, mode_fits, mode_max, mode_min, mode_rate_table, selected_dct_vectors, selected_modes, z):
    mode_selection_metadata = {
        "perturbed_file": perturbed_run["path"], "equilibrium_file": equilibrium_run["path"],
        "linear_start_time_s": float(linear_time[0]), "linear_end_time_s": end_time,
        "linear_end_saved_frame": end_index, "endpoint_included": True,
        "N_saved_frames": len(linear_time), "M_DCT_modes": len(z),
        "dct_mode_min": mode_min, "dct_mode_max": mode_max, "mode_range_inclusive": True,
        "M_selected_DCT_modes": len(selected_modes),
        "dct_type": 2, "dct_norm": "ortho", "fit_model": "A_m * exp(gamma_m * (t-t0)) + C_m",
        "method": "independent_DCT_coefficient_exponential_fits",
        "reference": "fixed_first_equilibrium_psi_profile",
        "fit_uses_original_saved_times": True, "time_resampled": False,
        "coefficient_precision_floor": coefficient_floor,
        "dimensionless_rate_search_bounds": [-50, 50],
        "max_relative_fit_error_for_ok_status": 0.1,
    }
    _metadata_json = json.dumps(mode_selection_metadata, indent=2)

    def make_mode_archive():
        buffer = io.BytesIO()
        np.savez_compressed(buffer, metadata_json=_metadata_json, time=linear_time, z=z,
            delta_psi=delta_psi, dct_vectors=dct_vectors,
            selected_modes=selected_modes, selected_dct_vectors=selected_dct_vectors,
            mode_rates=np.array([_fit.rate for _fit in mode_fits]),
            mode_rate_std=np.array([_fit.rate_std for _fit in mode_fits]),
            mode_fit_status=np.array([_fit.status for _fit in mode_fits]),
            mode_fitted_coefficients=np.column_stack([_fit.fitted for _fit in mode_fits]))
        return buffer.getvalue()

    mo.hstack([
        mo.download(_metadata_json, filename="dct_linear_regime_selection.json", label="Save selected regime"),
        mo.download(mode_rate_table.to_csv(index=False), filename="dct_mode_growth_rates.csv", label="Download rates per DCT mode"),
        mo.download(make_mode_archive, filename="dct_mode_growth_analysis.npz", label="Download DCT coefficients and fits"),
    ])
    return (mode_selection_metadata,)


@app.cell
def _():
    advanced_toggle = mo.ui.checkbox(value=True, label="Compute ESPRIT growth rates (Hankel / SVD)")
    advanced_toggle
    return (advanced_toggle,)


@app.cell
def _(advanced_toggle, linear_time, selected_dct_vectors):
    mo.stop(not advanced_toggle.value)
    sampling = prepare_uniform_observations(selected_dct_vectors, linear_time)
    hankel_time = sampling.time
    hankel_vectors = sampling.values
    saved_dt = uniform_step(hankel_time)
    _intervals = np.diff(linear_time)
    _message = (
        f"Saved intervals: min **{np.min(_intervals):.8g} s**, max **{np.max(_intervals):.8g} s**. "
        f"Hankel Δt = **{saved_dt:.8g} s** (rather than solver DT). "
    )
    if sampling.resampled:
        _message += (
            f"Saved times are irregular (maximum interval deviation from their mean: {sampling.max_relative_step_deviation:.3%}). "
            f"For shift invariance, DCT vectors are linearly interpolated onto **{len(hankel_time)} evenly spaced times** "
            "with the same first and last times. Original δψ/DCT arrays are preserved in the download. "
            "Interpolation is approximate, especially for fast oscillations or large sampling gaps."
        )
    else:
        _message += "Saved times are uniform; original DCT vectors are used directly."
    mo.callout(_message, kind="info")
    return hankel_time, hankel_vectors, sampling, saved_dt


@app.cell(hide_code=True)
def _(advanced_toggle):
    mo.stop(not advanced_toggle.value)
    mo.md(r"""
    ## ESPRIT: growth rates of extracted temporal components

    ESPRIT searches for components that share a single temporal law across
    the selected DCT coefficients:

    $$c_m(t)\approx\sum_{q=1}^{r}b_{mq}
    e^{(\gamma_q+i\omega_q)(t-t_0)}.$$

    **One component $q$ is a combination of spatial DCT modes.** Its
    $\gamma_q$ describes the growth/decay of that entire combination, and
    $\omega_q$ describes its oscillation. Component numbers label the extracted
    exponents, sorted by decreasing growth; they are not spatial mode numbers
    and may change when the endpoint or rank changes. Positive growth means an
    envelope $e^{\gamma_q(t-t_0)}$ that increases; negative growth means decay;
    near-zero growth means an approximately steady envelope.

    For example, $\gamma=0.02\,\mathrm{s}^{-1}$ means the envelope doubles in
    $\ln(2)/0.02\approx34.7$ s. If $\omega\ne0$, it oscillates while growing or
    decaying, with period $2\pi/|\omega|$. Complex conjugate exponents normally
    appear as two entries with equal growth and opposite frequency: **together
    they represent one real oscillating pattern**, not two separate spatial
    instabilities. A rate close to zero can also reflect a conserved direction
    or a constant bias in the equilibrium profile.

    The growth plot below shows these **shift-operator eigenvalue estimates**.
    The DCT-content heatmap shows how the fitted coefficient vector $b_{:q}$
    is distributed across the selected spatial modes. These are observable
    temporal components of $\delta\psi$; interpreting them as eigenmodes of a
    phi linearization requires a suitable equilibrium/reference and a linear
    time window. Modes not excited or not visible in the selected coefficients
    cannot be recovered.

    Here the retained SVD rank sets the number of temporal components, **not**
    the number of spatial DCT modes. A component can contain many cosine modes.
    The “dominant DCT mode” is only the strongest contribution, not an assignment
    of that growth rate to a single DCT mode.

    With $c_j\in\mathbb R^{M_{\rm selected}}$ containing only the selected DCT modes
    at saved time $t_j$, choose $L$ delay
    blocks and $K=N-L+1$ columns:

    $$H=\begin{bmatrix}c_0&c_1&\cdots&c_{K-1}\\
    c_1&c_2&\cdots&c_K\\ \vdots&\vdots&&\vdots\\
    c_{L-1}&c_L&\cdots&c_{N-1}\end{bmatrix}=U\Sigma V^*.$$

    Retain $U_r$. Remove its last $M_{\rm selected}$ rows to form $U_{\rm top}$ and its first
    $M_{\rm selected}$ rows to form $U_{\rm bottom}$. Least squares gives
    $R=U_{\rm top}^\dagger U_{\rm bottom}$. For each discrete pole
    $\lambda_q\in\operatorname{eig}(R)$,

    $$s_q=\frac{\log\lambda_q}{\Delta t},\qquad
    \gamma_q=\Re(s_q)=\frac{\log|\lambda_q|}{\Delta t}.$$

    Positive $\gamma$ means growth; negative $\gamma$ means decay. The imaginary
    part is angular frequency (rad/s), using the principal logarithm and subject
    to the saved-time Nyquist limit. Extracted components may mix DCT modes.
    This is the subspace shift-invariance principle of
    [Roy & Kailath's ESPRIT method](https://alumni.media.mit.edu/~aggelos/papers/roykailath89.pdf).
    When saved times are irregular, $c_j$ and $\Delta t$ refer to the interpolated
    uniform grid reported above.
    """)
    return


@app.cell
def _(hankel_vectors):
    delay_control = mo.ui.number(start=2, stop=len(hankel_vectors) - 1, step=1,
        value=min(20, max(2, len(hankel_vectors) // 3)), label="L: delay blocks").form(submit_button_label="Compute Hankel SVD")
    cutoff_control = mo.ui.dropdown(
        options={"10⁻⁸": 1e-8, "10⁻¹⁰": 1e-10, "10⁻¹²": 1e-12, "10⁻¹⁴": 1e-14},
        value="10⁻¹⁰", label="Relative numerical-rank cutoff",
    )
    mo.hstack([delay_control, cutoff_control])
    return cutoff_control, delay_control


@app.cell
def _(cutoff_control, delay_control, hankel_vectors):
    # The default runs once; editing the form recomputes only on submission.
    delays = int(delay_control.value) if delay_control.value is not None else min(20, max(2, len(hankel_vectors) // 3))
    _n, _m = hankel_vectors.shape
    _k = _n - delays + 1
    _mib = delays * _m * _k * 8 / 2**20
    mo.stop(_mib > 512, mo.callout(f"H alone would use {_mib:.1f} MiB, with additional SVD workspace. Reduce L or the selected time window (not the saved sampling).", kind="warn"))
    subspace = hankel_subspace(hankel_vectors, delays, relative_cutoff=cutoff_control.value)
    mo.md(f"H shape: **{subspace.shape[0]} × {subspace.shape[1]}**, {_mib:.1f} MiB before SVD workspace. Numerical rank at the selected relative cutoff: **{subspace.numerical_rank}**.")
    return delays, subspace


@app.cell
def _(subspace):
    mo.stop(subspace.numerical_rank == 0, mo.callout("The selected difference signal is identically zero; no growth rates are identifiable.", kind="warn"))
    _max_rank = min(subspace.numerical_rank, (subspace.delays - 1) * subspace.n_modes)
    rank_control = mo.ui.slider(start=1, stop=_max_rank, step=1, value=min(3, _max_rank),
        label="r: retained SVD rank / number of ESPRIT exponents", show_value=True, debounce=True)
    rank_control
    return (rank_control,)


@app.cell
def _(rank_control, subspace):
    _values = subspace.singular_values / subspace.singular_values[0]
    _figure, _axes = plt.subplots(1, 2, figsize=(12, 3.5), constrained_layout=True)
    _indices = np.arange(1, len(_values) + 1)
    _axes[0].semilogy(_indices, np.maximum(_values, np.finfo(float).tiny), ".-")
    _axes[0].axvline(rank_control.value, color="tab:red", linestyle="--")
    _axes[0].set(xlabel="Singular-value index", ylabel="σᵢ / σ₁", title="Hankel singular values")
    _axes[1].plot(_indices, np.cumsum(_values**2) / np.sum(_values**2))
    _axes[1].axvline(rank_control.value, color="tab:red", linestyle="--")
    _axes[1].set(xlabel="Retained rank", ylabel="Retained squared singular-value fraction", ylim=(0, 1.01))
    _output = mo.as_html(_figure)
    plt.close(_figure)
    _output
    return


@app.cell
def _(hankel_time, hankel_vectors, rank_control, selected_modes, subspace, z):
    _fit_error = None
    try:
        growth = estimate_growth(hankel_vectors, hankel_time, subspace, int(rank_control.value))
    except ValueError as _exc:
        _fit_error = str(_exc)
    mo.stop(_fit_error is not None, mo.callout(_fit_error or "", kind="warn"))
    dynamic_modes = recover_dynamic_modes(growth, selected_modes, len(z))
    dynamic_time_factors = growth.discrete_eigenvalues[None, :] ** np.arange(len(hankel_time))[:, None]
    growth_table = pd.DataFrame({
        "component": np.arange(1, len(growth.exponents) + 1),
        "growth_rate_s^-1": growth.exponents.real,
        "angular_frequency_rad_s^-1": growth.exponents.imag,
        "oscillation_period_s": np.divide(
            2 * np.pi, np.abs(growth.exponents.imag),
            out=np.full(len(growth.exponents), np.nan),
            where=np.abs(growth.exponents.imag) > 1e-10 / (hankel_time[1] - hankel_time[0]),
        ),
        "lambda_real": growth.discrete_eigenvalues.real,
        "lambda_imag": growth.discrete_eigenvalues.imag,
        "lambda_abs": np.abs(growth.discrete_eigenvalues),
        "dominant_DCT_mode": selected_modes[np.argmax(np.abs(growth.spatial_amplitudes), axis=0)],
        "spatial_amplitude_norm": np.linalg.norm(growth.spatial_amplitudes, axis=0),
        "finite_continuous_rate": np.isfinite(growth.exponents),
        "conjugate_partner_component": np.where(dynamic_modes.conjugate_partners >= 0,
            dynamic_modes.conjugate_partners + 1, np.nan),
    })
    mo.vstack([
        mo.md(f"**Retained Hankel energy:** {growth.retained_energy:.6%}  \n"
            f"**Relative shift residual:** {growth.shift_residual:.3e}  \n"
            f"**Shifted-subspace condition number:** {growth.shift_condition:.3e}  \n"
            f"**Relative global exponential reconstruction error:** {growth.reconstruction_error:.3e}"),
        mo.ui.table(growth_table, selection=None, show_download=True, label="ESPRIT component growth rates and oscillation frequencies"),
    ])
    return dynamic_modes, dynamic_time_factors, growth, growth_table


@app.cell
def _(growth):
    # A component's fitted amplitude includes excitation and complex phase.
    # Normalize each column stably to show its relative DCT content only.
    _magnitudes = np.abs(growth.spatial_amplitudes)
    _peaks = np.max(_magnitudes, axis=0)
    _scaled = np.divide(_magnitudes, _peaks, out=np.zeros_like(_magnitudes), where=_peaks > 0)
    _powers = _scaled ** 2
    esprit_dct_weights = np.divide(_powers, np.sum(_powers, axis=0), out=np.zeros_like(_powers), where=np.sum(_powers, axis=0) > 0)
    return (esprit_dct_weights,)


@app.cell
def _(esprit_dct_weights, growth, mode_max, mode_min):
    _components = np.arange(1, len(growth.exponents) + 1)
    _finite = np.isfinite(growth.exponents)
    _figure, _axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    _rates = growth.exponents.real
    _colors = np.where(_rates >= 0, "tab:blue", "tab:red")
    _axes[0].scatter(_components[_finite], _rates[_finite], color=_colors[_finite], s=65)
    for _q in np.flatnonzero(_finite):
        _axes[0].vlines(_components[_q], 0, _rates[_q], color=_colors[_q], alpha=0.5)
        _axes[0].annotate(f"{_rates[_q]:.4g}", (_components[_q], _rates[_q]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    _axes[0].axhline(0, color="grey", linewidth=1)
    _axes[0].set_xticks(_components)
    _axes[0].set_xlim(0.5, len(_components) + 0.5)
    _axes[0].set(xlabel="ESPRIT component q (not DCT mode m)", ylabel="Extracted growth rate γq [s⁻¹]", title="ESPRIT growth rates from shift invariance")
    _axes[0].grid(axis="y", alpha=0.25)
    _image = _axes[1].imshow(esprit_dct_weights, origin="lower", aspect="auto", cmap="viridis",
        vmin=0, vmax=max(float(np.max(esprit_dct_weights)), np.finfo(float).tiny),
        extent=(0.5, len(_components) + 0.5, mode_min - 0.5, mode_max + 0.5))
    _axes[1].set_xticks(_components)
    _axes[1].set(xlabel="ESPRIT component q", ylabel="Spatial DCT mode number m", title="DCT content of each ESPRIT component")
    _figure.colorbar(_image, ax=_axes[1], label="Normalized squared fitted coefficient magnitude")
    _output = mo.as_html(_figure)
    plt.close(_figure)
    mo.vstack([_output, mo.md(r"""
        **Left:** one point per finite extracted exponent, ordered by growth rate.
        A zero discrete pole has no finite continuous-time rate and is omitted. Two
        conjugate components can have the same growth rate; check their opposite
        frequencies in the table. Their envelope grows/decays at that shared rate.

        **Right:** each column is the DCT content of that same component,
        $|b_{mq}|^2/\sum_m|b_{mq}|^2$, normalized over the selected modes. A bright
        entry means that cosine pattern contributes strongly. Several bright rows
        mean the component mixes spatial DCT modes. Normalization removes overall
        excitation strength, so a bright column need not be strongly excited or
        reliable. These weights omit phase and are not separate energy fractions
        that can be added across components. The fitted amplitudes in the download
        preserve magnitude and phase.
        """)])
    return


@app.cell
def _(dynamic_modes, growth, hankel_time):
    _options = {
        f"q = {_q + 1}: γ = {_s.real:.5g} s⁻¹, ω = {_s.imag:.5g} rad/s"
        + (f" (pair with q = {dynamic_modes.conjugate_partners[_q] + 1})"
           if dynamic_modes.conjugate_partners[_q] >= 0 else ""): _q
        for _q, _s in enumerate(growth.exponents)
    }
    dynamic_mode_selector = mo.ui.dropdown(options=_options, value=next(iter(_options)),
        label="ESPRIT dynamic component to inspect")
    dynamic_time_slider = mo.ui.slider(steps=hankel_time.tolist(), value=float(hankel_time[0]),
        label="Time to display this component's contribution [s]", show_value=True,
        full_width=True, debounce=True)
    mo.vstack([mo.md(r"""
        ## Dynamic-mode explorer: structure + shared growth rate

        The ESPRIT exponents are fitted jointly to the selected DCT signals to
        recover complex coefficient vectors $b_{:q}$. Inverse DCT gives each
        observed spatial pattern. We write

        $$\delta\psi(z,t)\approx\sum_q a_q u_q(z)
        e^{(\gamma_q+i\omega_q)(t-t_0)}.$$

        **$u_q(z)$ is the structure; $\gamma_q$ is the shared growth rate of that
        entire structure.** The displayed shape has unit norm and its largest
        spatial entry is made real and positive. The complex $a_q$ retains the
        initial excitation and phase, so this normalization does not change the
        fitted contribution. For a nonoscillating component the shape is real;
        a complex shape has two spatial quadratures, shown as real and imaginary
        curves. A conjugate pair contributes
        $2\Re[a_q u_q e^{(\gamma_q+i\omega_q)(t-t_0)}]$ to a real field.

        The time slider shows the actual fitted contribution to **δψ**, with its
        conjugate partner included when available. Selecting either member of a
        pair shows the same real pattern. The envelope grows at $\gamma_q$ even
        when individual points or coefficients change sign during oscillation.

        These structures are **ψ projections of dynamic modes**, restricted to
        your selected DCT band. Coefficients outside that band are set to zero
        for the inverse DCT; widening the range can reveal more structure. Since
        ψ projects φ, this analysis cannot uniquely reconstruct the full φ
        eigenfunction. Equal eigenvalues cannot be separated into distinct
        eigenfunctions from a single trajectory. Stability across windows/ranks
        is needed before treating fitted components as linear eigenmodes.
        """), dynamic_mode_selector, dynamic_time_slider])
    return dynamic_mode_selector, dynamic_time_slider


@app.cell
def _(dynamic_mode_selector, dynamic_modes, dynamic_time_factors, dynamic_time_slider, esprit_dct_weights, growth, hankel_time, selected_modes, z):
    _q = int(dynamic_mode_selector.value)
    _s = growth.exponents[_q]
    _partner = int(dynamic_modes.conjugate_partners[_q])
    _members = [_q, _partner] if _partner >= 0 else [_q]
    _j = int(np.argmin(np.abs(hankel_time - dynamic_time_slider.value)))
    _shape = dynamic_modes.psi_shapes[:, _q]
    _coefficients = dynamic_time_factors[:, _members] @ growth.spatial_amplitudes[:, _members].T
    _snapshot_dct = np.zeros(len(z), dtype=complex)
    _snapshot_dct[selected_modes] = _coefficients[_j]
    _snapshot = idct(_snapshot_dct, type=2, norm="ortho")
    _envelope = np.sum(np.abs(dynamic_time_factors[:, _members])
        * np.abs(dynamic_modes.amplitudes[_members]), axis=1)
    _figure, _axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    _axes[0, 0].plot(100 * z, _shape.real, label="Re u(z)")
    _axes[0, 0].plot(100 * z, _shape.imag, "--", label="Im u(z)")
    _axes[0, 0].axhline(0, color="grey", linewidth=0.7)
    _axes[0, 0].set(xlabel="z [cm]", ylabel="Normalized shape", title=f"Dynamic structure q = {_q + 1} (unit norm)")
    _axes[0, 0].legend()
    _axes[0, 1].bar(selected_modes, esprit_dct_weights[:, _q])
    _axes[0, 1].set(xlabel="Spatial DCT mode m", ylabel="Normalized squared coefficient magnitude",
        title=f"Cosines sharing γ = {_s.real:.5g} s⁻¹", ylim=(0, 1.05))
    _label = f"q = {_q + 1}" + (f" + {_partner + 1}" if _partner >= 0 else "")
    _axes[1, 0].plot(100 * z, _snapshot.real)
    _axes[1, 0].axhline(0, color="grey", linewidth=0.7)
    _axes[1, 0].set(xlabel="z [cm]", ylabel="Fitted contribution to δψ [fraction]",
        title=f"{_label} at t = {hankel_time[_j]:.5g} s")
    _axes[1, 1].plot(hankel_time, _envelope, label="Exponential envelope (sum for a pair)")
    _axes[1, 1].plot(hankel_time, np.linalg.norm(_coefficients.real, axis=1), "--",
        label="Norm of real component/pair contribution")
    _axes[1, 1].axvline(hankel_time[_j], color="grey", linestyle=":")
    _axes[1, 1].set(xlabel="t [s]", ylabel="Amplitude / coefficient norm",
        title=f"Shared γ = {_s.real:.5g} s⁻¹; ω = {_s.imag:.5g} rad/s")
    _axes[1, 1].legend(fontsize=9)
    for _axis in _axes.flat:
        _axis.grid(alpha=0.2)
    _output = mo.as_html(_figure)
    plt.close(_figure)
    _length = len(z) * float(np.mean(np.diff(z)))
    _wavelengths = np.divide(2 * _length, selected_modes,
        out=np.full(len(selected_modes), np.inf), where=selected_modes > 0)
    dynamic_dct_table = pd.DataFrame({
        "DCT_mode_m": selected_modes,
        "wavelength_um": 1e6 * _wavelengths,
        "relative_DCT_weight": esprit_dct_weights[:, _q],
        "normalized_coefficient_real": dynamic_modes.dct_shapes[:, _q].real,
        "normalized_coefficient_imag": dynamic_modes.dct_shapes[:, _q].imag,
    })
    dynamic_spatial_table = pd.DataFrame({
        "z_m": z, "normalized_psi_shape_real": _shape.real,
        "normalized_psi_shape_imag": _shape.imag,
        "fitted_delta_psi_contribution_at_selected_time": _snapshot.real,
    })
    _unpaired = [mo.callout("This complex component has no conjugate partner at the retained rank. The snapshot shows its real part only; increase rank or check fit stability before interpreting a real oscillating pattern.", kind="warn")] if _partner < 0 and np.abs(_s.imag) > 1e-8 else []
    _e_fold = f"E-folding time: {1 / _s.real:.5g} s." if np.isfinite(_s.real) and _s.real > 0 else ""
    mo.vstack([mo.md(f"**Component {_q + 1}: γ = {_s.real:.6g} s⁻¹, ω = {_s.imag:.6g} rad/s.** "
        + _e_fold + f" Fitted initial excitation magnitude |a| = {abs(dynamic_modes.amplitudes[_q]):.5g}."),
        *_unpaired, _output,
        mo.md("**Wavelength content:** for the DCT-II grid, L = M Δz and wavelength = 2L/m. "
            "Mode 0 is constant (infinite wavelength). The table below shows the ten largest weights; downloads retain all selected modes. "
            "Weights describe a component's structure and omit phase; they are not additive energy fractions across components."),
        mo.ui.table(dynamic_dct_table.sort_values("relative_DCT_weight", ascending=False).head(10),
            selection=None, label="Strongest wavelength contributions to this dynamic mode"),
        mo.hstack([
            mo.download(dynamic_dct_table.to_csv(index=False), filename=f"dynamic_mode_{_q + 1}_dct.csv", label="Download this mode's DCT structure and wavelengths"),
            mo.download(dynamic_spatial_table.to_csv(index=False), filename=f"dynamic_mode_{_q + 1}_psi.csv", label="Download this mode's ψ structure and selected-time contribution"),
        ]),
    ])
    return dynamic_dct_table, dynamic_spatial_table


@app.cell
def _(growth, hankel_time, hankel_vectors):
    _figure, _axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    _finite = np.isfinite(growth.exponents)
    _axes[0].scatter(growth.exponents.real[_finite], growth.exponents.imag[_finite])
    for _q in np.flatnonzero(_finite):
        _axes[0].annotate(str(_q + 1), (growth.exponents[_q].real, growth.exponents[_q].imag))
    _axes[0].axvline(0, color="grey", linewidth=1)
    _axes[0].set(xlabel="ESPRIT growth rate γ [s⁻¹]", ylabel="Angular frequency ω [rad/s]", title="ESPRIT spectrum: growth and oscillation")
    _axes[0].grid(alpha=0.25)
    _axes[1].plot(hankel_time, np.linalg.norm(hankel_vectors, axis=1), label="Hankel input")
    _axes[1].plot(hankel_time, np.linalg.norm(growth.reconstruction, axis=1), "--", label="Exponential reconstruction")
    _axes[1].plot(hankel_time, np.linalg.norm(growth.reconstruction - hankel_vectors, axis=1), ":", label="Residual norm")
    _axes[1].set(xlabel="t [s]", ylabel="DCT-vector norm", title="Fit over the complete selected window")
    _axes[1].legend()
    _output = mo.as_html(_figure)
    plt.close(_figure)
    mo.vstack([_output, mo.md(
        "**Spectrum:** right of γ = 0 means growth, left means decay. Vertical position gives oscillation frequency; "
        "opposite frequencies at equal γ normally form one real oscillating pattern. "
        "**Reconstruction:** compares the selected DCT-vector norm with the norm reconstructed using the ESPRIT exponents. "
        "The residual norm uses the full coefficient-vector difference, not just the difference between these two norm curves. "
        "This checks the exponential model over the selected window; it does not establish the true equilibrium spectrum."
    )])
    return


@app.cell(hide_code=True)
def _(advanced_toggle, demo_toggle):
    mo.stop(not advanced_toggle.value)
    _demo_note = "The synthetic pair mixes growing patterns with rates +0.12 and +0.06 s⁻¹; rank 2 should recover both. " if demo_toggle.value else ""
    mo.callout(
        _demo_note + "Vary the endpoint, delay count, retained rank, and observed DCT range to check rate stability. "
        "Track nearby exponents and their DCT content; component numbers are just sorting labels and can change. "
        "High retained Hankel energy alone does not make every extracted rate reliable. "
        "These are poles of δψ, an observable projection of φ. The equilibrium's first profile is held fixed. "
        "A bias in that profile or a conserved direction can add a constant component (rate 0). "
        "Small residuals support the exponential model but do not establish that the window is linear or that every φ eigenmode is observable.",
        kind="info",
    )
    return


@app.cell
def _(equilibrium_run, cutoff_control, dct_vectors, delays, delta_psi, dynamic_modes, dynamic_time_factors, end_index, end_time, esprit_dct_weights, growth, growth_table, hankel_time, hankel_vectors, perturbed_run, linear_time, mode_max, mode_min, rank_control, sampling, saved_dt, selected_modes, subspace, z):
    selection_metadata = {
        "perturbed_file": perturbed_run["path"], "equilibrium_file": equilibrium_run["path"],
        "linear_start_time_s": float(linear_time[0]), "linear_end_time_s": end_time,
        "linear_end_saved_frame": end_index, "endpoint_included": True,
        "N_saved_frames": len(linear_time), "M_DCT_modes": len(z),
        "dct_mode_min": mode_min, "dct_mode_max": mode_max, "mode_range_inclusive": True,
        "M_selected_DCT_modes": len(selected_modes),
        "saved_dt_s": saved_dt, "delay_blocks": delays, "retained_rank": int(rank_control.value),
        "relative_rank_cutoff": cutoff_control.value, "dct_type": 2, "dct_norm": "ortho",
        "method": "block_Hankel_ESPRIT_shift_invariance",
        "reference": "fixed_first_equilibrium_psi_profile",
        "dynamic_mode_observable": "psi_projected_onto_selected_DCT_band",
        "dynamic_mode_normalization": "unit_L2_norm_largest_spatial_entry_real_positive",
        "mode_factorization": "spatial_amplitudes[:,q] = dynamic_dct_shapes[:,q] * dynamic_initial_amplitudes[q]",
        "spatial_reconstruction": "sum_q dynamic_psi_shapes[:,q] * dynamic_initial_amplitudes[q] * dynamic_time_factors[j,q]",
        "conjugate_partner_indexing": "zero_based_minus_one_means_no_partner",
        "component_order": "decreasing_real_part_of_continuous_exponent",
        "time_resampled": sampling.resampled,
        "time_interpolation": "linear" if sampling.resampled else "none",
        "max_relative_saved_interval_deviation": sampling.max_relative_step_deviation,
        "retained_energy": growth.retained_energy, "shift_residual": growth.shift_residual,
        "shift_condition": growth.shift_condition, "reconstruction_error": growth.reconstruction_error,
    }
    _metadata_json = json.dumps(selection_metadata, indent=2)

    def make_analysis_archive():
        buffer = io.BytesIO()
        np.savez_compressed(buffer, metadata_json=_metadata_json, time=linear_time, z=z,
            delta_psi=delta_psi, dct_vectors=dct_vectors, selected_modes=selected_modes, hankel_time=hankel_time,
            hankel_vectors=hankel_vectors, singular_values=subspace.singular_values,
            discrete_eigenvalues=growth.discrete_eigenvalues, exponents=growth.exponents,
            spatial_amplitudes=growth.spatial_amplitudes, esprit_dct_weights=esprit_dct_weights,
            equilibrium_psi_profile=equilibrium_run["psi"][0],
            dynamic_dct_shapes=dynamic_modes.dct_shapes, dynamic_psi_shapes=dynamic_modes.psi_shapes,
            dynamic_initial_amplitudes=dynamic_modes.amplitudes,
            dynamic_conjugate_partners=dynamic_modes.conjugate_partners,
            dynamic_time_factors=dynamic_time_factors,
            reconstruction=growth.reconstruction)
        return buffer.getvalue()

    mo.hstack([
        mo.download(_metadata_json, filename="linear_regime_selection.json", label="Save selected regime and settings"),
        mo.download(growth_table.to_csv(index=False), filename="hankel_component_growth_rates.csv", label="Download ESPRIT component rates"),
        mo.download(make_analysis_archive, filename="hankel_growth_analysis.npz", label="Download ESPRIT arrays and metadata"),
    ])
    return (selection_metadata,)


if __name__ == "__main__":
    app.run()
