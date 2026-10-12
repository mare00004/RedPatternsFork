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

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from red_patterns import get_rbc_cmap
    from red_patterns.dct_dispersion import (
        analyze_modes,
        demo_pair,
        dispersion_features,
        read_run,
        validate_pair,
    )

    REPO_ROOT = Path(__file__).resolve().parent.parent

    def edges(samples, singleton_width=1.0):
        """Cell edges for honest heatmaps even with irregular saved times."""
        if len(samples) == 1:
            return np.array([samples[0] - singleton_width / 2,
                             samples[0] + singleton_width / 2])
        midpoints = (samples[:-1] + samples[1:]) / 2
        return np.r_[samples[0] - (samples[1] - samples[0]) / 2,
                     midpoints, samples[-1] + (samples[-1] - samples[-2]) / 2]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # DCT-II perturbation growth and dispersion

    Select an **unperturbed equilibrium** and a **perturbed run** with the same
    dynamics and spatial grids. Inspect their ψ heatmaps, mark the last saved
    frame in the linear regime, then choose the inclusive DCT mode range you
    excited. The complete stored z grid, including any wings, defines the DCT.

    We subtract the first equilibrium profile, held fixed,
    $\delta\psi(z,t)=\psi_{\mathrm{perturbed}}(z,t)-\psi_{\mathrm{eq}}(z,t_0)$,
    and fit each orthonormal DCT-II coefficient in **log-amplitude space**:
    $\ln|c_m(t)|=b_m+\gamma_m(t-t_0)$, with no additive offset in $c_m$.
    Unweighted least squares gives every retained saved frame equal weight
    in log space, using its original time through the selected endpoint.
    Positive and negative coefficients are supported, but a resolved sign
    change rejects the mode. Samples at or below the coefficient precision
    floor are excluded, never replaced by an artificial epsilon; at least
    five samples must remain.

    This model assumes a directly excited exponential mode. An accumulated
    observable such as $A(\exp[\gamma t]-1)$ is not log-linear; use the
    offset-capable fits in `analyze_linear_growth.py` for that case.

    For your independently evolving cosine eigenmodes, $\gamma_m$ is the
    perturbation growth rate: positive grows, negative decays. Wavelength means
    the wavelength of the cosine basis, rather than a peak spacing measured
    from ψ. The [DCT-II basis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html)
    is $\cos[\pi m(j+1/2)/N]$, hence
    $L=N\Delta z$, $k_m=\pi m/L$, and **$\lambda_m=2L/m$**.
    Mode 0 is the constant mean and has no finite wavelength.
    """)
    return


@app.cell
def _():
    equilibrium_picker = mo.ui.file_browser(
        initial_path=REPO_ROOT / "data", filetypes=[".h5"], multiple=False,
        label="Unperturbed equilibrium run.h5",
    )
    perturbed_picker = mo.ui.file_browser(
        initial_path=REPO_ROOT / "data", filetypes=[".h5"], multiple=False,
        label="Perturbed run.h5 (same dynamics)",
    )
    demo_toggle = mo.ui.checkbox(
        value=mo.app_meta().mode == "script", label="Use synthetic cosine-eigenmode validation pair",
    )
    mo.vstack([mo.hstack([equilibrium_picker, perturbed_picker], widths="equal"), demo_toggle])
    return demo_toggle, equilibrium_picker, perturbed_picker


@app.cell
def _(demo_toggle, equilibrium_picker, perturbed_picker):
    mo.stop(
        not demo_toggle.value and (not equilibrium_picker.value or not perturbed_picker.value),
        mo.md("Select both runs above, or enable the synthetic validation pair."),
    )
    _load_error = None
    if demo_toggle.value:
        perturbed_run, equilibrium_run = demo_pair()
    else:
        _p, _e = perturbed_picker.path(0), equilibrium_picker.path(0)
        mo.stop(_p.resolve() == _e.resolve(), mo.callout("Select two different runs.", kind="warn"))
        try:
            perturbed_run, equilibrium_run = read_run(_p), read_run(_e)
        except (OSError, KeyError, ValueError) as _exc:
            _load_error = str(_exc)
    mo.stop(_load_error is not None, mo.callout(f"Could not read the selected files: {_load_error}", kind="warn"))
    _validation_error = None
    try:
        equilibrium_drift, pair_notes = validate_pair(perturbed_run, equilibrium_run)
    except ValueError as _exc:
        _validation_error = str(_exc)
    mo.stop(_validation_error is not None, mo.callout(_validation_error, kind="warn"))
    mo.vstack([
        mo.md(f"**Equilibrium:** `{equilibrium_run['path']}`  \n"
              f"**Perturbed:** `{perturbed_run['path']}`  \n"
              f"Equilibrium ψ relative drift: `{equilibrium_drift:.3e}`. "
              "Subtraction uses its first saved profile; saving cadences and durations may differ."),
        *[mo.callout(_note, kind="warn") for _note in pair_notes],
    ])
    return equilibrium_run, perturbed_run


@app.cell
def _(perturbed_run):
    end_time_slider = mo.ui.slider(
        steps=perturbed_run["time"].tolist(),
        value=float(perturbed_run["time"][max(4, (len(perturbed_run["time"]) - 1) // 3)]),
        label="End of the linear regime, t [s] (endpoint included)",
        show_value=True, full_width=True, debounce=True,
    )
    end_time_slider
    return (end_time_slider,)


@app.cell
def _(end_time_slider, perturbed_run):
    end_index = int(np.searchsorted(perturbed_run["time"], end_time_slider.value))
    end_time = float(perturbed_run["time"][end_index])
    mo.md(f"Selected **{end_index + 1} saved frames**, "
          f"t = {perturbed_run['time'][0]:.7g} through **{end_time:.7g} s**.")
    return end_index, end_time


@app.cell
def _(end_time, equilibrium_run, perturbed_run):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True, constrained_layout=True)
    _vmin = 100 * min(float(np.min(_r["psi"])) for _r in (equilibrium_run, perturbed_run))
    _vmax = 100 * max(float(np.max(_r["psi"])) for _r in (equilibrium_run, perturbed_run))
    if _vmin == _vmax:
        _vmax = _vmin + max(1e-6, abs(_vmin) * 1e-3)
    for _ax, _run, _title in zip(_axes, (equilibrium_run, perturbed_run),
                               ("Unperturbed equilibrium — saved ψ", "Perturbed run — saved ψ")):
        # Downsampling is for display only; fits use every frame and z sample.
        _ti = np.unique(np.linspace(0, len(_run["time"]) - 1, min(1500, len(_run["time"])), dtype=int))
        _zi = np.unique(np.linspace(0, len(_run["z"]) - 1, min(1000, len(_run["z"])), dtype=int))
        _image = _ax.pcolormesh(
            edges(_run["time"][_ti], np.mean(np.diff(perturbed_run["time"]))),
            100 * edges(_run["z"][_zi]), 100 * _run["psi"][np.ix_(_ti, _zi)].T,
            cmap=get_rbc_cmap(), vmin=_vmin, vmax=_vmax, shading="flat", rasterized=True,
        )
        if _run["time"][0] <= end_time <= _run["time"][-1]:
            _ax.axvline(end_time, color="cyan", linestyle="--", linewidth=2)
        if _run is perturbed_run and end_time < _run["time"][-1]:
            _ax.axvspan(end_time, _run["time"][-1], color="black", alpha=0.15)
        _ax.set(xlabel="t [s]", ylabel="z [cm]", title=_title)
    _fig.colorbar(_image, ax=list(_axes), label="ψ [%], shared color scale")
    _output = mo.as_html(_fig)
    plt.close(_fig)
    _output
    return


@app.cell
def _(demo_toggle, perturbed_run):
    mode_range = mo.ui.range_slider(
        start=0, stop=len(perturbed_run["z"]) - 1, step=1,
        value=[1, min(12, len(perturbed_run["z"]) - 1)] if demo_toggle.value
        else [1, len(perturbed_run["z"]) - 1],
        label="Inclusive DCT-II mode range [m_min, m_max]",
        show_value=True, full_width=True, debounce=True,
    )
    mode_range
    return (mode_range,)


@app.cell
def _(end_index, equilibrium_run, mode_range, perturbed_run):
    mo.stop(end_index < 4, mo.callout("Select at least five saved frames for exponential fits.", kind="warn"))
    mode_min, mode_max = map(int, mode_range.value)
    fit_time, coefficients, fits, rate_table, dct_length, coefficient_floor = analyze_modes(
        perturbed_run, equilibrium_run, end_index, mode_min, mode_max,
    )
    fastest, crossings = dispersion_features(rate_table)
    return coefficient_floor, coefficients, crossings, dct_length, fastest, fit_time, fits, mode_max, mode_min, rate_table


@app.cell
def _(coefficients, end_time, equilibrium_run, fit_time, mode_max, mode_min, perturbed_run):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    _ti = np.unique(np.linspace(0, len(fit_time) - 1, min(1500, len(fit_time)), dtype=int))
    _zi = np.unique(np.linspace(0, len(perturbed_run["z"]) - 1, min(1000, len(perturbed_run["z"])), dtype=int))
    _delta = perturbed_run["psi"][np.ix_(_ti, _zi)] - equilibrium_run["psi"][0, _zi]
    _limit = max(float(np.max(np.abs(_delta))), np.finfo(float).tiny)
    _im = _axes[0].pcolormesh(
        edges(fit_time[_ti]), 100 * edges(perturbed_run["z"][_zi]), _delta.T,
        cmap="RdBu_r", vmin=-_limit, vmax=_limit, shading="flat", rasterized=True,
    )
    _fig.colorbar(_im, ax=_axes[0], label="δψ [fraction]")
    _axes[0].set(xlabel="t [s]", ylabel="z [cm]", title="Perturbation in the linear window")
    _count = mode_max - mode_min + 1
    for _m in range(mode_min, mode_max + 1):
        _axes[1].plot(fit_time, np.abs(coefficients[:, _m]), alpha=0.7,
                      label=f"m={_m}" if _count <= 16 else None)
    _axes[1].set_yscale("symlog", linthresh=max(_limit * 1e-6, 1e-18))
    _axes[1].set(xlabel="t [s]", ylabel="|Signed DCT coefficient|", title="Selected cosine-mode amplitudes")
    if _count <= 16:
        _axes[1].legend(ncol=2, fontsize=8)
    _axes[1].grid(alpha=0.25)
    _output = mo.as_html(_fig)
    plt.close(_fig)
    _output
    return


@app.cell
def _(crossings, dct_length, fastest, mode_max, mode_min, rate_table):
    _resolved = int(np.count_nonzero(rate_table["status"] == "ok"))
    _summary = [mo.md(f"DCT-II length **L = {dct_length:.7g} m**; "
                     f"**{_resolved}/{len(rate_table)} modes** have resolved fits. "
                     "Characteristic wavelengths use resolved nonzero modes only.")]
    if fastest is None:
        _summary.append(mo.callout("No resolved growing nonzero mode in this selection; fastest-growing wavelength is unavailable.", kind="warn"))
    else:
        _m = int(fastest["mode"])
        _summary.append(mo.md(
            f"**Fastest measured growing wavelength: {100 * fastest['wavelength_m']:.7g} cm**  \n"
            f"DCT mode **m = {_m}**, γ = **{fastest['growth_rate_s^-1']:.7g} s⁻¹**. "
            "This maximum is among resolved selected modes; unexcited or excluded modes are unmeasured.",
        ))
        if _m in (mode_min, mode_max):
            _summary.append(mo.callout("The measured maximum lies at the selected mode boundary. Expand the range to check for a larger growth rate.", kind="warn"))
    _cutoffs = crossings[crossings["kind"] == "short-wave cutoff"]
    if _cutoffs.empty:
        _summary.append(mo.md("**Short-wave cutoff: not resolved in this range.** "
                              "A growing mode and its adjacent higher-k decaying mode must both have resolved fits; no extrapolation is used."))
    else:
        for _row in _cutoffs.to_dict("records"):
            _summary.append(mo.md(
                f"**Cutoff wavelength ≈ {100 * _row['wavelength_m']:.7g} cm**, "
                f"bracketed by modes {_row['mode_left']} and {_row['mode_right']} "
                f"({100 * _row['wavelength_bracket_min_m']:.7g}–{100 * _row['wavelength_bracket_max_m']:.7g} cm).",
            ))
    _summary.append(mo.md("Cutoffs interpolate γ linearly in k between adjacent modes. Multiple unstable bands retain all crossings. "
                         "A missing crossing means the cutoff is unmeasured, rather than absent from the system."))
    mo.vstack(_summary)
    return


@app.cell
def _(dct_length, demo_toggle, equilibrium_run, perturbed_run):
    taylor_prediction = None
    if demo_toggle.value:
        prediction_note = "The synthetic pair has prescribed rates; the Taylor continuum overlay is available for saved Taylor runs."
    elif perturbed_run["model"].get("modelType") not in ("TAYL", b"TAYL"):
        prediction_note = "The continuum overlay requires a Taylor run with saved NU and MU."
    else:
        _beta = perturbed_run["model"].get("beta")
        _nu = perturbed_run["variant"].get("NU")
        _mu = perturbed_run["variant"].get("MU")
        _linear_full = perturbed_run["model"].get("gradientType") in ("LINEAR_FULL", b"LINEAR_FULL")
        _alpha = perturbed_run["model"].get("alpha") if _linear_full else 0.0
        if any(_value is None or not np.isfinite(float(_value)) for _value in (_beta, _nu, _mu, _alpha)):
            prediction_note = "The continuum overlay requires finite beta, NU, and MU metadata, plus alpha for LINEAR_FULL."
        else:
            # CuKernelGradLinearFull uses a fixed density drop of 30 over N*DZ.
            _g = 30.0 / dct_length if _linear_full else 0.0
            taylor_prediction = {
                "beta": float(_beta), "nu": float(_nu), "mu": float(_mu),
                "avg_psi": float(np.mean(equilibrium_run["psi"][0])),
                "alpha": float(_alpha), "g": _g,
                "buoyancy_decay_rate_s^-1": float(_alpha) * _g,
                "g_definition": "30/(N*dz), matching CuKernelGradLinearFull" if _linear_full else "no buoyancy correction",
            }
            _formula = (r"$\sigma(k)=-\alpha g-\beta\langle\psi\rangle\nu k^2"
                        r"+\beta\langle\psi\rangle\mu k^4$. " if _linear_full else
                        r"$\sigma(k)=-\beta\langle\psi\rangle\nu k^2"
                        r"+\beta\langle\psi\rangle\mu k^4$. ")
            _assumption = (
                f"LINEAR_FULL: α = {float(_alpha):.6g}, g = {_g:.6g}, "
                f"αg = {float(_alpha) * _g:.6g} s⁻¹. "
                "This is the continuum prediction for displacements of a constant-weight equilibrium ridge. "
                if _linear_full else "This is the uniform-state continuum prediction without buoyancy. "
            )
            prediction_note = (
                "Dashed curve: " + _formula +
                f"β = {float(_beta):.6g}, ν = {float(_nu):.6g}, μ = {float(_mu):.6g}; "
                f"AVG_PSI = {taylor_prediction['avg_psi']:.6g} "
                "(fraction, spatial mean of the first equilibrium profile over the full stored grid). "
                + _assumption + "Finite-grid rates may differ."
            )
    return prediction_note, taylor_prediction


@app.cell
def _(crossings, fastest, prediction_note, rate_table, taylor_prediction):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    _finite = rate_table["mode"] > 0
    _ok = _finite & (rate_table["status"] == "ok")
    _poor = _finite & np.isfinite(rate_table["growth_rate_s^-1"]) & ~_ok
    if taylor_prediction is not None:
        _k = np.linspace(0, float(rate_table["wavenumber_rad_m"].max()), 512)
        _prefactor = taylor_prediction["beta"] * taylor_prediction["avg_psi"]
        _sigma = (-taylor_prediction["buoyancy_decay_rate_s^-1"]
                  - _prefactor * taylor_prediction["nu"] * _k**2
                  + _prefactor * taylor_prediction["mu"] * _k**4)
        _axes[0].plot(_k, _sigma, "--", color="tab:purple", linewidth=2,
                      label="Taylor continuum prediction")
    for _ax, _key, _scale, _label in zip(
        _axes, ("wavenumber_rad_m", "wavelength_m"), (1, 100),
        ("DCT-II wavenumber k [rad/m]", "DCT-II wavelength λ [cm]"),
    ):
        # NaNs break lines at unresolved modes, rather than implying measurements.
        _line = rate_table.loc[_finite, "growth_rate_s^-1"].where(_ok[_finite])
        _ax.plot(_scale * rate_table.loc[_finite, _key], _line, color="tab:blue", alpha=0.35)
        _ax.errorbar(_scale * rate_table.loc[_ok, _key], rate_table.loc[_ok, "growth_rate_s^-1"],
                     yerr=rate_table.loc[_ok, "rate_std_s^-1"], fmt="o", capsize=3, label="Resolved fit ± fit standard error")
        _ax.scatter(_scale * rate_table.loc[_poor, _key], rate_table.loc[_poor, "growth_rate_s^-1"],
                    marker="x", color="tab:orange", label="Poor fit (excluded from features)")
        for _row in crossings.to_dict("records"):
            _ax.axvline(_scale * _row[_key], linestyle=":", color="tab:red", alpha=0.8)
            _ax.scatter(_scale * _row[_key], 0, marker="D", color="tab:red")
        if fastest is not None:
            _ax.scatter(_scale * fastest[_key], fastest["growth_rate_s^-1"],
                        marker="*", s=180, color="tab:green", label="Fastest measured growing mode", zorder=5)
        _ax.axhline(0, color="grey", linewidth=1)
        _ax.set(xlabel=_label, ylabel="Perturbation growth rate γ [s⁻¹]", title="Cosine-mode dispersion relation")
        _ax.grid(alpha=0.25)
        _ax.legend(fontsize=8)
    _output = mo.as_html(_fig)
    plt.close(_fig)
    mo.vstack([_output, mo.md(prediction_note)])
    return


@app.cell
def _(crossings, rate_table):
    mo.vstack([
        mo.md("**Log-space fit diagnostics.** `ok` requires |γ| > 2 fit standard errors and "
              "log relative fit error ≤ 0.1 (residual norm divided by centered log-amplitude norm). "
              "R² and errors refer to log amplitude. Counts report samples above/below the precision floor. "
              "Sign-changing, flat, or weak coefficients cannot establish a rate. "
              "Standard errors assume independent, equal-variance log residuals; the precision floor is not a noise estimate. "
              "Inspect traces and vary the endpoint to check stability."),
        mo.ui.table(rate_table, selection=None, show_download=True, label="DCT-II modal rates and wavelengths"),
        mo.ui.table(crossings, selection=None, show_download=True, label="Resolved zero crossings and wavelength brackets"),
    ])
    return


@app.cell
def _(coefficients, mode_max, mode_min):
    inspect_mode = mo.ui.slider(
        start=mode_min, stop=mode_max, step=1,
        value=mode_min + int(np.argmax(np.ptp(coefficients[:, mode_min:mode_max + 1], axis=0))),
        label="DCT-II mode to inspect", show_value=True, debounce=True,
    )
    inspect_mode
    return (inspect_mode,)


@app.cell
def _(coefficient_floor, coefficients, fit_time, fits, inspect_mode, mode_min):
    _m = int(inspect_mode.value)
    _fit = fits[_m - mode_min]
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    _amplitude = np.abs(coefficients[:, _m])
    _log_values = np.full_like(_amplitude, np.nan)
    np.log(_amplitude, out=_log_values, where=_amplitude > 0)
    _used = _fit.used_samples
    _axes[0].plot(fit_time, _log_values, color="grey", alpha=0.4)
    _axes[0].scatter(fit_time[_used], _log_values[_used], s=12, label="Above precision floor")
    _axes[0].scatter(fit_time[~_used], _log_values[~_used], marker="x", color="grey", label="Excluded (zeros omitted)")
    _axes[0].plot(fit_time[_used], _fit.log_fitted[_used], "--", label="b + γ(t−t₀)")
    if coefficient_floor > 0:
        _axes[0].axhline(np.log(coefficient_floor), color="grey", linestyle=":", label="Precision floor")
    _axes[0].set(xlabel="t [s]", ylabel=f"ln|c{_m}(t)|", title=f"m={_m}: γ={_fit.rate:.7g} s⁻¹; {_fit.status}")
    _axes[0].legend()
    _axes[1].plot(fit_time[_used], _log_values[_used] - _fit.log_fitted[_used])
    _axes[1].axhline(0, color="grey", linewidth=1)
    _axes[1].set(xlabel="t [s]", ylabel="ln|c| − fitted log amplitude", title=f"Log residual; relative error={_fit.relative_rmse:.3g}")
    for _ax in _axes:
        _ax.grid(alpha=0.25)
    _output = mo.as_html(_fig)
    plt.close(_fig)
    _output
    return


@app.cell
def _(coefficient_floor, coefficients, crossings, dct_length, end_index, equilibrium_run, fastest, fit_time, fits, mode_max, mode_min, perturbed_run, rate_table, taylor_prediction):
    _metadata = {
        "perturbed_file": perturbed_run["path"], "equilibrium_file": equilibrium_run["path"],
        "reference": "fixed_first_saved_equilibrium_psi", "dct_type": 2, "dct_norm": "ortho",
        "dct_length_m": dct_length, "wavelength_definition": "2*N*dz/m; infinity for mode 0",
        "linear_start_time_s": float(fit_time[0]), "linear_end_time_s": float(fit_time[-1]),
        "linear_end_frame": end_index, "endpoint_included": True,
        "mode_min": mode_min, "mode_max": mode_max, "mode_range_inclusive": True,
        "fit_model": "ln(abs(c)) = b + gamma*(t-t0)", "uses_original_saved_times": True,
        "fit_method": "unweighted_linear_least_squares_in_log_amplitude",
        "coefficient_offset": 0.0, "fit_diagnostics_space": "natural_log_amplitude",
        "sample_policy": "abs(c) > precision floor; at least five samples; reject resolved sign changes",
        "coefficient_precision_floor": coefficient_floor,
        "taylor_continuum_prediction": taylor_prediction,
        "taylor_continuum_formula": "-alpha*g - beta*avg_psi*nu*k**2 + beta*avg_psi*mu*k**4",
        "avg_psi_definition": "spatial mean of first equilibrium psi over full stored grid, as fraction",
        "cutoff_method": "zero crossing linearly interpolated in k between adjacent resolved modes",
        "fastest_measured_mode": None if fastest is None else int(fastest["mode"]),
        "fastest_measured_wavelength_m": None if fastest is None else float(fastest["wavelength_m"]),
        "fastest_measured_rate_s^-1": None if fastest is None else float(fastest["growth_rate_s^-1"]),
        "zero_crossings": crossings.to_dict("records"),
    }
    _buffer = io.BytesIO()
    np.savez_compressed(
        _buffer, time=fit_time, z=perturbed_run["z"], modes=rate_table["mode"].to_numpy(),
        coefficients=coefficients[:, mode_min:mode_max + 1],
        fitted_coefficients=np.column_stack([_fit.fitted for _fit in fits]),
        fitted_log_amplitudes=np.column_stack([_fit.log_fitted for _fit in fits]),
        above_precision_floor=np.column_stack([_fit.used_samples for _fit in fits]),
        metadata_json=json.dumps(_metadata, indent=2),
    )
    mo.hstack([
        mo.download(rate_table.to_csv(index=False).encode(), filename="dct_dispersion_rates.csv", label="Download mode rates CSV"),
        mo.download(crossings.to_csv(index=False).encode(), filename="dct_dispersion_crossings.csv", label="Download cutoffs CSV"),
        mo.download(_buffer.getvalue(), filename="dct_dispersion.npz", label="Download coefficients, fits and metadata"),
    ])
    return


if __name__ == "__main__":
    app.run()
