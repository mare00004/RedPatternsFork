# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo>=0.23.6", "numpy", "scipy", "h5py", "matplotlib", "pandas"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="wide")


with app.setup:
    import io
    from pathlib import Path

    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.fft import idct

    from red_patterns.dct_dispersion import fit_log_mode, read_run
    from red_patterns.paired_mode_growth import early_prefix, paired_response


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Local ±ε single-mode or seeded multimode growth experiment

    In `workbench.py`, choose **Perturbed Linear Gradient Diagonal** phi,
    a seed, an inclusive mode range, and **+ε in metres** (including `1e-12`).
    Repeat with **−ε**, keeping **the same seed, mode range and all other settings**.
    For a deterministic single mode, use **Single-mode Linear Gradient Diagonal**
    and enter that mode as both range endpoints below.
    Use **linear-full** gradient and set
    **wing_z = wing_r = 0** for a full-domain DCT-II cosine. Run locally,
    keep the output path, then run again with **−ε** and identical settings.
    Workbench creates a separate temporary directory for each run; copy these
    directories somewhere permanent if you want to keep them.
    Select **Phi and psi** storage if you also want δφ snapshots below.
    Store enough early frames (at least eight) to resolve the growth.

    Pick the two local `run.h5` files below; no catalog, sweep, server, or
    unperturbed run is required. Enter the positive magnitude ε, not 2ε.
    We use **positive minus negative** for both observables:
    $\delta\varphi=(\varphi_+-\varphi_-)/(2\epsilon)$ and
    $\delta\psi=(\psi_+-\psi_-)/(2\epsilon)$.
    The DCT-II acts along z at every saved time, with `norm="ortho"`.
    Signed coefficients are retained; the plotted amplitude is $|A_m(t)|$.
    The seeded displacement is
    $T_\pm(z)=z\pm\epsilon\sum_{n=n_{min}}^{n_{max}}b_n\sin(n\pi z/L)$,
    where `numpy.random.default_rng(seed).normal(size=mode_count)` gives $b_n$.
    ε scales the whole random displacement; it is not its maximum magnitude.
    The conservative remap acts at fixed ρ and requires $1+\xi'(z)>0$.
    For a constant marginal, the small-ε response is approximately
    $\delta\psi=-\psi_0\sum_n b_n(n\pi/L)\cos(n\pi z/L)$.
    Each mode has its own initial amplitude. Expect the whole seeded band,
    rather than one mode, to be excited. Finite remapping can generate leakage.
    A small ε does not by itself resolve the one-bin ridge's grid sensitivity;
    compare epsilon and resolution before interpreting rates as linear growth.
    """)
    return


@app.cell
def _():
    plus_picker = mo.ui.file_browser(initial_path=Path.cwd(), filetypes=[".h5"], multiple=False, label="+epsilon run.h5")
    minus_picker = mo.ui.file_browser(initial_path=Path.cwd(), filetypes=[".h5"], multiple=False, label="−epsilon run.h5")
    epsilon_ui = mo.ui.number(start=1e-12, value=1e-6, step=1e-12, label="ε [m] (positive displacement magnitude)")
    band_min_ui = mo.ui.number(start=1, step=1, value=1, label="First excited mode (inclusive)")
    band_max_ui = mo.ui.number(start=1, step=1, value=32, label="Last excited mode (inclusive)")
    seed_ui = mo.ui.number(start=0, step=1, value=0, label="Seed used in both runs (recorded in download)")
    demo_ui = mo.ui.checkbox(value=mo.app_meta().mode == "script", label="Synthetic seeded validation pair (known modal rates)")
    mo.vstack([mo.hstack([plus_picker, minus_picker]), mo.hstack([epsilon_ui, seed_ui]),
               mo.hstack([band_min_ui, band_max_ui]), demo_ui])
    return band_max_ui, band_min_ui, demo_ui, epsilon_ui, minus_picker, plus_picker, seed_ui


@app.cell
def _(band_max_ui, band_min_ui, seed_ui):
    mo.stop(any(v is None or not np.isfinite(v) or int(v) != v for v in
                (band_min_ui.value, band_max_ui.value, seed_ui.value)),
            mo.callout("Enter integer mode endpoints and seed.", kind="warn"))
    band_min, band_max, seed = int(band_min_ui.value), int(band_max_ui.value), int(seed_ui.value)
    mo.stop(not 1 <= band_min <= band_max or seed < 0,
            mo.callout("Use 1 ≤ first mode ≤ last mode and a nonnegative seed.", kind="warn"))
    excited_modes = np.arange(band_min, band_max + 1)
    return band_max, band_min, excited_modes, seed


@app.cell
def _(demo_ui, epsilon_ui, excited_modes, minus_picker, plus_picker, seed):
    mo.stop(not demo_ui.value and (not plus_picker.value or not minus_picker.value), mo.md("Select both run files to continue."))
    if demo_ui.value:
        _t = np.linspace(0, 20, 101)
        _z = (np.arange(512) + 0.5) / 512
        mo.stop(excited_modes[-1] >= len(_z), mo.callout("Synthetic pair supports modes through 511.", kind="warn"))
        _coeff = np.zeros((len(_t), len(_z)))
        _rates = 0.12 - 0.001 * (excited_modes - 7)**2
        _weights = np.random.default_rng(seed).normal(size=len(excited_modes))
        _coeff[:, excited_modes] = _weights[None, :] * np.exp(_t[:, None] * _rates[None, :])
        _response = idct(_coeff, type=2, norm="ortho", axis=1)
        _common = dict(time=_t, z=_z, rho=np.arange(512),
                       model=dict(modelType="TAYL", gradientType="LINEAR_FULL", alpha=0., beta=1.),
                       variant=dict(NU=1., MU=1.), run=dict(DT=0.001), kernel=None)
        positive = dict(_common, psi=0.02 + epsilon_ui.value * _response, path="Synthetic +epsilon")
        negative = dict(_common, psi=0.02 - epsilon_ui.value * _response, path="Synthetic −epsilon")
    else:
        mo.stop(plus_picker.path(0).resolve() == minus_picker.path(0).resolve(), mo.callout("Select two different run files.", kind="warn"))
        positive = read_run(plus_picker.path(0))
        negative = read_run(minus_picker.path(0))
    return negative, positive


@app.cell
def _(band_max, epsilon_ui, excited_modes, negative, positive, seed):
    delta_psi, coefficients, numerical_floor = paired_response(positive, negative, epsilon_ui.value)
    time = positive["time"]
    mo.stop(band_max >= coefficients.shape[1], mo.callout("Last excited mode must be smaller than the saved z sample count.", kind="warn"))
    mo.md(f"**+ε:** `{positive['path']}`  \n**−ε:** `{negative['path']}`  \n"
          f"{len(time)} matching frames, {coefficients.shape[1]} z samples. "
          f"Estimated roundoff floor: **{numerical_floor:.3g}**. "
          "Grids, saved times, solver DT and saved dynamics are checked. "
          f"Declared seeded band: **{excited_modes[0]}–{excited_modes[-1]}**, seed **{seed}**. "
          "Confirm epsilon, seed, mode range, wings and base phi: run files do not retain all generator parameters.")
    return coefficients, delta_psi, numerical_floor, time


@app.cell
def _(band_max, coefficients):
    max_mode_ui = mo.ui.number(start=band_max, stop=coefficients.shape[1] - 1, step=1,
                               value=max(band_max, min(255, coefficients.shape[1] - 1)), label="Highest DCT mode shown (inclusive, starting at 0)")
    leakage_ui = mo.ui.number(start=0., value=1e-4, step=1e-5, label="Out-of-band threshold relative to seeded-band amplitude norm")
    tolerance_ui = mo.ui.number(start=0.001, value=0.05, step=0.005, label="Maximum absolute log residual for automatic early window")
    min_samples_ui = mo.ui.number(start=5, value=8, step=1, label="Minimum early-fit samples")
    mo.vstack([mo.hstack([max_mode_ui, leakage_ui]), mo.hstack([tolerance_ui, min_samples_ui])])
    return leakage_ui, max_mode_ui, min_samples_ui, tolerance_ui


@app.cell
def _(coefficients, min_samples_ui, numerical_floor, selected_mode, time, tolerance_ui):
    early = early_prefix(coefficients[:, selected_mode], time, floor=numerical_floor,
                         min_samples=int(min_samples_ui.value), log_tolerance=tolerance_ui.value)
    mo.md(f"**Automatic early window:** {early['count']} samples. {early['reason']}")
    return (early,)


@app.cell
def _(early, time):
    manual_ui = mo.ui.checkbox(value=False, label="Override automatic endpoint")
    end_ui = mo.ui.slider(start=4, stop=len(time) - 1, step=1,
                         value=max(4, early['count'] - 1), show_value=True,
                         label="Manual last frame (zero-based, inclusive)")
    mo.hstack([manual_ui, end_ui])
    return end_ui, manual_ui


@app.cell
def _(coefficients, early, end_ui, manual_ui, numerical_floor, selected_mode, time):
    fit_count = int(end_ui.value) + 1 if manual_ui.value else early["count"]
    growth_fit = (fit_log_mode(coefficients[:fit_count, selected_mode], time[:fit_count], amplitude_floor=numerical_floor)
                  if fit_count >= 5 else None)
    _summary = "No acceptable initial log-linear prefix. Increase save cadence, inspect the initial condition, or use the manual window."
    if growth_fit is not None:
        _summary = (f"Fit t = **{time[0]:.6g} … {time[fit_count-1]:.6g} s**, {fit_count} samples. "
                    f"**γ = {growth_fit.rate:.7g} s⁻¹**, slope SE = {growth_fit.rate_std:.3g} s⁻¹, "
                    f"R² = {growth_fit.r_squared:.6g}. Status: **{growth_fit.status}**.")
    mo.md(_summary)
    return fit_count, growth_fit


@app.cell(hide_code=True)
def _(tolerance_ui):
    mo.md(rf"""
    **How the early regime is chosen.** Starting at the first saved frame,
    fit $\ln|A_n(t)|=b+\gamma(t-t_0)$ by ordinary least squares using actual
    saved times. Begin with the minimum sample count and extend one frame at a
    time. Stop at the **first** prefix whose maximum absolute log residual
    exceeds **{tolerance_ui.value:g}** (roughly {100*np.expm1(tolerance_ui.value):.2f}% multiplicative
    deviation), or that hits the roundoff floor or changes sign. Retain the
    previous prefix; if the first candidate fails, report no automatic fit.
    If all frames pass, the data have not identified an end to the early regime.

    This is an adjustable log-linearity heuristic, not proof of the physical
    linear regime. It deliberately does not search for a convenient later
    window. Initial transients or mixed eigenmodes may prevent a fit. Inspect
    residuals and compare nearby manual endpoints. The slope is γ directly
    because we fit amplitude, not squared amplitude. The standard error assumes
    independent log residuals; numerical trajectory errors can be correlated.
    Repeat with a smaller |ε| to check finite-difference convergence and with
    smaller DT to check numerical convergence. Mode coupling, wings and
    nonuniform base profiles can generate other modes; purity is checked below.
    Displacement mode 0 is identically zero and is rejected by the ridge
    generator. Extracted DCT mode 0 remains available as a conservation diagnostic.
    """)
    return


@app.cell
def _(excited_modes, max_mode_ui, time):
    inspect_ui = mo.ui.slider(start=0, stop=int(max_mode_ui.value), step=1, value=int(excited_modes[0]), show_value=True, label="Extracted DCT-II mode m (fit shown for this mode)")
    frame_ui = mo.ui.slider(start=0, stop=len(time)-1, step=1, value=0, show_value=True, label="Frame for spectrum / δφ")
    mo.hstack([inspect_ui, frame_ui])
    return frame_ui, inspect_ui


@app.cell
def _(inspect_ui):
    selected_mode = int(inspect_ui.value)
    return (selected_mode,)


@app.cell
def _(coefficients, excited_modes, fit_count, growth_fit, numerical_floor, selected_mode, time):
    _m = selected_mode
    _a = np.abs(coefficients[:, _m])
    _resolved = _a > numerical_floor
    _fig, _axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    _axes[0].plot(time, np.where(_resolved, np.log(np.maximum(_a, np.finfo(float).tiny)), np.nan), label=f"ln |A_{_m}(t)|")
    if numerical_floor > 0:
        _axes[0].axhline(np.log(numerical_floor), ls=":", color="grey", label="roundoff floor")
    if growth_fit is not None:
        for _ax in _axes:
            _ax.axvspan(time[0], time[fit_count-1], alpha=0.12, color="green", label="fit window" if _ax is _axes[0] else None)
        _axes[0].plot(time[:fit_count], growth_fit.log_fitted, "--", label=f"early line, γ={growth_fit.rate:.5g} s⁻¹")
        _used = growth_fit.used_samples
        _axes[1].plot(time[:fit_count][_used], np.log(_a[:fit_count][_used]) - growth_fit.log_fitted[_used], ".-")
    _axes[0].set(ylabel="ln |A_m|", title=f"Extracted mode {_m}; excited band {excited_modes[0]}–{excited_modes[-1]}")
    _axes[0].legend()
    _axes[1].set(xlabel="t [s]", ylabel="Selected-mode log residual")
    for _ax in _axes:
        _ax.grid(alpha=0.25)
    _fig
    return


@app.cell
def _(coefficients, excited_modes, fit_count, frame_ui, leakage_ui, max_mode_ui, numerical_floor, time):
    _limit = int(max_mode_ui.value) + 1
    _i = int(frame_ui.value)
    _amps = np.abs(coefficients[:, :_limit])
    _denom = np.linalg.norm(coefficients[:, excited_modes], axis=1)
    _off = _amps.copy()
    _off[:, excited_modes] = 0
    _largest = np.max(_off, axis=1)
    _ratio = np.divide(_largest, _denom, out=np.full_like(_denom, np.nan), where=_denom > numerical_floor)
    _threshold = np.maximum(numerical_floor, leakage_ui.value * _denom)
    _count = np.sum(_off > _threshold[:, None], axis=1)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    _axes[0].semilogy(np.arange(_limit), np.maximum(_amps[_i], np.finfo(float).tiny), ".")
    _axes[0].axvspan(excited_modes[0] - 0.5, excited_modes[-1] + 0.5, color="green", alpha=0.12, label="seeded band")
    _axes[0].axhline(max(_threshold[_i], np.finfo(float).tiny), ls=":", color="grey")
    _axes[0].set(xlabel="DCT-II mode", ylabel="|A_m|", title=f"Spectrum at t={time[_i]:.6g} s")
    _axes[1].plot(time, _ratio)
    _axes[1].axhline(leakage_ui.value, ls=":", color="grey")
    _axes[1].set(xlabel="t [s]", ylabel="max out-of-band |A_m| / seeded-band norm", title="Leakage within displayed mode range")
    _table = pd.DataFrame(dict(time_s=time, largest_out_of_band=_largest, leakage_ratio=_ratio, out_of_band_modes_above_threshold=_count))
    mo.vstack([_fig, mo.md(f"At selected frame: **{_count[_i]}** out-of-band modes exceed both the relative threshold and roundoff floor. "
                           f"Initial leakage ratio: **{_ratio[0]:.3g}**. "
                           f"Max leakage in fit window: **{np.nanmax(_ratio[:fit_count]):.3g}**." if fit_count else
                           f"At selected frame: **{_count[_i]}** out-of-band modes exceed the diagnostic threshold."),
               mo.ui.table(_table, selection=None, pagination=True)])
    return


@app.cell
def _(delta_psi, negative, positive, time):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    _lo = min(np.min(positive["psi"]), np.min(negative["psi"]))
    _hi = max(np.max(positive["psi"]), np.max(negative["psi"]))
    for _ax, _run, _label in zip(_axes[:2], (positive, negative), ("+ε: ψ", "−ε: ψ")):
        _im = _ax.pcolormesh(time, _run["z"], _run["psi"].T, shading="nearest", vmin=_lo, vmax=_hi)
        _ax.set(xlabel="t [s]", ylabel="z [m]", title=_label)
        _fig.colorbar(_im, ax=_ax)
    _scale = max(float(np.max(np.abs(delta_psi))), np.finfo(float).tiny)
    _im = _axes[2].pcolormesh(time, positive["z"], delta_psi.T, shading="nearest", cmap="RdBu_r", vmin=-_scale, vmax=_scale)
    _axes[2].set(xlabel="t [s]", ylabel="z [m]", title="δψ = (ψ₊ − ψ₋)/(2ε)")
    _fig.colorbar(_im, ax=_axes[2])
    _fig
    return


@app.cell
def _():
    mo.md("""
    ## Growth rate versus mode number

    All excited modes are fitted automatically, each with its own early window.
    Changing the early-fit tolerance or minimum sample count updates this plot.
    The manual endpoint affects only the selected-mode explorer.
    Unresolved rates remain in the table and are omitted from the plot.
    """)
    return


@app.cell
def _(coefficients, demo_ui, excited_modes, min_samples_ui, numerical_floor, time, tolerance_ui):
    _rows = []
    for _mode in excited_modes:
        _early = early_prefix(coefficients[:, _mode], time, floor=numerical_floor,
                              min_samples=int(min_samples_ui.value), log_tolerance=tolerance_ui.value)
        _fit = _early["fit"]
        _row = dict(mode=int(_mode), rate_s_inv=np.nan if _fit is None else _fit.rate,
                    slope_se_s_inv=np.nan if _fit is None else _fit.rate_std,
                    r_squared=np.nan if _fit is None else _fit.r_squared,
                    fit_samples=_early["count"],
                    end_time_s=np.nan if _fit is None else time[_early["count"] - 1],
                    status=_early["reason"] if _fit is None else _fit.status,
                    window_reason=_early["reason"])
        if demo_ui.value:
            _row["known_rate_s_inv"] = 0.12 - 0.001 * (_mode - 7)**2
        _rows.append(_row)
    band_rates = pd.DataFrame(_rows)
    _fig, _ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    _accepted = band_rates["status"] == "ok"
    _ax.plot(band_rates.loc[_accepted, "mode"], band_rates.loc[_accepted, "rate_s_inv"], "o", label="accepted log-linear fits")
    _ax.plot(band_rates.loc[~_accepted, "mode"], band_rates.loc[~_accepted, "rate_s_inv"], "x", label="flagged fits")
    if demo_ui.value:
        _ax.plot(band_rates["mode"], band_rates["known_rate_s_inv"], "--", label="known synthetic rates")
    _ax.axhline(0, color="grey", linewidth=0.8)
    _ax.set(xlabel="DCT-II mode number n", ylabel="Growth rate γₙ [s⁻¹]", title="Growth rate versus mode number")
    _ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    _ax.grid(alpha=0.25)
    _ax.legend()
    mo.vstack([_fig, mo.ui.table(band_rates, selection=None, pagination=True),
               mo.download(band_rates.to_csv(index=False).encode(), filename="paired_multimode_rates.csv", label="Download per-mode rates and early windows")])
    return (band_rates,)


@app.cell
def _(demo_ui, epsilon_ui, frame_ui, negative, positive):
    mo.stop(demo_ui.value, mo.md("Synthetic pair contains ψ only."))
    with h5py.File(positive["path"], "r") as _p, h5py.File(negative["path"], "r") as _m:
        mo.stop("fields/phi" not in _p or "fields/phi" not in _m, mo.md("δφ snapshots require **Phi and psi** storage in both runs. DCT analysis uses ψ only."))
        _i = int(frame_ui.value)
        _delta_phi = (np.asarray(_p["fields/phi"][_i], dtype=np.float64) - np.asarray(_m["fields/phi"][_i], dtype=np.float64)) / (2 * epsilon_ui.value)
    _fig, _ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    _v = max(float(np.max(np.abs(_delta_phi))), np.finfo(float).tiny)
    _im = _ax.pcolormesh(positive["z"], positive["rho"], _delta_phi, shading="nearest", cmap="RdBu_r", vmin=-_v, vmax=_v)
    _ax.set(xlabel="z [m]", ylabel="ρ", title=f"δφ at t={positive['time'][_i]:.6g} s")
    _fig.colorbar(_im, ax=_ax)
    _fig
    return


@app.cell
def _(coefficients, delta_psi, epsilon_ui, excited_modes, fit_count, growth_fit, negative, positive, seed, selected_mode, time):
    _buffer = io.BytesIO()
    np.savez_compressed(_buffer, time=time, z=positive["z"], delta_psi=delta_psi, signed_dct=coefficients,
                        epsilon=epsilon_ui.value, excited_modes=excited_modes, seed=seed,
                        selected_mode=selected_mode, fit_count=fit_count,
                        growth_rate=np.nan if growth_fit is None else growth_fit.rate,
                        plus_path=positive["path"], minus_path=negative["path"])
    mo.download(_buffer.getvalue(), filename="paired_mode_response.npz", label="Download δψ, all signed DCT modes and fit result")
    return


if __name__ == "__main__":
    app.run()
