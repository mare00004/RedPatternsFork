# /// script
# dependencies = [
#     "h5py==3.16.0",
#     "marimo>=0.19.4",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "wigglystuff==0.3.3",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="wide")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator
    from wigglystuff import PlaySlider

    NOTEBOOK_FILE = (
        Path(__file__).resolve()
        if "__file__" in globals()
        else (Path.cwd() / "analysis" / "analyze_fourier_modes.py").resolve()
    )
    ANALYSIS_DIR = NOTEBOOK_FILE.parent
    REPO_ROOT = ANALYSIS_DIR.parent
    if str(ANALYSIS_DIR) not in sys.path:
        sys.path.insert(0, str(ANALYSIS_DIR))

    DEFAULT_RUN_H5 = REPO_ROOT / "data" / "tayl_const_linear" / "run.h5"

    from red_patterns import RunData, get_rbc_cmap, plot_psi
    from red_patterns.phi import PhiResult, plot_phi


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Fourier Mode Analysis

    Analyze the spatial Fourier modes of
    $\delta\psi(z,t) = \psi(z,t) - \overline{\psi(t)}_z$
    for a simulation `run.h5`.
    """)
    return


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _():
    initial_path = (
        DEFAULT_RUN_H5.parent if DEFAULT_RUN_H5.parent.exists() else Path.cwd()
    )
    file_picker = mo.ui.file_browser(
        initial_path=initial_path,
        filetypes=[".h5"],
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="file",
        label="Choose run.h5 file to analyze",
    )
    return (file_picker,)


@app.cell(hide_code=True)
def _(file_picker):
    default_note = (
        mo.md(f"Default: `{DEFAULT_RUN_H5}`")
        if DEFAULT_RUN_H5.exists()
        else mo.md(f"Default not found yet: `{DEFAULT_RUN_H5}`")
    )
    mo.vstack([default_note, file_picker], align="stretch")
    return


@app.cell
def _(file_picker):
    selected_run_h5 = Path(file_picker.path()) if file_picker.value else None
    return (selected_run_h5,)


@app.cell
def _(is_script_mode, selected_run_h5):
    active_run_h5 = selected_run_h5
    active_run_source = "Selected file"

    if active_run_h5 is None and DEFAULT_RUN_H5.exists():
        active_run_h5 = DEFAULT_RUN_H5
        active_run_source = "Default from `make run-tayl-const-linear`"

    mo.stop(
        active_run_h5 is None,
        mo.md(
            "Pick a `run.h5` file to continue. "
            + f"The default path `{DEFAULT_RUN_H5}` is currently unavailable."
        ),
    )

    active_run_h5 = active_run_h5.resolve()
    active_run_md = mo.md(
        f"**Active file:** `{active_run_h5}`  \n"
        + f"**Source:** {active_run_source}  \n"
        + f"**Mode:** {'script' if is_script_mode else 'interactive'}"
    )
    return active_run_h5, active_run_md


@app.cell
def _(active_run_h5):
    run = RunData.from_h5(active_run_h5, load_fields=False)
    mo.stop(
        run.n_saved < 2, mo.md("The selected run needs at least two saved timesteps.")
    )
    return (run,)


@app.cell
def _(run):
    psi = np.asarray(run.load_psi(), dtype=np.float64)
    time = np.asarray(run.time, dtype=np.float64)
    z = np.asarray(run.z, dtype=np.float64)
    rho = np.asarray(run.rho, dtype=np.float64)
    return psi, time, z


@app.cell
def _(z):
    z_index_range = mo.ui.range_slider(
        start=0,
        stop=z.shape[0] - 1,
        step=1,
        value=[0, z.shape[0] - 1],
        debounce=True,
        show_value=True,
        full_width=True,
        label="FFT z-index range",
    )
    return (z_index_range,)


@app.cell
def _(z_index_range):
    z_start_index, z_stop_index = (int(v) for v in z_index_range.value)
    return z_start_index, z_stop_index


@app.cell
def _(psi, z, z_start_index, z_stop_index):
    z_slice = slice(z_start_index, z_stop_index + 1)
    z_fft = np.asarray(z[z_slice], dtype=np.float64)
    n_fft_points = int(z_fft.shape[0])

    mo.stop(
        n_fft_points < 2,
        mo.md("Select at least two z indices for the Fourier transform."),
    )

    psi_fft = np.asarray(psi[:, z_slice], dtype=np.float64)

    delta_psi = psi_fft - psi_fft.mean(axis=1, keepdims=True)
    fft_coeffs = np.fft.rfft(delta_psi, axis=1)
    fft_amplitudes = np.abs(fft_coeffs)
    fft_phases = np.angle(fft_coeffs)

    dz = float(z_fft[1] - z_fft[0])
    mode_numbers = np.arange(fft_coeffs.shape[1], dtype=int)
    spatial_freqs = np.fft.rfftfreq(z_fft.shape[0], d=dz)
    wavenumbers = 2.0 * np.pi * spatial_freqs
    wavelengths = np.full(spatial_freqs.shape, np.inf, dtype=np.float64)
    nonzero_modes = spatial_freqs > 0.0
    wavelengths[nonzero_modes] = 1.0 / spatial_freqs[nonzero_modes]
    return (
        fft_amplitudes,
        fft_coeffs,
        fft_phases,
        mode_numbers,
        n_fft_points,
        spatial_freqs,
        wavelengths,
        wavenumbers,
        z_fft,
    )


@app.cell
def _(run):
    time_slider = mo.ui.anywidget(
        PlaySlider(
            value=0,
            min_value=0,
            max_value=run.n_saved - 1,
            step=1,
            interval_ms=200,
            loop=False,
            width=480,
        )
    )
    return (time_slider,)


@app.cell
def _(fft_coeffs):
    max_mode = fft_coeffs.shape[1] - 1
    mo.stop(
        max_mode < 1,
        mo.md("The selected run does not contain any non-zero Fourier modes."),
    )
    mode_selector = mo.ui.slider(
        start=1,
        stop=max_mode,
        step=1,
        value=min(1, max_mode),
        label="Fourier mode n",
    )
    return (mode_selector,)


@app.cell
def _(time_slider):
    time_index = int(time_slider.value["value"])
    return (time_index,)


@app.cell
def _(mode_selector):
    selected_mode = int(mode_selector.value)
    return (selected_mode,)


@app.cell
def _(n_fft_points, z, z_fft, z_index_range, z_start_index, z_stop_index):
    z_range_panel = mo.vstack(
        [
            mo.md("### FFT z-Index Range"),
            z_index_range,
            mo.md(
                f"Indices `{z_start_index}` to `{z_stop_index}`  \n"
                + f"Physical range `{100.0 * z_fft[0]:.6g}` to `{100.0 * z_fft[-1]:.6g}` cm  \n"
                + f"Grid points used in FFT `{n_fft_points}` of `{z.shape[0]}`"
            ),
        ],
        align="stretch",
    )
    return (z_range_panel,)


@app.cell
def _(run, time, time_index, time_slider):
    time_panel = mo.vstack(
        [
            mo.md("### Time Step"),
            time_slider,
            mo.md(
                f"Step `{time_index}` of `{run.n_saved - 1}`  \n"
                f"Time `{time[time_index]:.6g}` s"
            ),
        ],
        align="stretch",
    )
    return (time_panel,)


@app.cell
def _(
    fft_amplitudes,
    mode_selector,
    selected_mode,
    spatial_freqs,
    wavelengths,
):
    wavelength_text = (
        r"$\infty$"
        if not np.isfinite(wavelengths[selected_mode])
        else f"{100.0 * wavelengths[selected_mode]:.6g} cm"
    )
    mode_panel = mo.vstack(
        [
            mo.md("### Mode Selection"),
            mode_selector,
            mo.md(
                f"Mode `{selected_mode}`  \n"
                + f"Spatial frequency `{spatial_freqs[selected_mode]:.6g}` m$^{{-1}}$  \n"
                + f"Wavelength `{wavelength_text}`  \n"
                + f"Stored coefficient series shape `{fft_amplitudes[:, selected_mode].shape}`"
            ),
        ],
        align="stretch",
    )
    return (mode_panel,)


@app.cell
def _(run, time, time_index):
    phi_frame = run.phi_frame(time_index)
    phi_figure = plot_phi(
        PhiResult(
            rho=run.rho,
            z=run.z,
            phi_values=phi_frame,
        )
    )
    phi_figure.axes[0].set_title(
        rf"$\varphi(\rho, z)$ at $t={time[time_index]:.3f}\,\mathrm{{s}}$"
    )
    phi_panel = mo.vstack(
        [
            mo.md("### Phi(z, rho)"),
            mo.as_html(phi_figure),
        ],
        align="stretch",
    )
    return (phi_panel,)


@app.cell
def _(active_run_h5, run, time, time_index, z, z_start_index, z_stop_index):
    psi_figure = plot_psi(
        run,
        vmin=0.0,
        vmax=100.0,
        cmap=get_rbc_cmap(),
        title=active_run_h5.parent.name,
    )
    psi_figure.axes[0].axvline(
        time[time_index],
        color="white",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
    )
    psi_figure.axes[0].axhline(
        100.0 * z[z_start_index],
        color="#fbbf24",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
    )
    psi_figure.axes[0].axhline(
        100.0 * z[z_stop_index],
        color="#fbbf24",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
    )
    psi_panel = mo.vstack(
        [
            mo.md("### Psi(z, t)"),
            mo.as_html(psi_figure),
        ],
        align="stretch",
    )
    return (psi_panel,)


@app.cell
def _(fft_amplitudes, mode_numbers, selected_mode, time_index):
    _fig, _ax = plt.subplots(constrained_layout=True)
    _ax.plot(
        mode_numbers[1:],
        fft_amplitudes[time_index, 1:],
        color="#2563eb",
        linewidth=1.5,
    )
    _ax.scatter(
        [selected_mode],
        [fft_amplitudes[time_index, selected_mode]],
        color="#dc2626",
        zorder=3,
        label=f"mode {selected_mode}",
    )
    _ax.set_xlabel("Mode number n")
    _ax.set_ylabel(r"$A_n(t) = |\delta\hat{\psi}_n(t)|$")
    _ax.set_title(rf"FFT amplitude at step {time_index}")
    _ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _ax.legend()
    fft_panel = mo.vstack(
        [
            mo.md("### FFT Amplitude"),
            mo.ui.matplotlib(_ax),
        ],
        align="stretch",
    )
    return (fft_panel,)


@app.cell
def _(fft_amplitudes, selected_mode, time, time_index):
    mode_amplitude = np.asarray(fft_amplitudes[:, selected_mode], dtype=np.float64)
    safe_amplitude = np.clip(mode_amplitude, np.finfo(np.float64).tiny, None)
    ln_amplitude = np.log(safe_amplitude)

    _fig, _ax = plt.subplots(constrained_layout=True)
    _ax.plot(time, ln_amplitude, color="#059669", linewidth=1.5)
    _ax.scatter(
        [time[time_index]],
        [ln_amplitude[time_index]],
        color="#dc2626",
        zorder=3,
        label=f"mode {selected_mode} at step {time_index}",
    )
    _ax.set_xlabel(r"$t\;[s]$")
    _ax.set_ylabel(r"$\ln A_n(t)$")
    _ax.set_title(rf"Growth of mode {selected_mode}")
    _ax.legend()

    growth_panel = mo.vstack(
        [
            mo.md("### Growth Rate"),
            mo.ui.matplotlib(_ax),
        ],
        align="stretch",
    )
    return growth_panel, mode_amplitude


@app.cell(hide_code=True)
def _(
    active_run_md,
    fft_coeffs,
    fft_panel,
    fft_phases,
    growth_panel,
    mode_amplitude,
    mode_panel,
    n_fft_points,
    phi_panel,
    psi_panel,
    selected_mode,
    spatial_freqs,
    time_index,
    time_panel,
    wavenumbers,
    z_fft,
    z_range_panel,
    z_start_index,
    z_stop_index,
):
    coefficient_summary = mo.md(
        f"Stored complex Fourier coefficients with shape `{fft_coeffs.shape}`.  \n"
        + f"FFT window uses z indices `{z_start_index}:{z_stop_index}` inclusive, i.e. `{100.0 * z_fft[0]:.6g}` to `{100.0 * z_fft[-1]:.6g}` cm over `{n_fft_points}` grid points.  \n"
        + f"Selected mode `{selected_mode}` at step `{time_index}`:  \n"
        + f"`|coeff| = {mode_amplitude[time_index]:.6g}`, `phase = {fft_phases[time_index, selected_mode]:.6g}` rad, `k = {wavenumbers[selected_mode]:.6g}` m$^{{-1}}$, `nu = {spatial_freqs[selected_mode]:.6g}` m$^{{-1}}$."
    )

    mo.vstack(
        [
            active_run_md,
            mo.hstack([phi_panel, psi_panel], align="start", justify="start", gap=1),
            mo.hstack(
                [fft_panel, growth_panel],
                align="start",
                justify="start",
                gap=1,
            ),
            mo.hstack(
                [time_panel, mode_panel, z_range_panel],
                align="start",
                justify="start",
                gap=1,
            ),
            coefficient_summary,
        ],
        align="stretch",
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
