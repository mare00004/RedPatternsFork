# /// script
# dependencies = [
#     "h5py==3.16.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "pydantic==2.13.4",
#     "scipy==1.17.1",
#     "wigglystuff==0.3.3",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="wide")

with app.setup:
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path

    import marimo as mo
    from wigglystuff import PlaySlider, ProgressBar

    NOTEBOOK_FILE = (
        Path(__file__).resolve()
        if "__file__" in globals()
        else (Path.cwd() / "analysis" / "workbench.py").resolve()
    )
    ANALYSIS_DIR = NOTEBOOK_FILE.parent
    REPO_ROOT = ANALYSIS_DIR.parent
    if str(ANALYSIS_DIR) not in sys.path:
        sys.path.insert(0, str(ANALYSIS_DIR))

    # The raw Taylor coefficients are tiny; the number widgets show them in these
    # units (displayed value = actual / scale) so they stay human-readable.
    NU_DISPLAY_SCALE = 1e-30
    MU_DISPLAY_SCALE = 1e-37
    STORE_PLOT_OPTIONS = {
        "Psi": ("psi",),
        "Phi and psi": ("phi", "psi"),
        "Phi, psi, and percoll": ("phi", "psi", "percoll"),
        "Phi, psi, face velocity, and face flux": (
            "phi",
            "psi",
            "face-velocity",
            "face-flux",
        ),
    }

    from red_patterns import RunData, find_peaks, get_rbc_cmap, plot_psi
    from red_patterns.kernel import (
        compute_kernel,
        kernel_config_from_ui,
        kernel_ui_layout,
        make_kernel_ui,
        plot_kernel,
        write_kernel_h5,
    )
    from red_patterns.phi import (
        PhiResult,
        make_phi_ui,
        phi_field_from_ui,
        phi_ui_layout,
        plot_phi,
    )
    from red_patterns.sim import (
        DEFAULT_POLL_SEC,
        DEFAULT_STALE_SEC,
        build_cli_args,
        estimate_total_steps,
        locate_binary,
        progress_summary,
        read_progress,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Workbench

    A general-purpose notebook that builds a kernel and an initial phi, runs the
    simulation locally with live progress, and inspects the result — all by
    importing the **library + UI-factory** API. No UI is recreated here: the
    kernel/phi controls are the exact same widgets used by `kernel.py` and
    `phi_init.py`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Build a kernel
    """)
    return


@app.cell
def cell_kernel_ui():
    kernel_ui = make_kernel_ui()
    return (kernel_ui,)


@app.cell
def _(kernel_ui):
    kernel_ui_layout(kernel_ui)
    return


@app.cell
def cell_kernel_cfg(kernel_ui):
    kernel_cfg = kernel_config_from_ui(kernel_ui.value)
    kernel_result = compute_kernel(kernel_cfg)
    return kernel_cfg, kernel_result


@app.cell
def _(kernel_cfg, kernel_result):
    plot_kernel(kernel_result, kernel_cfg)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Build an initial phi
    """)
    return


@app.cell
def cell_phi_ui():
    phi_ui = make_phi_ui()
    return (phi_ui,)


@app.cell
def _(phi_ui):
    phi_ui_layout(phi_ui)
    return


@app.cell
def cell_phi_cfg(phi_ui):
    phi_cfg = phi_field_from_ui(phi_ui.value)
    phi_result = phi_cfg.compute()
    return phi_cfg, phi_result


@app.cell
def _(phi_result):
    _phi_figure = plot_phi(phi_result)

    _psi_initial = 100.0 * np.asarray(phi_result.phi_values, dtype=np.float64).sum(axis=0)
    _z_cm = 100.0 * np.asarray(phi_result.z, dtype=np.float64)
    _psi_fig, _psi_ax = plt.subplots(constrained_layout=True)
    _psi_ax.plot(_z_cm, _psi_initial, color="#2563eb", linewidth=1.5)
    _psi_ax.set_xlabel(r"$z$ [cm]")
    _psi_ax.set_ylabel(r"$\psi$ [%]")
    _psi_ax.set_title(r"Initial $\psi(z) = \int \varphi(\rho, z)\, d\rho$")
    _psi_ax.grid(alpha=0.3)

    mo.hstack(
        [mo.as_html(_phi_figure), mo.ui.matplotlib(_psi_ax)],
        align="start",
        justify="start",
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Run the simulation
    """)
    return


@app.cell
def cell_sim_binary():
    _default_binary = locate_binary(REPO_ROOT)
    ui_binary = mo.ui.file_browser(
        initial_path=_default_binary.parent,
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="file",
        label="Simulation binary",
    )
    mo.vstack([mo.md(f"Default: `{_default_binary}`"), ui_binary])
    return (ui_binary,)


@app.cell
def cell_sim_controls():
    # ν / μ source selector + custom inputs, shown inside the Taylor tab. The
    # number widgets display the coefficients in NU_DISPLAY_SCALE / MU_DISPLAY_SCALE
    # units; cell_run scales them back to physical values for the CLI.
    ui_taylor_source = mo.ui.radio(
        options=["Kernel-derived", "Custom"],
        value="Kernel-derived",
        label="ν / μ source",
    )
    ui_nu = mo.ui.number(
        step=1e-6,
        value=-2.832638,
        label=f"ν [× {NU_DISPLAY_SCALE:g}]",
    )
    ui_mu = mo.ui.number(
        step=1e-6,
        value=-4.468455,
        label=f"μ [× {MU_DISPLAY_SCALE:g}]",
    )
    ui_mode = mo.ui.tabs(
        {
            "Convolution": mo.md("Uses the exported `kernel.h5` directly."),
            "Taylor": mo.vstack(
                [
                    mo.md(
                        "Uses `NU` / `MU` — either the values derived from the "
                        "current kernel, or custom values entered below."
                    ),
                    ui_taylor_source,
                    ui_nu,
                    ui_mu,
                ],
                gap=0.5,
            ),
        },
        value="Convolution",
    )
    ui_gradient = mo.ui.tabs(
        {
            "sigmoid": mo.md("Sigmoid pressure gradient."),
            "linear": mo.md("Linear pressure gradient."),
            "zero": mo.md("Zero pressure gradient (no driving term)."),
            "linear-full": mo.md("Linear pressure gradient over the whole domain (no wings)."),
        },
        value="sigmoid",
    )
    ui_t_final = mo.ui.number(start=0.1, step=0.1, value=1000.0, label="T")
    ui_dt = mo.ui.number(start=1e-5, step=1e-4, value=1e-2, label="DT")
    ui_storeTime = mo.ui.number(start=0, step=1e-3, value=1.0, label="storeTime")
    ui_store_fields = mo.ui.radio(
        options=list(STORE_PLOT_OPTIONS),
        value="Phi and psi",
        label="Stored fields and plots",
    )
    ui_run_button = mo.ui.run_button(label="Build inputs and run simulation")
    mo.vstack(
        [
            ui_mode,
            ui_gradient,
            ui_t_final,
            ui_dt,
            ui_storeTime,
            ui_store_fields,
            ui_run_button,
        ],
        gap=1,
    )
    return (
        ui_dt,
        ui_gradient,
        ui_mode,
        ui_mu,
        ui_nu,
        ui_run_button,
        ui_storeTime,
        ui_store_fields,
        ui_t_final,
        ui_taylor_source,
    )


@app.cell
def cell_run(
    kernel_cfg,
    kernel_result,
    phi_cfg,
    phi_result,
    ui_binary,
    ui_dt,
    ui_gradient,
    ui_mode,
    ui_mu,
    ui_nu,
    ui_run_button,
    ui_storeTime,
    ui_store_fields,
    ui_t_final,
    ui_taylor_source,
):
    run_store_fields = STORE_PLOT_OPTIONS[ui_store_fields.value]
    if not ui_run_button.value:
        run_h5 = None
        _result = mo.md("Configure the run and click the button to launch.")
    elif not ui_binary.value:
        run_h5 = None
        _result = mo.md("Select a simulation binary first.")
    else:
        _binary = Path(ui_binary.value[0].path)
        _work = Path(tempfile.mkdtemp(prefix="workbench_"))
        _phi_path = _work / "phi.h5"
        _kernel_path = _work / "kernel.h5"
        _run_dir = _work / "run"
        _run_dir.mkdir(parents=True, exist_ok=True)

        # Write inputs directly via the library (no `uv run ... export` hops).
        from dataclasses import replace as _replace

        phi_cfg.write_phi_h5(_phi_path, phi_result)
        write_kernel_h5(
            _kernel_path, kernel_result, _replace(kernel_cfg, output_path=_kernel_path)
        )

        # In Taylor mode, ν/μ are either derived from the current kernel or taken
        # from the custom number widgets (displayed in scaled units).
        if ui_mode.value == "Taylor" and ui_taylor_source.value == "Custom":
            _nu = float(ui_nu.value) * NU_DISPLAY_SCALE
            _mu = float(ui_mu.value) * MU_DISPLAY_SCALE
        else:
            _nu = float(kernel_result.nu)
            _mu = float(kernel_result.mu)

        _cli = build_cli_args(
            binary_path=_binary,
            mode=ui_mode.value,
            out_dir=_run_dir,
            phi_path=_phi_path,
            kernel_path=_kernel_path,
            gradient=ui_gradient.value,
            N=phi_cfg.N,
            t_final=float(ui_t_final.value),
            dt=float(ui_dt.value),
            storeTime=int(ui_storeTime.value),
            nu=_nu,
            mu=_mu,
            store_fields=run_store_fields,
        )

        _progress = mo.ui.anywidget(
            ProgressBar(
                value=0,
                max_value=estimate_total_steps(
                    float(ui_t_final.value), float(ui_dt.value)
                ),
                color="#22c55e",
                show_text=False,
                width="100%",
                height=24,
            )
        )
        _progress_path = _run_dir / "progress.json"
        _snapshot = None
        _returncode = None
        _proc = subprocess.Popen(_cli, cwd=REPO_ROOT)
        _last_seen = time.monotonic()
        while True:
            _maybe = read_progress(_progress_path)
            if _maybe is not None:
                _snapshot = _maybe
                _last_seen = time.monotonic()
                _total = max(1, int(_snapshot.get("total_steps", 1)))
                _progress.max_value = _total
                _progress.value = min(_total, max(0, int(_snapshot.get("step", 0))))
            _returncode = _proc.poll()
            _age = time.monotonic() - _last_seen
            _status = None if _snapshot is None else str(_snapshot.get("status", ""))
            mo.output.replace(
                mo.vstack(
                    [
                        _progress,
                        mo.md(
                            progress_summary(
                                snapshot=_snapshot,
                                t_final=float(ui_t_final.value),
                                is_waiting=_snapshot is None,
                                is_stale=_age > DEFAULT_STALE_SEC,
                                returncode=_returncode,
                            )
                        ),
                    ],
                    gap=1,
                )
            )
            if _returncode is not None or _status in {"finished", "failed"}:
                break
            time.sleep(DEFAULT_POLL_SEC)
        _proc.wait()

        run_h5 = _run_dir / "run.h5"
        _result = mo.vstack(
            [
                _progress,
                mo.md(f"`run_h5 = {run_h5}`"),
                mo.md(f"`returncode = {_returncode}`"),
            ],
            gap=1,
        )
    _result
    return run_h5, run_store_fields


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Run Inspection
    """)
    return


@app.cell
def cell_inspect_run(run_h5):
    mo.stop(
        run_h5 is None or not run_h5.exists(),
        mo.md("Run a simulation to inspect the resulting `run.h5`."),
    )
    inspect_run = RunData.from_h5(run_h5, load_fields=False)
    inspect_run_md = mo.md(f"**Active file:** `{run_h5}`")
    return inspect_run, inspect_run_md


@app.cell
def _(inspect_run):
    inspect_psi = np.asarray(inspect_run.load_psi(), dtype=np.float64)
    inspect_time = np.asarray(inspect_run.time, dtype=np.float64)
    inspect_z = np.asarray(inspect_run.z, dtype=np.float64)
    return inspect_psi, inspect_time, inspect_z


@app.cell
def _(inspect_run):
    fft_time_slider = mo.ui.anywidget(
        PlaySlider(
            value=0,
            min_value=0,
            max_value=inspect_run.n_saved - 1,
            step=1,
            interval_ms=200,
            loop=False,
            width=480,
        )
    )
    return (fft_time_slider,)


@app.cell
def _(inspect_z):
    fft_z_index_range = mo.ui.range_slider(
        start=0,
        stop=inspect_z.shape[0] - 1,
        step=1,
        value=[0, inspect_z.shape[0] - 1],
        debounce=True,
        show_value=True,
        full_width=True,
        label="FFT z-index range",
    )
    return (fft_z_index_range,)


@app.cell
def _(fft_time_slider):
    fft_time_index = int(fft_time_slider.value["value"])
    return (fft_time_index,)


@app.cell
def _(fft_z_index_range):
    fft_z_start_index, fft_z_stop_index = (
        int(_value) for _value in fft_z_index_range.value
    )
    return fft_z_start_index, fft_z_stop_index


@app.cell
def _(fft_z_start_index, fft_z_stop_index, inspect_psi, inspect_z):
    _z_slice = slice(fft_z_start_index, fft_z_stop_index + 1)
    fft_z = np.asarray(inspect_z[_z_slice], dtype=np.float64)
    fft_n_points = int(fft_z.shape[0])

    mo.stop(
        fft_n_points < 2,
        mo.md("Select at least two z indices for the Fourier transform."),
    )

    _psi_fft = np.asarray(inspect_psi[:, _z_slice], dtype=np.float64)
    _delta_psi = _psi_fft - _psi_fft.mean(axis=1, keepdims=True)
    fft_coeffs = np.fft.rfft(_delta_psi, axis=1)
    fft_amplitudes = np.abs(fft_coeffs)

    _dz = float(fft_z[1] - fft_z[0])
    fft_mode_numbers = np.arange(fft_coeffs.shape[1], dtype=int)
    fft_spatial_freqs = np.fft.rfftfreq(fft_z.shape[0], d=_dz)
    fft_wavelengths = np.full(fft_spatial_freqs.shape, np.inf, dtype=np.float64)
    _nonzero_modes = fft_spatial_freqs > 0.0
    fft_wavelengths[_nonzero_modes] = 1.0 / fft_spatial_freqs[_nonzero_modes]
    return (
        fft_amplitudes,
        fft_mode_numbers,
        fft_n_points,
        fft_wavelengths,
        fft_z,
    )


@app.cell
def _(
    fft_n_points,
    fft_z,
    fft_z_index_range,
    fft_z_start_index,
    fft_z_stop_index,
    inspect_z,
):
    fft_z_range_panel = mo.vstack(
        [
            mo.md("### FFT z-Index Range"),
            fft_z_index_range,
            mo.md(
                f"Indices `{fft_z_start_index}` to `{fft_z_stop_index}`  \n"
                + f"Physical range `{100.0 * fft_z[0]:.6g}` to `{100.0 * fft_z[-1]:.6g}` cm  \n"
                + f"Grid points used in FFT `{fft_n_points}` of `{inspect_z.shape[0]}`"
            ),
        ],
        align="stretch",
    )
    return (fft_z_range_panel,)


@app.cell
def _(fft_time_index, fft_time_slider, inspect_run, inspect_time):
    fft_time_panel = mo.vstack(
        [
            mo.md("### Time Step"),
            fft_time_slider,
            mo.md(
                f"Step `{fft_time_index}` of `{inspect_run.n_saved - 1}`  \n"
                + f"Time `{inspect_time[fft_time_index]:.6g}` s"
            ),
        ],
        align="stretch",
    )
    return (fft_time_panel,)


@app.cell
def _(fft_time_index, inspect_run, inspect_time, run_h5, run_store_fields):
    if "phi" not in run_store_fields:
        phi_panel = None
    else:
        _phi_frame = inspect_run.phi_frame(fft_time_index)
        _phi_figure = plot_phi(
            PhiResult(
                rho=inspect_run.rho,
                z=inspect_run.z,
                phi_values=_phi_frame,
            )
        )
        _phi_ax = _phi_figure.axes[0]
        _phi_figure.set_size_inches(16, 9, forward=True)
        _phi_ax.set_title(
            rf"$\varphi(\rho, z)$ at $t={inspect_time[fft_time_index]:.3f}\,\mathrm{{s}}$"
        )
        _z_padding = 0.05 * float(inspect_run.z[-1] - inspect_run.z[0])
        _rho_padding = 0.05 * float(inspect_run.rho[-1] - inspect_run.rho[0])
        _phi_ax.set_xlim(
            float(inspect_run.z[0] - _z_padding),
            float(inspect_run.z[-1] + _z_padding),
        )
        _phi_ax.set_ylim(
            float(inspect_run.rho[0] - _rho_padding),
            float(inspect_run.rho[-1] + _rho_padding),
        )

        if "face-velocity" in run_store_fields:
            with h5py.File(run_h5, "r") as _h5:
                _z_face = np.asarray(_h5["coords/z_face"], dtype=np.float64)
                _face_velocity = np.asarray(
                    _h5["fields/face_velocity"][fft_time_index], dtype=np.float64
                )
                _face_flux = np.asarray(
                    _h5["fields/face_flux"][fft_time_index], dtype=np.float64
                )

            _rho_count, _face_count = _face_velocity.shape
            _rho_offset = min(8, max(0, _rho_count - 1))
            _rho_indices = np.unique(
                np.linspace(
                    0,
                    _rho_count - 1 - _rho_offset,
                    min(12, max(1, _rho_count - _rho_offset)),
                    dtype=int,
                )
            )
            _flux_rho_indices = _rho_indices + _rho_offset
            _face_indices = np.unique(
                np.linspace(
                    1 if _face_count > 2 else 0,
                    _face_count - 2 if _face_count > 2 else _face_count - 1,
                    min(16, max(1, _face_count - 2)),
                    dtype=int,
                )
            )
            _velocity_z_grid, _velocity_rho_grid = np.meshgrid(
                _z_face[_face_indices], inspect_run.rho[_rho_indices]
            )
            _flux_z_grid, _flux_rho_grid = np.meshgrid(
                _z_face[_face_indices], inspect_run.rho[_flux_rho_indices]
            )
            _z_spacing = (
                float(np.median(np.diff(_z_face))) if _face_count > 1 else 1.0
            )
            _sample_z_spacing = (
                float(np.median(np.diff(_z_face[_face_indices])))
                if _face_indices.size > 1
                else _z_spacing
            )
            _arrow_length = 0.7 * _sample_z_spacing

            def _normalized_arrow_lengths(_field, _field_rho_indices):
                _sampled = _field[np.ix_(_field_rho_indices, _face_indices)]
                _max_abs = float(np.max(np.abs(_sampled)))
                if _max_abs == 0.0:
                    return np.zeros_like(_sampled)
                return _arrow_length * _sampled / _max_abs

            _velocity_quiver = _phi_ax.quiver(
                _velocity_z_grid,
                _velocity_rho_grid,
                _normalized_arrow_lengths(_face_velocity, _rho_indices),
                np.zeros_like(_velocity_z_grid),
                angles="xy",
                scale_units="xy",
                scale=1,
                color="#2563eb",
                width=0.003,
                label="Face velocity (normalized)",
            )
            _flux_quiver = _phi_ax.quiver(
                _flux_z_grid,
                _flux_rho_grid,
                _normalized_arrow_lengths(_face_flux, _flux_rho_indices),
                np.zeros_like(_flux_z_grid),
                angles="xy",
                scale_units="xy",
                scale=1,
                color="#dc2626",
                width=0.003,
                label="Face flux (normalized)",
            )
            _phi_ax.quiverkey(
                _velocity_quiver,
                0.84,
                1.05,
                0.5 * _arrow_length,
                "velocity: independently normalized",
                labelpos="E",
            )
            _phi_ax.quiverkey(
                _flux_quiver,
                0.84,
                1.11,
                0.5 * _arrow_length,
                "flux: independently normalized",
                labelpos="E",
            )
            _phi_ax.legend(loc="upper left")

        phi_panel = mo.vstack(
            [
                mo.md("### Phi(z, rho)"),
                mo.as_html(_phi_figure),
            ],
            align="stretch",
        )
    return (phi_panel,)


@app.cell
def _(phi_panel):
    mo.stop(phi_panel is None)
    phi_panel
    return


@app.cell
def _(fft_time_index, inspect_run, inspect_time, run_h5, run_store_fields):
    if "percoll" not in run_store_fields:
        percoll_panel = None
    else:
        with h5py.File(run_h5, "r") as _h5:
            _percoll = np.asarray(
                _h5["fields/percoll"][fft_time_index], dtype=np.float64
            )

        _percoll_fig, _percoll_ax = plt.subplots(constrained_layout=True)
        _percoll_ax.plot(100.0 * inspect_run.z, 1100.0 - _percoll, color="#7c3aed")
        _percoll_ax.set_xlabel(r"$z$ [cm]")
        _percoll_ax.set_ylabel("Percoll field")
        _percoll_ax.set_title(
            rf"Percoll field at $t={inspect_time[fft_time_index]:.3f}\,\mathrm{{s}}$"
        )
        _percoll_ax.grid(alpha=0.3)
        percoll_panel = mo.vstack(
            [mo.md("### Percoll(z)"), mo.ui.matplotlib(_percoll_ax)],
            align="stretch",
        )
    return (percoll_panel,)


@app.cell
def _(
    fft_time_index,
    fft_z_start_index,
    fft_z_stop_index,
    inspect_run,
    inspect_time,
    inspect_z,
    run_h5,
):
    _psi_figure = plot_psi(
        inspect_run,
        vmin=0.0,
        vmax=100.0,
        cmap=get_rbc_cmap(),
        title=run_h5.parent.name,
    )
    _psi_figure.axes[0].axvline(
        inspect_time[fft_time_index],
        color="white",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
    )
    _psi_figure.axes[0].axhline(
        100.0 * inspect_z[fft_z_start_index],
        color="#fbbf24",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
    )
    _psi_figure.axes[0].axhline(
        100.0 * inspect_z[fft_z_stop_index],
        color="#fbbf24",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
    )
    psi_panel = mo.vstack(
        [
            mo.md("### Psi(z, t)"),
            mo.as_html(_psi_figure),
        ],
        align="stretch",
    )
    return (psi_panel,)


@app.cell
def _(fft_amplitudes, fft_mode_numbers, inspect_time):
    _fft_fig, _fft_ax = plt.subplots(constrained_layout=True)
    _fft_image = _fft_ax.pcolormesh(
        inspect_time,
        fft_mode_numbers[1:],
        fft_amplitudes[:, 1:].T,
        shading="nearest",
        cmap="viridis",
    )
    _fft_colorbar = _fft_fig.colorbar(_fft_image, ax=_fft_ax)
    _fft_colorbar.set_label(r"$A_n(t) = |\delta\hat{\psi}_n(t)|$")
    _fft_ax.set_xlabel(r"$t\;[s]$")
    _fft_ax.set_ylabel("Mode number n")
    _fft_ax.set_title("FFT amplitude by mode and time")
    fft_heatmap_panel = mo.vstack(
        [
            mo.md("### FFT Amplitude Heatmap"),
            mo.ui.matplotlib(_fft_ax),
        ],
        align="stretch",
    )
    return (fft_heatmap_panel,)


@app.cell
def _(fft_amplitudes, fft_mode_numbers, inspect_time):
    _traces_fig, _traces_ax = plt.subplots(constrained_layout=True)
    _modes = fft_mode_numbers[1:]
    _mode_colormap = plt.get_cmap("viridis")
    _mode_scale = max(1, int(_modes[-1] - _modes[0]))
    for _mode in _modes:
        _traces_ax.plot(
            inspect_time,
            fft_amplitudes[:, _mode],
            color=_mode_colormap((_mode - _modes[0]) / _mode_scale),
            linewidth=1.0,
        )
    _mode_colors = plt.cm.ScalarMappable(cmap=_mode_colormap)
    _mode_colors.set_clim(float(_modes[0]), float(_modes[-1]))
    _mode_colorbar = _traces_fig.colorbar(_mode_colors, ax=_traces_ax)
    _mode_colorbar.set_label("Mode number n")
    _traces_ax.set_xlabel(r"$t\;[s]$")
    _traces_ax.set_ylabel(r"$A_n(t) = |\delta\hat{\psi}_n(t)|$")
    _traces_ax.set_title("FFT amplitude of every mode over time")
    fft_traces_panel = mo.vstack(
        [
            mo.md("### FFT Amplitude Traces"),
            mo.ui.matplotlib(_traces_ax),
        ],
        align="stretch",
    )
    return (fft_traces_panel,)


@app.cell
def _(fft_amplitudes, fft_time_index, fft_wavelengths, inspect_time):
    fft_dominant_mode = 1 + np.argmax(fft_amplitudes[:, 1:], axis=1)
    fft_dominant_wavelength = np.asarray(
        fft_wavelengths[fft_dominant_mode], dtype=np.float64
    )
    _dominant_wavelength_cm = 100.0 * fft_dominant_wavelength

    _dominant_fig, _dominant_ax = plt.subplots(constrained_layout=True)
    _dominant_ax.plot(
        inspect_time,
        _dominant_wavelength_cm,
        color="#7c3aed",
        linewidth=1.5,
        drawstyle="steps-mid",
    )
    _dominant_ax.scatter(
        [inspect_time[fft_time_index]],
        [_dominant_wavelength_cm[fft_time_index]],
        color="#dc2626",
        zorder=3,
        label=f"step {fft_time_index}",
    )
    _dominant_ax.set_xlabel(r"$t\;[s]$")
    _dominant_ax.set_ylabel(r"$\lambda_{\mathrm{dom}}(t)\;[\mathrm{cm}]$")
    _dominant_ax.set_title(
        r"Wavelength of the mode with maximum $|\delta\hat{\psi}_n(t)|$"
    )
    _dominant_ax.legend()

    fft_dominant_wavelength_panel = mo.vstack(
        [
            mo.md("### Dominant Wavelength"),
            mo.ui.matplotlib(_dominant_ax),
        ],
        align="stretch",
    )
    return (
        fft_dominant_mode,
        fft_dominant_wavelength,
        fft_dominant_wavelength_panel,
    )


@app.cell(hide_code=True)
def _(
    fft_amplitudes,
    fft_dominant_mode,
    fft_dominant_wavelength,
    fft_dominant_wavelength_panel,
    fft_heatmap_panel,
    fft_n_points,
    fft_traces_panel,
    fft_time_index,
    fft_z,
    fft_z_range_panel,
    fft_z_start_index,
    fft_z_stop_index,
    inspect_run_md,
    percoll_panel,
    psi_panel,
):
    _coefficient_summary = mo.md(
        f"FFT amplitude array shape `{fft_amplitudes.shape}` (time steps × modes).  \n"
        + f"FFT window uses z indices `{fft_z_start_index}:{fft_z_stop_index}` inclusive, i.e. `{100.0 * fft_z[0]:.6g}` to `{100.0 * fft_z[-1]:.6g}` cm over `{fft_n_points}` grid points.  \n"
        + f"At step `{fft_time_index}`, the maximum-amplitude non-DC mode is `{fft_dominant_mode[fft_time_index]}` with wavelength `{100.0 * fft_dominant_wavelength[fft_time_index]:.6g}` cm."
    )

    mo.vstack(
        [
            inspect_run_md,
            mo.hstack(
                [
                    panel
                    for panel in [psi_panel, percoll_panel]
                    if panel is not None
                ],
                align="start",
                justify="start",
                gap=1,
            ),
            mo.vstack(
                [
                    mo.hstack(
                        [fft_heatmap_panel, fft_traces_panel],
                        align="start",
                        justify="start",
                        gap=1,
                    ),
                    fft_dominant_wavelength_panel,
                ],
                align="start",
                gap=1,
            ),
            mo.hstack(
                [fft_time_panel, fft_z_range_panel],
                align="start",
                justify="start",
                gap=1,
            ),
            _coefficient_summary,
        ],
        align="stretch",
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Peak Detection
    """)
    return


@app.cell
def cell_peaks(inspect_run):
    _z_cm = inspect_run.z * 100.0
    _psi_last_pct = inspect_run.load_psi()[-1] * 100.0
    _peak_z, _peak_psi, _peak_spacing, _peak_dev = find_peaks(_z_cm, _psi_last_pct)

    _fig, _ax = plt.subplots(constrained_layout=True)
    _ax.plot(_z_cm, _psi_last_pct, label=r"$\psi(z)$")
    # _ax.plot(_peak_z, _peak_psi, "x", color="red", label="Detected peaks")
    _ax.set_xlabel(r"$z \; [cm]$")
    _ax.set_ylabel(r"$\psi \; [\%]$")
    _ax.set_title("Peak detection")
    _ax.legend()

    _freq = 1.0 / _peak_spacing
    _freq_dev = _peak_dev / _peak_spacing**2  # error propagation
    _table = mo.md(
        f"""
    | Quantity | Value |
    |----------|-------|
    | **Number of peaks** | {len(_peak_z)} |
    | **λ** (avg. spacing) | {_peak_spacing:.4f} ± {_peak_dev:.4f} cm |
    | **ν** (frequency) | {_freq:.4f} ± {_freq_dev:.4f} cm⁻¹ |
    """
    )
    mo.vstack([mo.as_html(_fig), _table])
    return


if __name__ == "__main__":
    app.run()
