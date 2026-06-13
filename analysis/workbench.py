# /// script
# dependencies = [
#     "h5py==3.16.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "scipy==1.17.1",
#     "wigglystuff==0.3.3",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")

with app.setup:
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path

    import marimo as mo
    from wigglystuff import ProgressBar

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
        compute_phi,
        make_phi_ui,
        phi_config_from_ui,
        phi_ui_layout,
        plot_phi,
        write_phi_h5,
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
    phi_cfg = phi_config_from_ui(phi_ui.value)
    phi_result = compute_phi(phi_cfg)
    return phi_cfg, phi_result


@app.cell
def _(phi_result):
    plot_phi(phi_result)
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
        },
        value="sigmoid",
    )
    ui_t_final = mo.ui.number(start=0.1, step=0.1, value=1000.0, label="T")
    ui_dt = mo.ui.number(start=1e-5, step=1e-4, value=1e-2, label="DT")
    ui_save_every = mo.ui.number(start=1, step=1, value=500, label="NO")
    ui_run_button = mo.ui.run_button(label="Build inputs and run simulation")
    mo.vstack(
        [ui_mode, ui_gradient, ui_t_final, ui_dt, ui_save_every, ui_run_button],
        gap=1,
    )
    return (
        ui_dt,
        ui_gradient,
        ui_mode,
        ui_mu,
        ui_nu,
        ui_run_button,
        ui_save_every,
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
    ui_save_every,
    ui_t_final,
    ui_taylor_source,
):
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

        write_phi_h5(_phi_path, phi_result, _replace(phi_cfg, output_path=_phi_path))
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
            t_final=float(ui_t_final.value),
            dt=float(ui_dt.value),
            save_every=int(ui_save_every.value),
            nu=_nu,
            mu=_mu,
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
    return (run_h5,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Inspect the result
    """)
    return


@app.cell
def cell_psi_plot(run_h5):
    mo.stop(
        run_h5 is None or not run_h5.exists(),
        mo.md("Run a simulation to inspect the resulting `run.h5`."),
    )
    run = RunData.from_h5(run_h5, load_fields=False)
    plot_psi(run, vmin=0.0, vmax=100.0, cmap=get_rbc_cmap(), title=run_h5.parent.name)
    return (run,)


@app.cell
def cell_peaks(run):
    z = run.z * 100  # m -> cm
    psi = run.load_psi()[-1] * 100  # last timestep, fraction -> %
    peak_z, peak_psi, peak_spacing, peak_dev = find_peaks(z, psi)

    import matplotlib.pyplot as _plt

    _fig, _ax = _plt.subplots(constrained_layout=True)
    _ax.plot(z, psi, label=r"$\psi(z)$")
    _ax.plot(peak_z, peak_psi, "x", color="red", label="Detected peaks")
    _ax.set_xlabel(r"$z \; [cm]$")
    _ax.set_ylabel(r"$\psi \; [\%]$")
    _ax.set_title("Peak detection")
    _ax.legend()

    _freq = 1.0 / peak_spacing
    _freq_dev = peak_dev / peak_spacing**2  # error propagation
    _table = mo.md(
        f"""
    | Quantity | Value |
    |----------|-------|
    | **Number of peaks** | {len(peak_z)} |
    | **λ** (avg. spacing) | {peak_spacing:.4f} ± {peak_dev:.4f} cm |
    | **ν** (frequency) | {_freq:.4f} ± {_freq_dev:.4f} cm⁻¹ |
    """
    )
    mo.vstack([mo.as_html(_fig), _table])
    return


if __name__ == "__main__":
    app.run()
