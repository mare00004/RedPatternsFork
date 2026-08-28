# /// script
# dependencies = [
#     "h5py>=3.16.0",
#     "marimo>=0.23.6",
#     "matplotlib>=3.10.9",
#     "numpy>=2.4.5",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="wide")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    NOTEBOOK_FILE = (
        Path(__file__).resolve()
        if "__file__" in globals()
        else (Path.cwd() / "analysis" / "compare_simulations.py").resolve()
    )
    ANALYSIS_DIR = NOTEBOOK_FILE.parent
    REPO_ROOT = ANALYSIS_DIR.parent
    if str(ANALYSIS_DIR) not in sys.path:
        sys.path.insert(0, str(ANALYSIS_DIR))

    from red_patterns import RunData, get_rbc_cmap


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Compare two simulations

    Select two `run.h5` files to compare their full $\psi(z,t)$ fields. The first
    simulation is the reference: the difference panel shows
    $\Delta\psi = \psi_2 - \psi_1$.
    """)
    return


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _():
    run_1_browser = mo.ui.file_browser(
        initial_path=REPO_ROOT,
        filetypes=[".h5"],
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="file",
        label="Simulation 1 (reference)",
        restrict_navigation=False,
    )
    run_2_browser = mo.ui.file_browser(
        initial_path=REPO_ROOT,
        filetypes=[".h5"],
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="file",
        label="Simulation 2",
        restrict_navigation=False,
    )
    return run_1_browser, run_2_browser


@app.cell(hide_code=True)
def _(run_1_browser, run_2_browser):
    mo.hstack([run_1_browser, run_2_browser], widths="equal", gap=1.5)
    return


@app.cell
def _(run_1_browser, run_2_browser):
    run_1_path = Path(run_1_browser.path()) if run_1_browser.value else None
    run_2_path = Path(run_2_browser.path()) if run_2_browser.value else None
    return run_1_path, run_2_path


@app.cell
def _(is_script_mode, run_1_path, run_2_path):
    def load_run(path: Path):
        try:
            run = RunData.from_h5(path, load_fields=False)
            psi = np.asarray(run.load_psi(), dtype=np.float64)
            time = np.asarray(run.time, dtype=np.float64)
            z = np.asarray(run.z, dtype=np.float64)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return None, f"{type(exc).__name__}: {exc}"
        return (psi, time, z), None

    if is_script_mode:
        time_1 = np.linspace(0.0, 60.0, 121)
        z_1 = np.linspace(0.0, 0.06, 160)
        time_mesh = time_1[:, None]
        z_mesh = z_1[None, :]
        envelope = 1.0 - np.exp(-time_mesh / 15.0)
        psi_1 = 0.02 + 0.004 * envelope * np.sin(2.0 * np.pi * z_mesh / 0.012)
        psi_2 = 0.02 + 0.0042 * envelope * np.sin(
            2.0 * np.pi * (z_mesh - 0.0003) / 0.012
        )
        time_2 = time_1.copy()
        z_2 = z_1.copy()
        run_1_label = "Synthetic reference"
        run_2_label = "Synthetic comparison"
    else:
        mo.stop(
            run_1_path is None or run_2_path is None,
            mo.md("Select both `run.h5` files to start the comparison."),
        )
        assert run_1_path is not None and run_2_path is not None
        run_1_data, run_1_error = load_run(run_1_path)
        run_2_data, run_2_error = load_run(run_2_path)
        load_errors = [
            message
            for message in (
                f"Simulation 1: {run_1_error}" if run_1_error else None,
                f"Simulation 2: {run_2_error}" if run_2_error else None,
            )
            if message is not None
        ]
        mo.stop(
            bool(load_errors),
            mo.callout(
                mo.md("**Could not load the selected run:**\n\n" + "\n\n".join(load_errors)),
                kind="danger",
            ),
        )
        assert run_1_data is not None and run_2_data is not None
        psi_1, time_1, z_1 = run_1_data
        psi_2, time_2, z_2 = run_2_data
        run_1_label = str(run_1_path)
        run_2_label = str(run_2_path)
    return psi_1, psi_2, run_1_label, run_2_label, time_1, time_2, z_1, z_2


@app.cell
def _(psi_1, psi_2, time_1, time_2, z_1, z_2):
    def grid_diagnostic() -> str | None:
        if psi_1.ndim != 2 or psi_2.ndim != 2:
            return f"Expected 2D psi arrays; got {psi_1.shape} and {psi_2.shape}."
        if time_1.ndim != 1 or time_2.ndim != 1 or z_1.ndim != 1 or z_2.ndim != 1:
            return "The time and z coordinates must be one-dimensional."
        if psi_1.shape != (time_1.size, z_1.size):
            return f"Simulation 1 psi shape {psi_1.shape} does not match its axes."
        if psi_2.shape != (time_2.size, z_2.size):
            return f"Simulation 2 psi shape {psi_2.shape} does not match its axes."
        if psi_1.shape != psi_2.shape:
            return f"Psi shape mismatch: {psi_1.shape} versus {psi_2.shape}."
        if not np.allclose(time_1, time_2, rtol=1e-6, atol=1e-9):
            return "The simulations use different time coordinates."
        if not np.allclose(z_1, z_2, rtol=1e-6, atol=1e-9):
            return "The simulations use different z coordinates."
        if not np.all(np.isfinite(psi_1)) or not np.all(np.isfinite(psi_2)):
            return "At least one psi field contains NaN or infinite values."
        if time_1.size == 0 or z_1.size == 0:
            return "The selected simulations contain an empty coordinate axis."
        if np.any(np.diff(time_1) <= 0.0) or np.any(np.diff(z_1) <= 0.0):
            return "The time and z coordinates must be strictly increasing."
        return None

    comparison_error = grid_diagnostic()
    mo.stop(
        comparison_error is not None,
        mo.callout(mo.md(f"**Cannot compare these runs:** {comparison_error}"), kind="danger"),
    )
    comparison_ready = True
    return (comparison_ready,)


@app.cell
def _(comparison_ready, psi_1, psi_2):
    assert comparison_ready
    psi_1_pct = 100.0 * psi_1
    psi_2_pct = 100.0 * psi_2
    delta_psi_pct = psi_2_pct - psi_1_pct

    rmse_pct = float(np.sqrt(np.mean(np.square(delta_psi_pct))))
    reference_l2 = float(np.linalg.norm(psi_1))
    relative_l2 = (
        None
        if reference_l2 == 0.0
        else float(np.linalg.norm(psi_2 - psi_1) / reference_l2)
    )
    return delta_psi_pct, psi_1_pct, psi_2_pct, relative_l2, rmse_pct


@app.cell(hide_code=True)
def _(relative_l2, rmse_pct, run_1_label, run_2_label):
    relative_l2_text = (
        "undefined (the reference norm is zero)"
        if relative_l2 is None
        else f"{relative_l2:.6g}"
    )
    mo.vstack(
        [
            mo.md(
                f"**Simulation 1:** `{run_1_label}`  \n"
                f"**Simulation 2:** `{run_2_label}`"
            ),
            mo.callout(
                mo.md(
                    rf"""
                    **RMSE:** ${rmse_pct:.6g}$ percentage points  
                    **Relative $L^2$ error:** `{relative_l2_text}`

                    $\mathrm{{RMSE}}=\sqrt{{\operatorname{{mean}}((\psi_2-\psi_1)^2)}}$,
                    $\quad \varepsilon_{{L^2}}=\lVert\psi_2-\psi_1\rVert_2/\lVert\psi_1\rVert_2$.
                    """
                ),
                kind="info",
            ),
        ],
        align="stretch",
    )
    return


@app.function
def build_comparison_figure(
    psi_reference,
    psi_comparison,
    difference,
    time,
    z,
):
    difference_limit = max(float(np.max(np.abs(difference))), 1e-12)
    extent = (
        float(time[0]),
        float(time[-1]),
        float(100.0 * z[0]),
        float(100.0 * z[-1]),
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    source_images = []
    for axis, values, title in zip(
        axes[:2],
        (psi_reference, psi_comparison),
        (r"Simulation 1: $\psi_1(z,t)$", r"Simulation 2: $\psi_2(z,t)$"),
        strict=True,
    ):
        image = axis.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            vmin=0.0,
            vmax=100.0,
            cmap=get_rbc_cmap(),
        )
        source_images.append(image)
        axis.set_title(title)
        axis.set_xlabel(r"$t\;[\mathrm{s}]$")
        axis.set_ylabel(r"$z\;[\mathrm{cm}]$")

    difference_image = axes[2].imshow(
        difference.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-difference_limit, vcenter=0.0, vmax=difference_limit),
    )
    axes[2].set_title(r"Difference: $\psi_2-\psi_1$")
    axes[2].set_xlabel(r"$t\;[\mathrm{s}]$")
    axes[2].set_ylabel(r"$z\;[\mathrm{cm}]$")

    figure.colorbar(
        source_images[0],
        ax=axes[:2],
        shrink=0.9,
        pad=0.02,
        label=r"$\psi\;[\%]$",
    )
    figure.colorbar(
        difference_image,
        ax=axes[2],
        shrink=0.9,
        pad=0.02,
        label=r"$\Delta\psi$ [percentage points]",
    )
    return figure, axes


@app.cell(hide_code=True)
def _(delta_psi_pct, psi_1_pct, psi_2_pct, time_1, z_1):
    comparison_figure, comparison_axes = build_comparison_figure(
        psi_1_pct,
        psi_2_pct,
        delta_psi_pct,
        time_1,
        z_1,
    )
    mo.ui.matplotlib(comparison_axes[0])
    return


if __name__ == "__main__":
    app.run()
