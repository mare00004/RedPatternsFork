# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "h5py==3.15.1",
#     "marimo>=0.19.4",
#     "matplotlib==3.10.8",
#     "numpy==2.4.1",
#     "pandas==3.0.0",
#     "scipy==1.17.1",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    from red_patterns import (
        RunData,
        plot_psi,
        plot_psi_arrays,
        cli_args_from_run_h5,
        get_rbc_cmap,
        Array1F,
    )
    import numpy as np
    from scipy.signal import find_peaks as scipy_find_peaks

    return (
        Array1F,
        Path,
        RunData,
        get_rbc_cmap,
        mo,
        np,
        plot_psi,
        plt,
        scipy_find_peaks,
    )


@app.cell
def _(Path, RunData, plot_psi, plt):
    def plot_psi_file(run_h5: Path, **kwargs) -> plt.Figure:
        run = RunData.from_h5(run_h5, load_fields=False)
        return plot_psi(run, **kwargs)

    return (plot_psi_file,)


@app.cell
def _(Path, mo):
    file_picker = mo.ui.file_browser(
        initial_path=Path.cwd(),
        filetypes=[".h5"],
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="file",
        label="Choose run.h5 file to plot!",
    )
    file_picker
    return (file_picker,)


@app.cell
def _(file_picker, get_rbc_cmap, mo, plot_psi_file):
    mo.stop(not file_picker.value, mo.md("Please pick a file to plot!"))

    result = plot_psi_file(
        file_picker.path(), vmin=0.0, vmax=100.0, cmap=get_rbc_cmap()
    )
    return (result,)


@app.cell
def _(RunData, file_picker):
    run = RunData.from_h5(file_picker.path(), load_fields=False)
    z = run.z * 100
    psi = run.load_psi()[-1] * 100  # psi at last time step
    t = run.time
    return psi, z


@app.cell
def _(Array1F, np, scipy_find_peaks):
    def find_peaks(
        z: Array1F,
        psi: Array1F,
    ) -> tuple[Array1F, Array1F, float, float]:
        """Find the peaks of $\psi(z)$

        Args:
            psi: 1D array with shape (z)
            z: 1D array with shape (z)

        Returns:
            Tuple[z-coordinates of peaks, psi(z) for each peak, average spacing, uncertenty of average spacing]
        """
        peak_indices, _ = scipy_find_peaks(psi, prominence=0.1)

        # Ignore peaks that lay at the edges
        # - You can only remove the first peak, the last peak, or no peaks at all
        # - The first peak only gets removed when its z-coordinate is < 1
        # - The last peak only gets removed when its z-coordinate is > 6
        # - If the z-coordinates of the first two peaks are < 1, then only the first peak gets removed
        peak_indices = peak_indices[
            1 if z[peak_indices[0]] < 1 else 0 : -1 if z[peak_indices[-1]] > 6 else None
        ]

        peak_z = z[peak_indices]
        peak_psi = psi[peak_indices]

        distances = np.diff(peak_z)
        average_spacing = np.mean(distances)
        standard_deviation = np.std(distances)

        return tuple([peak_z, peak_psi, average_spacing, standard_deviation])

    return (find_peaks,)


@app.cell
def _(find_peaks, psi, z):
    peak_z, peak_psi, peak_spacing, peak_deviation = find_peaks(z, psi)
    return peak_deviation, peak_psi, peak_spacing, peak_z


@app.cell
def _(mo, peak_deviation, peak_psi, peak_spacing, peak_z, plt, psi, z):
    # Plot
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(z, psi, label=r"$\psi (z)$")
    ax.plot(peak_z, peak_psi, "x", color="red", label="Detected Peaks")
    ax.set_xlabel(r"$z \; [cm]$")
    ax.set_ylabel(r"$\psi \; [\%]$")
    ax.legend()
    ax.set_title(f"Peak Detection")

    plot = mo.ui.matplotlib(ax)

    # Table
    _std_spacing = peak_deviation
    _frequency = 1.0 / peak_spacing
    _std_frequency = _std_spacing / peak_spacing**2  # error propagation: δν = δλ / λ²

    _n_peaks = len(peak_z)

    table = mo.md(
        f"""
    | Quantity | Value |
    |----------|-------|
    | **Number of peaks** | {_n_peaks} |
    | **λ** (avg. peak spacing) | {peak_spacing:.4f} ± {_std_spacing:.4f} cm |
    | **ν** (spatial frequency) | {_frequency:.4f} ± {_std_frequency:.4f} cm⁻¹ |
    """
    )
    return plot, table


@app.cell
def _(mo, plot, result, table):
    mo.vstack(
        [mo.hstack([result, plot], align="center", justify="end"), table],
        align="stretch",
    )
    return


if __name__ == "__main__":
    app.run()
