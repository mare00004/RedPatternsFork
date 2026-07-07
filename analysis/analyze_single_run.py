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
        r"""Find the peaks of $\psi(z)$

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


@app.cell
def _(Array1F, mo, np, peak_spacing, plt, psi, z):
    def fft_dominant_wavelengths(
        z: Array1F,
        psi: Array1F,
        n_components: int = 5,
        apply_window: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Find dominant wavelength components using FFT of $\psi(z)$.

        Args:
            z: 1D spatial coordinate array [cm]
            psi: 1D signal array
            n_components: number of dominant components to return
            apply_window: whether to apply a Hann window to reduce spectral leakage

        Returns:
            Tuple of (wavelengths [cm], spatial frequencies [cm⁻¹], power)
            sorted by power in descending order (DC component excluded).
        """
        dz = z[1] - z[0]
        n = len(psi)

        signal = psi.copy()
        if apply_window:
            signal = signal * np.hanning(n)

        fft_vals = np.fft.rfft(signal)
        fft_freqs = np.fft.rfftfreq(n, d=dz)

        power = np.abs(fft_vals) ** 2

        # Exclude DC component (index 0) — its "wavelength" is infinite
        nonzero_mask = fft_freqs > 0
        freqs_nz = fft_freqs[nonzero_mask]
        power_nz = power[nonzero_mask]

        # Sort by power descending
        sorted_idx = np.argsort(power_nz)[::-1]
        top_idx = sorted_idx[:n_components]

        dominant_freqs = freqs_nz[top_idx]
        dominant_wavelengths = 1.0 / dominant_freqs
        dominant_powers = power_nz[top_idx]

        return dominant_wavelengths, dominant_freqs, dominant_powers


    _fft_wavelengths, _fft_freqs, _fft_powers = fft_dominant_wavelengths(z, psi)

    # --- FFT Power Spectrum Plot ---
    fig_fft, ax_fft = plt.subplots(constrained_layout=True)
    ax_fft.plot(_fft_freqs, _fft_powers, "o-", markersize=3, label="FFT Power")
    ax_fft.set_xlabel(r"Spatial Frequency $\nu$ [cm$^{-1}$]")
    ax_fft.set_ylabel("Power")
    ax_fft.set_title("FFT Power Spectrum of $\\psi(z)$")

    # Mark dominant components
    for _i, (_f, _wl, _p) in enumerate(zip(_fft_freqs, _fft_wavelengths, _fft_powers)):
        ax_fft.annotate(
            f"#{_i+1}: λ={_wl:.2f} cm",
            xy=(_f, _p),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="red",
        )
        ax_fft.plot(_f, _p, "rx", markersize=8)

    ax_fft.legend()

    _fft_plot = mo.ui.matplotlib(ax_fft)

    # --- Table ---
    _n_show = min(5, len(_fft_wavelengths))
    _fft_table_rows = "\n".join(
        f"| {i+1} | {_fft_wavelengths[i]:.4f} | {_fft_freqs[i]:.4f} | {_fft_powers[i]:.2e} |"
        for i in range(_n_show)
    )

    _fft_table = mo.md(
        f"""
    ### FFT Dominant Wavelength Components

    | # | Wavelength [cm] | Spatial Freq [cm⁻¹] | Power |
    |---|-----------------|---------------------|-------|
    {_fft_table_rows}

    **Peak spacing from find_peaks:** {peak_spacing:.4f} cm  
    **Dominant FFT wavelength:** {_fft_wavelengths[0]:.4f} cm
    """
    )

    mo.vstack([_fft_plot, _fft_table], align="stretch")
    return


if __name__ == "__main__":
    app.run()
