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

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    from red_patterns import RunData, plot_psi, plot_psi_arrays, cli_args_from_run_h5
    import numpy as np

    return (
        Path,
        RunData,
        cli_args_from_run_h5,
        mo,
        np,
        plot_psi,
        plot_psi_arrays,
        plt,
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
def _(file_picker, mo, plot_psi_file):
    mo.stop(not file_picker.value, mo.md("Please pick a file to plot!"))

    result = plot_psi_file(file_picker.path(), vmin=0.0, vmax=0.3, cmap="viridis")
    result
    return


@app.cell
def _(file_picker, mo, np, psi, t, z):
    output_path = file_picker.path().with_suffix(".npz")
    np.savez_compressed(output_path, t=t, z=z, psi=psi)
    mo.md(f"Saved to `{output_path}`")
    return


@app.cell
def _(np):
    data = np.load("/home/max/projects/RedPatternsFork/data/conv_const_linear/run.npz")
    z_loaded = data["z"]
    psi_loaded = data["psi"]
    t_loaded = data["t"]
    return


@app.cell
def _(cli_args_from_run_h5, file_picker):
    cli = cli_args_from_run_h5(file_picker.path())
    cli
    return


@app.cell
def _(RunData, file_picker, plot_psi_arrays):
    run = RunData.from_h5(file_picker.path(), load_fields=False)
    z = run.z
    psi = run.load_psi()[-1] # psi at last time step
    t = run.time
    git_commit = run.config.git_commit
    plot_psi_arrays(run.psi, t, z, cmap="viridis", vmin=0.0, vmax=0.3)
    return psi, t, z


@app.cell
def _(np, plt, psi, z):
    from scipy.signal import find_peaks
    x = z
    y = psi
    peak_indices, properties = find_peaks(y, prominence=0.01)

    # Extract the actual x and y values of the peaks
    peak_x = x[peak_indices]
    peak_y = y[peak_indices]

    # ---------------------------------------------------------
    # 3. Calculate Spacing and Frequency
    # ---------------------------------------------------------
    # Calculate the distance between each consecutive peak
    distances = np.diff(peak_x)
    print(f"Standard Deviation: {np.std(distances)}")
    # Calculate the average spacing
    average_spacing = np.mean(distances)

    # Calculate the spatial frequency (1 / spacing)
    frequency = 1.0 / average_spacing

    # ---------------------------------------------------------
    # 4. Output Results
    # ---------------------------------------------------------
    print(f"Total peaks found: {len(peak_indices)}")
    print(f"Average Peak Spacing (Δx): {average_spacing:.6f}")
    print(f"Spatial Frequency (1/Δx): {frequency:.2f}")

    # Optional: Visualize to ensure the algorithm caught the right peaks
    plt.plot(x, y, label="Signal")
    plt.plot(peak_x, peak_y, "x", color="red", label="Detected Peaks")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title(f"Peak Detection (Avg Spacing: {average_spacing:.5f})")
    return


if __name__ == "__main__":
    app.run()
