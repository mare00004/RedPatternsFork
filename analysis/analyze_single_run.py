# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "h5py==3.15.1",
#     "marimo>=0.19.4",
#     "matplotlib==3.10.8",
#     "numpy==2.4.1",
#     "pandas==3.0.0",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    from red_patterns import RunData, plot_psi

    return Path, RunData, mo, plot_psi, plt


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

    result = plot_psi_file(file_picker.path(), vmin=0.0, vmax=0.2)
    result
    return


if __name__ == "__main__":
    app.run()
