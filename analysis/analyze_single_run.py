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

__generated_with = "0.19.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from __future__ import annotations

    from dataclasses import dataclass, asdict
    from pathlib import Path
    import re
    import shlex
    from typing import Optional, Union
    import numpy as np
    import pandas as pd
    import h5py
    import matplotlib.pyplot as plt
    import marimo as mo
    return Path, h5py, mo, np, plt


@app.cell
def _(np):
    # Start + [0, ..., N-1] * step
    z_vals = 0.0 + (np.arange(256) * 2.676511e-04)
    return (z_vals,)


@app.cell
def _(Path, h5py, np, plt, z_vals):
    def plot_run(
        run_h5: Path,
        *,
        vmin: float = -0.2,
        vmax: float = 1.5,
        max_t_pixels: int = 2400,  # downsample time if larger
        max_z_pixels: int = 1200,  # downsample z if larger
        interpolation: str = "nearest",
    ) -> plt.Figure:
        """
        Plot psi(t,z) from run.h5 as an image with correct axes scaling.

        Assumes:
          - /psi has shape (nt, nz) or (time, z)
          - z_vals is a 1D numpy array of length nz (global/outer scope)
        """
        with h5py.File(run_h5, "r") as f:
            psi_ds = f["/psi"]
            t_ds = f["/time"]

            nt, nz = psi_ds.shape

            # sanity check against global z_vals
            if len(z_vals) != nz:
                raise ValueError(
                    f"z_vals has length {len(z_vals)} but /psi has nz={nz}"
                )

            # choose strides to cap displayed resolution (keeps plotting fast)
            t_step = max(1, nt // max_t_pixels)
            z_step = max(1, nz // max_z_pixels)

            t_idx = slice(None, None, t_step)
            z_idx = slice(None, None, z_step)

            # read only what we will actually display
            psi = psi_ds[t_idx, z_idx]  # (nt', nz')
            t = np.asarray(t_ds[t_idx])  # (nt',)
            z = np.asarray(z_vals[z_idx])  # (nz',)

        # imshow expects (rows, cols) = (z, t)
        C = np.asarray(psi, dtype=np.float32).T

        fig, ax = plt.subplots(constrained_layout=True)

        # Map pixels to your physical axes
        extent = (float(t[0]), float(t[-1]), float(z[0]), float(z[-1]))

        im = ax.imshow(
            C,
            origin="lower",
            aspect="auto",
            interpolation=interpolation,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )

        ax.set_xlabel("t")
        ax.set_ylabel("z")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$\psi(t,z)$")

        return fig
    return (plot_run,)


@app.cell
def _(Path, mo):
    file_picker = mo.ui.file_browser(
        initial_path=Path.cwd().parent / "data",
        filetypes=[".h5"],
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="file",
        label="Choose run.h5 file to plot!"
    )
    file_picker
    return (file_picker,)


@app.cell
def _(file_picker, plot_run):
    plot_run(file_picker.path(), vmin=-0.1, vmax=1.1)
    return


if __name__ == "__main__":
    app.run()
