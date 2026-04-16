# /// script
# dependencies = [
#     "h5py==3.16.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "wigglystuff==0.3.3",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    from wigglystuff import CopyToClipboard

    return CopyToClipboard, Path, h5py, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initial Phi

    This notebook reproduces the legacy CUDA initializer for the initial
    two-dimensional RBC field `phi(rho, z)` and lets you export a frozen HDF5
    file for `--phi-file`.

    The exported file stores the field on `/phi/values` with shape `(N, N)` and
    storage order `phi[rho_idx, z_idx]`.
    """)
    return


@app.cell
def _(mo):
    n = mo.ui.number(start=8, stop=4096, step=1, value=256, label="N")
    psi = mo.ui.number(start=1e-6, stop=0.999999, step=1e-3, value=0.02, label="PSI")
    rho_center = mo.ui.number(
        start=1000.0, stop=1200.0, step=0.1, value=1100.0, label="Rmu"
    )
    rho_sigma = mo.ui.number(
        start=1e-3, stop=100.0, step=0.1, value=4.0, label="Rsigma"
    )
    wing = mo.ui.number(start=0, stop=1024, step=1, value=30, label="wingL")
    rho_center_axis = mo.ui.number(
        start=1000.0, stop=1200.0, step=0.1, value=1100.0, label="RC"
    )
    rho_span = mo.ui.number(start=1.0, stop=200.0, step=0.5, value=30.0, label="RL")

    mo.md(
        """
        ## Parameters

        {n}

        {psi}

        {rho_center}

        {rho_sigma}

        {wing}

        {rho_center_axis}

        {rho_span}
        """
    ).batch(
        n=n,
        psi=psi,
        rho_center=rho_center,
        rho_sigma=rho_sigma,
        wing=wing,
        rho_center_axis=rho_center_axis,
        rho_span=rho_span,
    )
    return n, psi, rho_center, rho_center_axis, rho_sigma, rho_span, wing


@app.cell
def _(n, np, psi, rho_center, rho_center_axis, rho_sigma, rho_span, wing):
    N = int(n.value)
    PSI = float(psi.value)
    Rmu = float(rho_center.value)
    Rsigma = float(rho_sigma.value)
    wingL = int(wing.value)
    RC = float(rho_center_axis.value)
    RL = float(rho_span.value)

    rho = np.linspace(RC - RL / 2.0, RC + RL / 2.0, N, dtype=np.float64)
    rho_idx = np.arange(N, dtype=np.int32)
    z_idx = np.arange(N, dtype=np.int32)
    radial_profile = np.exp(-((rho - Rmu) ** 2) / (2.0 * Rsigma**2))
    phi = np.repeat(radial_profile[:, None], N, axis=1)

    if wingL > 0:
        z_mask = (z_idx < wingL + 2) | (z_idx > (N - 1 - (wingL + 2)))
        rho_mask = (rho_idx < wingL) | (rho_idx > (N - 1 - wingL))
        phi[:, z_mask] = 0.0
        phi[rho_mask, :] = 0.0

    phi_sum = float(phi.sum())
    if phi_sum > 0.0:
        phi = phi / phi_sum * PSI * (N - 2 * (wingL + 2))

    psi_profile = phi.sum(axis=0)
    return N, PSI, RC, RL, Rmu, Rsigma, phi, phi_sum, psi_profile, rho, wingL


@app.cell
def _(N, mo, phi, plt, rho):
    def _():
        z_extent = [0, N - 1]
        rho_extent = [float(rho[0]), float(rho[-1])]

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(
            phi,
            origin="lower",
            aspect="auto",
            extent=[z_extent[0], z_extent[1], rho_extent[0], rho_extent[1]],
            cmap="magma",
        )
        ax.set_xlabel("z index")
        ax.set_ylabel(r"$\rho$ [g/L]")
        ax.set_title("Initial phi field")
        fig.colorbar(im, ax=ax, label=r"$\phi$")
        return mo.ui.matplotlib(ax)

    _()
    return


@app.cell
def _(mo, plt, psi_profile):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(psi_profile, color="tab:blue", linewidth=2)
    ax.set_xlabel("z index")
    ax.set_ylabel(r"$\sum_\rho \phi$")
    ax.set_title("Derived psi profile")
    ax.grid(True, linestyle=":", alpha=0.6)
    mo.ui.matplotlib(ax)
    return


@app.cell(hide_code=True)
def _(N, PSI, mo, phi, phi_sum, psi_profile, wingL):
    mo.md(rf"""
    ## Summary

    - `N = {N}`
    - `PSI = {PSI:.6g}`
    - raw pre-normalization sum: `{phi_sum:.6e}`
    - exported sum of `phi`: `{float(phi.sum()):.6e}`
    - mean of derived discrete `psi(z)`: `{float(psi_profile.mean()):.6e}`
    - `wingL = {wingL}`
    """)
    return


@app.cell
def _(Path, mo):
    export_dir = mo.ui.file_browser(
        initial_path=Path.cwd(),
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="directory",
        label="Export directory",
    )
    export_name = mo.ui.text(value="initial_phi.h5", label="File name")

    export_form = (
        mo.md(
            """
        ## Export

        {export_dir}

        {export_name}
        """
        )
        .batch(export_dir=export_dir, export_name=export_name)
        .form(
            submit_button_label="Export phi file",
            clear_on_submit=False,
            show_clear_button=True,
        )
    )
    export_form
    return (export_form,)


@app.cell
def _(
    CopyToClipboard,
    N,
    PSI,
    Path,
    RC,
    RL,
    Rmu,
    Rsigma,
    export_form,
    h5py,
    mo,
    np,
    phi,
    rho,
    wingL,
):
    if export_form.value is None:
        result = mo.md(
            "Submit the form to export the CUDA-compatible initial phi file."
        )
    else:
        export_dir_entries = export_form.value.get("export_dir") or []
        file_name = str(export_form.value.get("export_name", "")).strip()

        if not export_dir_entries:
            result = mo.md("Please select a directory before exporting.")
        elif not file_name:
            result = mo.md("Please enter a file name before exporting.")
        else:
            selected_dir = Path(export_dir_entries[0].path)
            path = selected_dir / file_name
            z = np.arange(N, dtype=np.float64)

            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(path, "w") as f:
                    group = f.create_group("phi")
                    group.create_dataset(
                        "values", data=np.asarray(phi, dtype=np.float64)
                    )
                    group.create_dataset("rho", data=np.asarray(rho, dtype=np.float64))
                    group.create_dataset("z", data=np.asarray(z, dtype=np.float64))
                    group.attrs["N"] = int(N)
                    group.attrs["PSI"] = float(PSI)
                    group.attrs["RC"] = float(RC)
                    group.attrs["RL"] = float(RL)
                    group.attrs["Rmu"] = float(Rmu)
                    group.attrs["Rsigma"] = float(Rsigma)
                    group.attrs["wingL"] = int(wingL)
                    group.attrs["storage_order"] = "phi[rho_idx, z_idx]"
                    group.attrs["cuda_compatible"] = 1
                    group.attrs["generated_by"] = "analysis/phi_init.py"
                    group.attrs["normalization"] = "no runtime renormalization required"
            except Exception as exc:
                result = mo.md(rf"Export failed: `{exc}`")
            else:
                clipboard = mo.ui.anywidget(CopyToClipboard(text_to_copy=str(path)))
                result = mo.vstack(
                    [
                        mo.md(rf"Exported initial phi to `{path}`."),
                        mo.md(r"Use it in the simulation with `--phi-file <path>`."),
                        mo.hstack(
                            [mo.md("Copy exported path:"), clipboard], justify="start"
                        ),
                    ]
                )

    result
    return


if __name__ == "__main__":
    app.run()
