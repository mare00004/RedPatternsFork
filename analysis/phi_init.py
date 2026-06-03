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

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    from red_patterns import Array1F, Array2F
    import numpy as np


@app.cell
def _():
    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    from pathlib import Path
    from wigglystuff import CopyToClipboard

    return CopyToClipboard, Path, h5py, mo, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initial Phi

    The exported file stores the field on `/phi/values` with shape `(N, N)` and
    storage order `phi[rho_idx, z_idx]`.
    """)
    return


@app.function
def phi_homogeneous(rho: Array1F, z: Array1F, psi_avg: np.floating) -> Array2F:
    """
    Calcualtes
    $$
        \\varphi(\\rho, z) = \\frac{\\langle \\psi \\rangle}{L_\\rho}
    $$
    ensuring the normalization
    $$
        \\langle \\psi \\rangle = \\frac{1}{L_z} \\int_J \\int_I \\varphi(\\rho,z) d \\rho dz
    $$
    is satisfied. Where $J = \\texttt{rho}$, $I = \\texttt{z}$ are the domains and $L_\\rho$ and $L_z$ are the respective lengths.
    """
    N_rho = rho.shape[0]
    N_z = z.shape[0]
    rho_len = rho[-1] - rho[0]
    return (psi_avg / rho_len) * np.ones((N_rho, N_z), dtype=np.float32)


@app.function
def phi_gaussian(rho: Array1F, z: Array1F, psi_avg: np.floating, mu: np.floating, sigma: np.floating) -> Array2F:
    """
    Calculates
    $$
        \\varphi(\\rho, z) = \\langle \\psi \\rangle \\frac{1}{\\sqrt{2 \\pi \\sigma_\\rho^2}} \\exp\\left(-\\frac{(\\rho - \\mu)^2}{2 \\sigma_\\rho^2}\\right)
    $$
    (which is just the probability density funciton of the normal distribution scaled by `psi_avg`). The normalization
    $$
        \\langle \\psi \\rangle = \\frac{1}{L_z} \\int_J \\int_I \\varphi(\\rho,z) d \\rho dz
    $$
    is satisfied. Where $J = \\texttt{rho}$, $I = \\texttt{z}$ are the domains and $L_\\rho$ and $L_z$ are the respective lengths.
    """
    N_z = z.shape[0]
    radial_profile = psi_avg * (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((rho - mu) ** 2) / (2.0 * sigma**2))
    return radial_profile[:, np.newaxis] * np.ones(N_z)


@app.function
def phi_add_wing(phi, wing):
    N_rho, N_z = phi.shape
    rho_idx = np.arange(N_rho)
    z_idx = np.arange(N_z)
    z_mask = (z_idx < wing + 2) | (z_idx > (N_z - 1 - (wing + 2)))
    rho_mask = (rho_idx < wing) | (rho_idx > (N_rho - 1 - wing))

    result = phi.copy()
    result[:, z_mask] = 0.0
    result[rho_mask, :] = 0.0
    return result


@app.function
def renormalize_phi(phi, rho, z, psi_avg, wing):
    N_rho, N_z = phi.shape

    z_start = wing + 2
    z_end = N_z - (wing + 2)
    n_z_eff = z_end - z_start  # N_z - 2*(wing+2), matches CUDA edgeZ = wingL+2

    # Index-space sum with no physical step factors — matches initPhi in simulations.cu:
    #   sum(phi) = PSI * (N - 2*edgeZ)  =>  mean over active z of psi_discrete = PSI
    psi_profile = phi.sum(axis=0)
    current_avg = psi_profile[z_start:z_end].sum() / n_z_eff

    if current_avg > 0:
        return phi * (psi_avg / current_avg)
    else:
        return phi.copy()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each of the selectable $\varphi$ distributions satisfy the following normalization condition:
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Normalization": mo.md(
                r"""
    Let $I, J \subseteq \mathbb{R}$ be Intervals. We call $I$ the density domain and $J$ the  $z$-domain. At time $t = 0$ we define

    $$
    	\varphi: I \times J, (\rho, z) \mapsto \phi(\rho, z)
    $$

    We remember the definition (at time $t = 0$)

    $$
    	\psi(z,t = 0) = \psi(z) := \int_I d\rho\,\varphi(\rho,z).
    $$

    of the specific volume fraction, then the normalization with respect to the Average Volume Fraction is

    $$
    	\langle \psi \rangle := \frac{1}{\int_J 1 d z} \int_J  \psi(z) d z = \frac{1}{L_z} \int_J \psi(z) d z
    $$

                """
            ),
            "Normalization with Wing": mo.md(
                r"""
    Adding a wing of size $w = \texttt{wingL}$ zeroes $\varphi$ at the grid boundaries:

    $$
        \varphi_{ij} = 0 \quad \text{if } j < w_z \text{ or } j > N-1-w_z, \qquad w_z := w + 2
    $$
    $$
        \varphi_{ij} = 0 \quad \text{if } i < w \phantom{_z} \text{ or } i > N-1-w
    $$

    where index $i$ runs over $\rho$ and index $j$ runs over $z$.
    This shrinks the active domain and breaks the original normalization, so $\varphi$ must be rescaled.

    Let $I' \subset I$ and $J' \subset J$ denote the active (non-wing) sub-domains. The continuous normalization condition becomes

    $$
        \langle \psi \rangle := \frac{1}{L_{z}'} \int_{J'} \psi(z)\, dz = \texttt{PSI}, \qquad L_z' = \int_{J'} dz
    $$

    where $\psi(z) = \int_{I'} \varphi(\rho, z)\, d\rho$ is now integrated only over the active $\rho$-domain.

    **Discrete implementation.** The CUDA simulation (`initPhi` in `simulations.cu`) uses a discrete, index-space normalization with no physical step-size factors.
    Defining the discrete $\psi$ profile as a plain index sum over $\rho$:

    $$
        \psi_j := \sum_{i=0}^{N-1} \varphi_{ij}
    $$

    the normalization condition is

    $$
        \langle \psi \rangle := \frac{1}{N - 2\,w_z} \sum_{j=w_z}^{N-1-w_z} \psi_j = \texttt{PSI}
    $$

    which is equivalent to

    $$
        \sum_{i,j} \varphi_{ij} = \texttt{PSI} \cdot (N - 2\,w_z).
    $$

    `renormalize_phi` computes the current $\langle \psi \rangle$ after the wing has been applied and rescales $\varphi$ uniformly by $\texttt{PSI} / \langle \psi \rangle$ to satisfy this condition.
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate a $\varphi$ distribution!

    ### 1. Pick the Average Volume Fraction
    """)
    return


@app.cell
def _(mo):
    ui_gaussian_text =  mo.md(
    r"""
    Gaussian $\varphi$ distribution
    $$
        \varphi(\rho, z) = \langle \psi \rangle \frac{1}{\sqrt{2 \pi \sigma_\rho^2}} \exp\left(-\frac{(\rho - \mu_\rho)^2}{2 \sigma_\rho^2}\right)
    $$
    """)
    ui_gaussian_mu = mo.ui.number(start=0.0, stop=2000, step=0.1, value=1100, label="$\\mu_\\rho \\; [\\frac{g}{L}]$")
    ui_gaussian_sigma = mo.ui.number(start=0.0, stop=15, step=0.1, value=4, label="$\\sigma_\\rho \\; [\\frac{g}{L}]$")

    ui_gaussian = mo.vstack([
        ui_gaussian_text,
        ui_gaussian_mu,
        ui_gaussian_sigma
    ])
    return ui_gaussian, ui_gaussian_mu, ui_gaussian_sigma


@app.cell
def _(mo):
    ui_homogeneous_text = mo.md(
    r"""
    Homogeneous $\varphi$ distribution
    $$
    \varphi(\rho, z) = \frac{\langle \psi \rangle}{L_\rho}
    $$
    where $L_\rho$ is the size of the $\rho$-dimension.
    """
    )
    ui_homogeneous = ui_homogeneous_text
    return (ui_homogeneous,)


@app.cell
def _(mo):
    ui_psi_avg = mo.ui.number(start=0.0, stop=1.0, step=0.001, value=0.02, label="$\\langle \\psi \\rangle$")
    ui_psi_avg
    return (ui_psi_avg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Pick a Distribution Type
    """)
    return


@app.cell
def _(mo, ui_gaussian, ui_homogeneous):
    ui_phi_type = mo.ui.tabs(
        {
            "Gaussian": ui_gaussian,
            "Homogeneous": ui_homogeneous
        },
        value="Force closure",
    )
    ui_phi_type
    return (ui_phi_type,)


@app.cell
def _():
    # Common parameters
    N = 256
    wing = 30

    _RC = 1100
    _RL = 30

    rho = np.linspace(_RC - _RL / 2.0, _RC + _RL / 2.0, N, dtype=np.float64)

    _dz = 0.00027
    z = np.linspace(0.0, (N-1) * _dz, N)
    return N, rho, wing, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Inspect your initial $\varphi$
    """)
    return


@app.cell(hide_code=True)
def _(mo, wing):
    mo.md(rf"""
    A wing of size {wing} is added in the $\rho$ and $z$ dimension.
    """)
    return


@app.cell
def _(
    mo,
    plt,
    rho,
    ui_gaussian_mu,
    ui_gaussian_sigma,
    ui_phi_type,
    ui_psi_avg,
    wing,
    z,
):
    if ui_phi_type.value == "Gaussian":
        phi = phi_gaussian(rho, z, ui_psi_avg.value, ui_gaussian_mu.value, ui_gaussian_sigma.value)
    else:
        phi = phi_homogeneous(rho, z, ui_psi_avg.value)

    phi_wing = renormalize_phi(phi_add_wing(phi, wing), rho, z, ui_psi_avg.value, wing)

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _im = _ax.imshow(
        phi_wing,
        origin="lower",
        aspect="auto",
        extent=[z[0], z[-1], rho[0], rho[-1]],
        cmap="viridis",
    )
    _ax.set_xlabel("$z$ [m]")
    _ax.set_ylabel(r"$\rho$ [g/L]")
    _fig.colorbar(_im, ax=_ax, label=r"$\phi$")
    _ax.set_title(r"$\varphi(\rho, z)$")
    mo.ui.matplotlib(_ax)
    return (phi_wing,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. Export your $\varphi$
    """)
    return


@app.cell
def _(Path, mo):
    last_export_path, set_last_export_path = mo.state(None)
    last_export_submission, set_last_export_submission = mo.state(None)
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
        {export_dir}

        {export_name}
        """
        )
        .batch(export_dir=export_dir, export_name=export_name)
        .form(
            submit_button_label="Export phi file",
            clear_on_submit=True,
            show_clear_button=True,
        )
    )
    export_form
    return (
        export_form,
        last_export_path,
        last_export_submission,
        set_last_export_path,
        set_last_export_submission,
    )


@app.cell
def _(
    CopyToClipboard,
    N,
    Path,
    export_form,
    h5py,
    last_export_path,
    last_export_submission,
    mo,
    phi_wing,
    rho,
    set_last_export_path,
    set_last_export_submission,
    ui_phi_type,
    ui_psi_avg,
    wing,
    z,
):
    if export_form.value is None:
        if last_export_path() is None:
            result = mo.md(
                "Submit the form to export the CUDA-compatible initial phi file."
            )
        else:
            result = mo.md(
                rf"Last export: `{last_export_path()}`. Submit again to export a new file."
            )
    else:
        # Marimo cells are reactive: this cell reruns when e.g. `phi` changes.
        # Guard against re-exporting on recomputation by only exporting once per
        # distinct form submission. Use object identity so two submissions with
        # the same values (same path/filename) are still treated as distinct.
        submission = export_form.value
        submission_token = id(submission)
        if last_export_submission() == submission_token:
            if last_export_path() is None:
                result = mo.md(
                    "Export already processed for the current form submission."
                )
            else:
                result = mo.md(
                    rf"Export already processed: `{last_export_path()}`. Submit again to export a new file."
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

                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with h5py.File(path, "w") as f:
                        group = f.create_group("phi")
                        group.create_dataset(
                            "values", data=np.asarray(phi_wing, dtype=np.float64)
                        )
                        group.create_dataset(
                            "rho", data=np.asarray(rho, dtype=np.float64)
                        )
                        group.create_dataset("z", data=np.asarray(z, dtype=np.float64))
                        group.attrs["N"] = int(N)
                        group.attrs["PSI"] = float(ui_psi_avg.value)
                        group.attrs["wingL"] = int(wing)
                        group.attrs["phi_type"] = str(ui_phi_type.value)
                        group.attrs["storage_order"] = "phi[rho_idx, z_idx]"
                        group.attrs["generated_by"] = "analysis/phi_init.py"
                        group.attrs["normalization"] = (
                            "no runtime renormalization required"
                        )
                except Exception as exc:
                    result = mo.md(rf"Export failed: `{exc}`")
                else:
                    set_last_export_path(str(path))
                    set_last_export_submission(submission_token)
                    clipboard = mo.ui.anywidget(CopyToClipboard(text_to_copy=str(path)))
                    result = mo.vstack(
                        [
                            mo.md(rf"Exported initial phi to `{path}`."),
                            mo.md(
                                r"Use it in the simulation with `--phi-file <path>`."
                            ),
                            mo.hstack(
                                [mo.md("Copy exported path:"), clipboard],
                                justify="start",
                            ),
                        ]
                    )
    result
    return


@app.cell
def _():
    # n = mo.ui.number(start=8, stop=4096, step=1, value=256, label="N")
    # psi = mo.ui.number(start=1e-6, stop=0.999999, step=1e-3, value=0.02, label="PSI")
    # init_mode = mo.ui.radio(
        # options=["Gaussian (legacy)", "Constant"],
        # value="Gaussian (legacy)",
        # label="Initializer",
    # )
    # rho_center = mo.ui.number(
        # start=1000.0, stop=1200.0, step=0.1, value=1100.0, label="Rmu"
    # )
    # rho_sigma = mo.ui.number(
        # start=1e-3, stop=100.0, step=0.1, value=4.0, label="Rsigma"
    # )
    # wing = mo.ui.number(start=0, stop=1024, step=1, value=30, label="wingL")
    # rho_center_axis = mo.ui.number(
        # start=1000.0, stop=1200.0, step=0.1, value=1100.0, label="RC"
    # )
    # rho_span = mo.ui.number(start=1.0, stop=200.0, step=0.5, value=30.0, label="RL")
    # 
    # ui = mo.md(
        # """
        # ## Parameters
    # 
        # {n}
    # 
        # {psi}
    # 
        # {init_mode}
    # 
        # {rho_center}
    # 
        # {rho_sigma}
    # 
        # {wing}
    # 
        # {rho_center_axis}
    # 
        # {rho_span}
        # """
    # ).batch(
        # n=n,
        # psi=psi,
        # init_mode=init_mode,
        # rho_center=rho_center,
        # rho_sigma=rho_sigma,
        # wing=wing,
        # rho_center_axis=rho_center_axis,
        # rho_span=rho_span,
    # )
    # 
    # ui
    return


@app.cell
def _():
    # values = ui.value or {}
    # 
    # N = int(values["n"])
    # PSI = float(values["psi"])
    # init_mode_value = str(values["init_mode"])
    # Rmu = float(values["rho_center"])
    # Rsigma = float(values["rho_sigma"])
    # wingL = int(values["wing"])
    # RC = float(values["rho_center_axis"])
    # RL = float(values["rho_span"])
    # 
    # rho = np.linspace(RC - RL / 2.0, RC + RL / 2.0, N, dtype=np.float64)
    # rho_idx = np.arange(N, dtype=np.int32)
    # z_idx = np.arange(N, dtype=np.int32)
    # 
    # if init_mode_value == "Constant":
        # # Constant in (rho, z) before masking and normalization.
        # phi = np.ones((N, N), dtype=np.float64)
    # else:
        # radial_profile = np.exp(-((rho - Rmu) ** 2) / (2.0 * Rsigma**2))
        # phi = np.repeat(radial_profile[:, None], N, axis=1)
    # 
    # if wingL > 0:
        # z_mask = (z_idx < wingL + 2) | (z_idx > (N - 1 - (wingL + 2)))
        # rho_mask = (rho_idx < wingL) | (rho_idx > (N - 1 - wingL))
        # phi[:, z_mask] = 0.0
        # phi[rho_mask, :] = 0.0
    # 
    # phi_sum = float(phi.sum())
    # if phi_sum > 0.0:
        # phi = phi / phi_sum * PSI * (N - 2 * (wingL + 2))
    # 
    # psi_profile = phi.sum(axis=0)
    return


@app.cell
def _():
    # def _():
        # z_extent = [0, N - 1]
        # rho_extent = [float(rho[0]), float(rho[-1])]
    # 
        # fig, ax = plt.subplots(figsize=(8, 5))
        # im = ax.imshow(
            # phi,
            # origin="lower",
            # aspect="auto",
            # extent=[z_extent[0], z_extent[1], rho_extent[0], rho_extent[1]],
            # cmap="magma",
        # )
        # ax.set_xlabel("z index")
        # ax.set_ylabel(r"$\rho$ [g/L]")
        # ax.set_title(f"Initial phi field ({init_mode_value})")
        # fig.colorbar(im, ax=ax, label=r"$\phi$")
        # return mo.ui.matplotlib(ax)
    # 
    # _()
    return


@app.cell
def _():
    # fig, ax = plt.subplots(figsize=(8, 3.5))
    # ax.plot(psi_profile, color="tab:blue", linewidth=2)
    # ax.set_xlabel("z index")
    # ax.set_ylabel(r"$\sum_\rho \phi$")
    # ax.set_title("Derived psi profile")
    # ax.grid(True, linestyle=":", alpha=0.6)
    # mo.ui.matplotlib(ax)
    return


if __name__ == "__main__":
    app.run()
