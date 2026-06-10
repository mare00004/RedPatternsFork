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

__generated_with = "0.23.9"
app = marimo.App(width="medium")

with app.setup:
    import argparse
    from dataclasses import dataclass, replace
    from pathlib import Path

    import h5py
    import numpy as np
    from red_patterns import Array1F, Array2F

    DEFAULT_N = 256
    DEFAULT_WING = 30
    DEFAULT_RHO_CENTER = 1100.0
    DEFAULT_RHO_SPAN = 30.0
    DEFAULT_DZ = 0.000267651
    DEFAULT_PSI_AVG = 0.02
    DEFAULT_GAUSSIAN_MU = 1100.0
    DEFAULT_GAUSSIAN_SIGMA = 4.0

    GAUSSIAN_PHI = "Gaussian"
    HOMOGENEOUS_PHI = "Homogeneous"

    CLI_TO_PHI_TYPE_LABEL = {
        "gaussian": GAUSSIAN_PHI,
        "homogeneous": HOMOGENEOUS_PHI,
    }
    PHI_TYPE_LABEL_TO_CLI = {
        value: key for key, value in CLI_TO_PHI_TYPE_LABEL.items()
    }


@app.cell
def _():
    @dataclass(frozen=True)
    class PhiExportConfig:
        output_path: "Path"
        phi_type: str
        psi_avg: float
        N: int
        wing: int
        rho_center: float
        rho_span: float
        dz: float
        gaussian_mu: float | None = None
        gaussian_sigma: float | None = None

    def normalize_phi_type_name(value: str) -> str:
        if value in CLI_TO_PHI_TYPE_LABEL:
            return value
        if value in PHI_TYPE_LABEL_TO_CLI:
            return PHI_TYPE_LABEL_TO_CLI[value]
        raise ValueError(f"Unknown phi type: {value!r}")

    def phi_type_label(value: str) -> str:
        return CLI_TO_PHI_TYPE_LABEL[normalize_phi_type_name(value)]

    def build_export_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog=f"{Path(__file__).name} export",
            description="Export a CUDA-compatible initial phi field to HDF5.",
        )
        parser.add_argument(
            "--output",
            required=True,
            help="Path to the output HDF5 file.",
        )
        parser.add_argument(
            "--phi-type",
            required=True,
            choices=sorted(CLI_TO_PHI_TYPE_LABEL.keys()),
            help="Initial phi distribution to use.",
        )
        parser.add_argument(
            "--psi-avg",
            required=True,
            type=float,
            help="Average volume fraction.",
        )
        parser.add_argument(
            "--N",
            type=int,
            default=DEFAULT_N,
            help="Grid size in rho and z.",
        )
        parser.add_argument(
            "--wing",
            type=int,
            default=DEFAULT_WING,
            help="Wing size used by the CUDA initialization.",
        )
        parser.add_argument(
            "--rho-center",
            type=float,
            default=DEFAULT_RHO_CENTER,
            help="Center of the rho axis in g/L.",
        )
        parser.add_argument(
            "--rho-span",
            type=float,
            default=DEFAULT_RHO_SPAN,
            help="Total rho-axis span in g/L.",
        )
        parser.add_argument(
            "--dz",
            type=float,
            default=DEFAULT_DZ,
            help="Z-axis spacing in meters.",
        )
        parser.add_argument(
            "--gaussian-mu",
            type=float,
            help="Gaussian rho center in g/L.",
        )
        parser.add_argument(
            "--gaussian-sigma",
            type=float,
            help="Gaussian rho width in g/L.",
        )
        return parser

    def validate_export_namespace(
        parser: argparse.ArgumentParser, args: argparse.Namespace
    ) -> PhiExportConfig:
        errors: list[str] = []

        if args.N < 3:
            errors.append("--N must be an integer >= 3.")
        if args.wing < 0:
            errors.append("--wing must be non-negative.")
        if args.psi_avg < 0.0:
            errors.append("--psi-avg must be non-negative.")
        if args.rho_span <= 0.0:
            errors.append("--rho-span must be positive.")
        if args.dz <= 0.0:
            errors.append("--dz must be positive.")

        active_rho = args.N - 2 * args.wing
        active_z = args.N - 2 * (args.wing + 2)
        if active_rho <= 0:
            errors.append(
                "--wing is too large for --N: the active rho region would be empty."
            )
        if active_z <= 0:
            errors.append(
                "--wing is too large for --N: the active z region would be empty."
            )

        if args.phi_type == "gaussian":
            if args.gaussian_mu is None:
                errors.append("--gaussian-mu is required with --phi-type=gaussian.")
            if args.gaussian_sigma is None:
                errors.append(
                    "--gaussian-sigma is required with --phi-type=gaussian."
                )
            elif args.gaussian_sigma <= 0.0:
                errors.append("--gaussian-sigma must be positive.")
        else:
            if args.gaussian_mu is not None or args.gaussian_sigma is not None:
                errors.append(
                    "--gaussian-mu and --gaussian-sigma are only valid with "
                    "--phi-type=gaussian."
                )

        if errors:
            parser.error("\n".join(errors))

        return PhiExportConfig(
            output_path=Path(args.output),
            phi_type=args.phi_type,
            psi_avg=args.psi_avg,
            N=args.N,
            wing=args.wing,
            rho_center=args.rho_center,
            rho_span=args.rho_span,
            dz=args.dz,
            gaussian_mu=args.gaussian_mu,
            gaussian_sigma=args.gaussian_sigma,
        )

    return (
        PhiExportConfig,
        build_export_parser,
        normalize_phi_type_name,
        phi_type_label,
        validate_export_namespace,
    )


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
def phi_gaussian(
    rho: Array1F,
    z: Array1F,
    psi_avg: np.floating,
    mu: np.floating,
    sigma: np.floating,
) -> Array2F:
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
    radial_profile = psi_avg * (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(
        -((rho - mu) ** 2) / (2.0 * sigma**2)
    )
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


@app.cell
def _(
    PhiExportConfig,
    build_export_parser,
    phi_type_label,
    validate_export_namespace,
):
    def build_phi_axes(
        *, N: int, rho_center: float, rho_span: float, dz: float
    ) -> tuple[np.ndarray, np.ndarray]:
        rho = np.linspace(
            rho_center - rho_span / 2.0,
            rho_center + rho_span / 2.0,
            N,
            dtype=np.float64,
        )
        z = np.linspace(0.0, (N - 1) * dz, N, dtype=np.float64)
        return rho, z

    def compute_phi_field_data(
        cfg: PhiExportConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho, z = build_phi_axes(
            N=cfg.N,
            rho_center=cfg.rho_center,
            rho_span=cfg.rho_span,
            dz=cfg.dz,
        )

        if cfg.phi_type == "gaussian":
            assert cfg.gaussian_mu is not None
            assert cfg.gaussian_sigma is not None
            phi = phi_gaussian(
                rho,
                z,
                cfg.psi_avg,
                cfg.gaussian_mu,
                cfg.gaussian_sigma,
            )
        else:
            phi = phi_homogeneous(rho, z, cfg.psi_avg)

        phi_wing = renormalize_phi(phi_add_wing(phi, cfg.wing), rho, z, cfg.psi_avg, cfg.wing)
        return rho, z, np.asarray(phi_wing, dtype=np.float64)

    def write_phi_h5(
        output_path: str | Path,
        *,
        phi_values: np.ndarray,
        rho: np.ndarray,
        z: np.ndarray,
        N: int,
        psi_avg: float,
        wing: int,
        phi_type: str,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as f:
            group = f.create_group("phi")
            group.create_dataset("values", data=np.asarray(phi_values, dtype=np.float64))
            group.create_dataset("rho", data=np.asarray(rho, dtype=np.float64))
            group.create_dataset("z", data=np.asarray(z, dtype=np.float64))
            group.attrs["N"] = int(N)
            group.attrs["PSI"] = float(psi_avg)
            group.attrs["wingL"] = int(wing)
            group.attrs["phi_type"] = phi_type_label(phi_type)
            group.attrs["storage_order"] = "phi[rho_idx, z_idx]"
            group.attrs["generated_by"] = "analysis/phi_init.py"
            group.attrs["normalization"] = "no runtime renormalization required"

        return output_path

    def export_phi_file(cfg: PhiExportConfig):
        rho, z, phi_wing = compute_phi_field_data(cfg)
        output_path = write_phi_h5(
            cfg.output_path,
            phi_values=phi_wing,
            rho=rho,
            z=z,
            N=cfg.N,
            psi_avg=cfg.psi_avg,
            wing=cfg.wing,
            phi_type=cfg.phi_type,
        )
        return output_path, rho, z, phi_wing

    def run_export_cli(argv: list[str]) -> int:
        parser = build_export_parser()
        args = parser.parse_args(argv)
        cfg = validate_export_namespace(parser, args)
        output_path, _, _, phi_wing = export_phi_file(cfg)

        print(f"Exported initial phi to {output_path}")
        print(
            f"Stored shape {phi_wing.shape} on /phi/values with storage order "
            "phi[rho_idx, z_idx]"
        )
        summary = [
            f"phi_type={cfg.phi_type}",
            f"PSI={cfg.psi_avg:.6e}",
            f"N={cfg.N}",
            f"wing={cfg.wing}",
            f"rho_center={cfg.rho_center:.6e}",
            f"rho_span={cfg.rho_span:.6e}",
            f"DZ={cfg.dz:.6e}",
        ]
        if cfg.phi_type == "gaussian":
            assert cfg.gaussian_mu is not None
            assert cfg.gaussian_sigma is not None
            summary.extend(
                [
                    f"gaussian_mu={cfg.gaussian_mu:.6e}",
                    f"gaussian_sigma={cfg.gaussian_sigma:.6e}",
                ]
            )
        print("Used parameters: " + ", ".join(summary))
        return 0

    return (
        build_phi_axes,
        compute_phi_field_data,
        export_phi_file,
        run_export_cli,
    )


@app.cell
def _(run_export_cli):
    import marimo as _mo
    import sys

    if _mo.app_meta().mode == "script" and sys.argv[1:2] == ["export"]:
        raise SystemExit(run_export_cli(sys.argv[2:]))
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    return mo, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initial Phi

    The exported file stores the field on `/phi/values` with shape `(N, N)` and
    storage order `phi[rho_idx, z_idx]`.
    """)
    return


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
    ui_gaussian_text = mo.md(
        r"""
    Gaussian $\varphi$ distribution
    $$
        \varphi(\rho, z) = \langle \psi \rangle \frac{1}{\sqrt{2 \pi \sigma_\rho^2}} \exp\left(-\frac{(\rho - \mu_\rho)^2}{2 \sigma_\rho^2}\right)
    $$
    """
    )
    ui_gaussian_mu = mo.ui.number(
        start=0.0,
        stop=2000,
        step=0.1,
        value=DEFAULT_GAUSSIAN_MU,
        label="$\\mu_\\rho \\; [\\frac{g}{L}]$",
    )
    ui_gaussian_sigma = mo.ui.number(
        start=0.0,
        stop=15,
        step=0.1,
        value=DEFAULT_GAUSSIAN_SIGMA,
        label="$\\sigma_\\rho \\; [\\frac{g}{L}]$",
    )

    ui_gaussian = mo.vstack(
        [
            ui_gaussian_text,
            ui_gaussian_mu,
            ui_gaussian_sigma,
        ]
    )
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
    ui_psi_avg = mo.ui.number(
        start=0.0,
        stop=1.0,
        step=0.001,
        value=DEFAULT_PSI_AVG,
        label="$\\langle \\psi \\rangle$",
    )
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
            GAUSSIAN_PHI: ui_gaussian,
            HOMOGENEOUS_PHI: ui_homogeneous,
        },
        value=GAUSSIAN_PHI,
    )
    ui_phi_type
    return (ui_phi_type,)


@app.cell
def _(build_phi_axes):
    N = DEFAULT_N
    wing = DEFAULT_WING
    rho, z = build_phi_axes(
        N=N,
        rho_center=DEFAULT_RHO_CENTER,
        rho_span=DEFAULT_RHO_SPAN,
        dz=DEFAULT_DZ,
    )
    return rho, wing, z


@app.cell
def _(
    PhiExportConfig,
    normalize_phi_type_name,
    ui_gaussian_mu,
    ui_gaussian_sigma,
    ui_phi_type,
    ui_psi_avg,
):
    phi_type = normalize_phi_type_name(ui_phi_type.value)
    live_export_cfg = PhiExportConfig(
        output_path=Path("initial_phi.h5"),
        phi_type=phi_type,
        psi_avg=float(ui_psi_avg.value),
        N=DEFAULT_N,
        wing=DEFAULT_WING,
        rho_center=DEFAULT_RHO_CENTER,
        rho_span=DEFAULT_RHO_SPAN,
        dz=DEFAULT_DZ,
        gaussian_mu=float(ui_gaussian_mu.value) if phi_type == "gaussian" else None,
        gaussian_sigma=(
            float(ui_gaussian_sigma.value) if phi_type == "gaussian" else None
        ),
    )
    return (live_export_cfg,)


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
def _(compute_phi_field_data, live_export_cfg, mo, plt, rho, z):
    _, _, phi_wing = compute_phi_field_data(live_export_cfg)

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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. Export your $\varphi$
    """)
    return


@app.cell
def _(mo):
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
def _():
    try:
        from wigglystuff import CopyToClipboard
    except ImportError:
        CopyToClipboard = None
    return (CopyToClipboard,)


@app.cell
def _(
    CopyToClipboard,
    export_form,
    export_phi_file,
    last_export_path,
    last_export_submission,
    live_export_cfg,
    mo,
    set_last_export_path,
    set_last_export_submission,
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
                export_cfg = replace(live_export_cfg, output_path=path)

                try:
                    output_path, _, _, _ = export_phi_file(export_cfg)
                except Exception as exc:
                    result = mo.md(rf"Export failed: `{exc}`")
                else:
                    set_last_export_path(str(output_path))
                    set_last_export_submission(submission_token)
                    items = [
                        mo.md(rf"Exported initial phi to `{output_path}`."),
                        mo.md(r"Use it in the simulation with `--phi-file <path>`."),
                    ]
                    if CopyToClipboard is not None:
                        clipboard = mo.ui.anywidget(
                            CopyToClipboard(text_to_copy=str(output_path))
                        )
                        items.append(
                            mo.hstack(
                                [mo.md("Copy exported path:"), clipboard],
                                justify="start",
                            )
                        )
                    result = mo.vstack(items)
    result
    return


if __name__ == "__main__":
    app.run()
