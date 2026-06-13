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
    import sys
    from dataclasses import replace
    from pathlib import Path

    import marimo as mo

    NOTEBOOK_FILE = (
        Path(__file__).resolve()
        if "__file__" in globals()
        else (Path.cwd() / "analysis" / "phi_init.py").resolve()
    )
    ANALYSIS_DIR = NOTEBOOK_FILE.parent
    if str(ANALYSIS_DIR) not in sys.path:
        sys.path.insert(0, str(ANALYSIS_DIR))

    from red_patterns.phi import (
        compute_phi,
        make_phi_ui,
        phi_config_from_ui,
        phi_ui_layout,
        plot_phi,
        run_export_cli,
        write_phi_h5,
    )


@app.cell
def _():
    # Script-mode CLI: `uv run analysis/phi_init.py export --output ... --phi-type ...`
    if mo.app_meta().mode == "script" and sys.argv[1:2] == ["export"]:
        raise SystemExit(run_export_cli(sys.argv[2:], prog="phi_init.py export"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Initial Phi

    The exported file stores the field on `/phi/values` with shape `(N, N)` and
    storage order `phi[rho_idx, z_idx]`. Each selectable $\varphi$ distribution
    satisfies the normalization condition below.
    """)
    return


@app.cell
def _():
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
    	\psi(z,t = 0) = \psi(z) := \int_I \varphi(\rho,z)d\rho\,.
    $$

    of the specific volume fraction, then the normalization with respect to the Average Volume Fraction is

    $$
    	\langle \psi \rangle := \frac{1}{\int_J 1 d z} \int_J  \psi(z) d z = \frac{1}{L_z} \int_J \psi(z) d z
    $$
    In the Code the continuous integral gets approximated by a Riemann sum on a uniform grid, giving

    $$
    \begin{align*}
    	\langle \psi \rangle &= \frac{1}{L_z} \int_J \psi(z) d z \\
        &\approx \frac{1}{N \cdot \Delta z} \sum_j \psi_j \Delta z \\
        &\approx \frac{1}{N} \sum_j \sum_i \phi_{i,j} \Delta \rho \\
        &= \frac{1}{N} \sum_j \sum_i \mathrm{phi[i][j]} \\
    \end{align*}
    $$

    because `phi[i][j]` $:= \phi(\rho_i, z_j) \cdot \Delta \rho$ in the Simulation.
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

    **Discrete implementation.** The CUDA simulation (`initPhi` in `simulations.cu`) uses a discrete, index-space normalization with no physical step-size factors. With $\psi_j := \sum_{i} \varphi_{ij}$,

    $$
        \langle \psi \rangle := \frac{1}{N - 2\,w_z} \sum_{j=w_z}^{N-1-w_z} \psi_j = \texttt{PSI}
        \quad\Longleftrightarrow\quad
        \sum_{i,j} \varphi_{ij} = \texttt{PSI} \cdot (N - 2\,w_z).
    $$

    `renormalize_phi` rescales $\varphi$ uniformly by $\texttt{PSI} / \langle \psi \rangle$ to satisfy this.
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generate a $\varphi$ distribution!

    All controls live in a single `mo.ui.dictionary` built by `make_phi_ui()`.
    """)
    return


@app.cell
def cell_phi_ui():
    phi_ui = make_phi_ui()
    return (phi_ui,)


@app.cell
def cell_phi_ui_display(phi_ui):
    phi_ui_layout(phi_ui)
    return


@app.cell
def cell_phi_cfg(phi_ui):
    phi_cfg = phi_config_from_ui(phi_ui.value)
    phi_result = compute_phi(phi_cfg)
    return phi_cfg, phi_result


@app.cell(hide_code=True)
def _(phi_cfg):
    mo.md(rf"""
    ### Inspect your initial $\varphi$

    A wing of size {phi_cfg.wing} is added in the $\rho$ and $z$ dimension.
    """)
    return


@app.cell
def _(phi_result):
    plot_phi(phi_result)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Export your $\varphi$
    """)
    return


@app.cell
def cell_export_form():
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
            clear_on_submit=False,
            show_clear_button=True,
        )
    )
    export_form
    return (export_form,)


@app.cell
def _(export_form, phi_cfg, phi_result):
    if export_form.value is None:
        _result = mo.md("Submit the form to export the initial phi file.")
    else:
        _dir_entries = export_form.value.get("export_dir") or []
        _file_name = str(export_form.value.get("export_name", "")).strip()
        if not _dir_entries:
            _result = mo.md("Please select a directory before exporting.")
        elif not _file_name:
            _result = mo.md("Please enter a file name before exporting.")
        else:
            _path = Path(_dir_entries[0].path) / _file_name
            try:
                _written = write_phi_h5(
                    _path, phi_result, replace(phi_cfg, output_path=_path)
                )
            except Exception as _exc:
                _result = mo.md(rf"Export failed: `{_exc}`")
            else:
                _result = mo.vstack(
                    [
                        mo.md(rf"Exported initial phi to `{_written}`."),
                        mo.md(r"Use it in the simulation with `--phi-file <path>`."),
                    ]
                )
    _result
    return


if __name__ == "__main__":
    app.run()
