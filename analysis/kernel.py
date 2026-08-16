# /// script
# dependencies = [
#     "h5py==3.16.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "pydantic==2.13.4",
#     "scipy==1.17.1",
#     "wigglystuff==0.3.3",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App()

with app.setup:
    import sys
    from dataclasses import replace
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    NOTEBOOK_FILE = (
        Path(__file__).resolve()
        if "__file__" in globals()
        else (Path.cwd() / "analysis" / "kernel.py").resolve()
    )
    ANALYSIS_DIR = NOTEBOOK_FILE.parent
    if str(ANALYSIS_DIR) not in sys.path:
        sys.path.insert(0, str(ANALYSIS_DIR))

    from red_patterns.kernel import (
        compute_kernel,
        kernel_config_from_ui,
        kernel_ui_layout,
        latex_scientific,
        effective_morse_potential,
        lj_potential,
        morse_potential,
        make_kernel_ui,
        plot_kernel,
        plot_pair_distribution,
        run_export_cli,
        write_kernel_h5,
    )


@app.cell
def _():
    # Script-mode CLI: `uv run analysis/kernel.py export --output ... --closure ...`
    if mo.app_meta().mode == "script" and sys.argv[1:2] == ["export"]:
        raise SystemExit(run_export_cli(sys.argv[2:], prog="kernel.py export"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Kernel Generation

    ## 0. Theory

    This notebook constructs the effective one-dimensional interaction kernel
    $K$ that appears in the reduced DDFT model.

    $$
        \partial_t \varphi(\rho,z,t) + \partial_z J_z(\rho,z,t) = 0
    $$

    with flux

    $$
        J_z(\rho,z,t) = \Gamma\,\varphi(\rho,z,t) \left( \frac{2 \pi}{V} \int_0^L \psi(z',t)\,K(z-z')\,d z' - \partial_z u_{\mathrm{ext}}(\rho,z,t) \right).
    $$

    The total volume fraction is

    $$
        \psi(z,t) := \int_M \varphi(\rho,z,t)\, d\rho.
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The reduced model admits two common closures:

    1. Potential closure

    $$
        K(x) = -x\,g(|x|)\,u(|x|)
    $$

    2. Force closure
        $$
            K(x) = x\int_{|x|}^{\infty} g(R)\,f(R)\,d R,
        $$
        where for conservative forces $f(R) = -u'(R)$.

    The pair distribution function $g$ models correlations between cells. The
    closure then determines how $u$, $g$, and the pair force enter the kernel.

    In the mean-field case $g=1$, both closures coincide. Otherwise they can
    produce visibly different kernels.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This notebook also calculates the coefficients used in the taylor approximation

    ### 0.1 Taylor Approximation

    We approximate

    $$
        I(z,t) = \frac{2 \pi}{V} \int_{0}^{L} d z' \, \psi(z', t) K(z - z')
    $$

    by expanding $\psi(z',t)$ around $z$ and introducing $\eta = z-z'$. The
    odd-kernel symmetry removes all even orders on a symmetric domain, giving

    $$
        \frac{V}{2 \pi} \cdot I(z,t) \approx -\nu \, \partial_z \psi(z,t) - \mu \, \partial_z^3 \psi(z,t).
    $$

    with moments

    $$
    \begin{aligned}
        \nu &= \int_{-\infty}^{+\infty} d\eta \, K(\eta)\eta, \\
        \mu &= \frac{1}{3!}\int_{-\infty}^{+\infty} d\eta \, K(\eta)\eta^3.
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Configure the kernel

    All kernel controls live in a single `mo.ui.dictionary` built by
    `make_kernel_ui()`. Choose **Original** for the Lennard-Jones closure/PDF
    construction, or **HNC effective Morse** for the direct kernel

    $$K(x)=x\left[a\left(e^{-u_M(|x|)/b}-1\right)-c\,u_M(|x|)\right].$$
    """)
    return


@app.cell
def cell_kernel_ui():
    kernel_ui = make_kernel_ui()
    return (kernel_ui,)


@app.cell
def cell_kernel_ui_display(kernel_ui):
    kernel_ui_layout(kernel_ui)
    return


@app.cell
def cell_kernel_cfg(kernel_ui):
    kernel_cfg = kernel_config_from_ui(kernel_ui.value)
    kernel_result = compute_kernel(kernel_cfg)
    return kernel_cfg, kernel_result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Inspect the source potential, effective potential, and kernel
    """)
    return


@app.cell
def _(kernel_cfg):
    if kernel_cfg.kernel_type.value == "hnc":
        _r = np.linspace(0.01 * kernel_cfg.beta, 3 * kernel_cfg.beta, 500)
        _u = morse_potential(_r, kernel_cfg.alpha, kernel_cfg.beta, kernel_cfg.gamma)
        _label = "Morse"
        _minimum = kernel_cfg.beta
    else:
        _r = np.linspace(0.95 * kernel_cfg.sigma, 3 * kernel_cfg.sigma, 500)
        _u = lj_potential(_r, kernel_cfg.U, kernel_cfg.sigma)
        _label = "Lennard-Jones"
        _minimum = kernel_cfg.sigma

    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.plot(_r * 1e6, _u * 1e18, color="blue", linewidth=2, label=_label)
    _ax.axhline(0, color="black", linewidth=1)
    _ax.axvline(
        _minimum * 1e6,
        color="red",
        linestyle="--",
        label=rf"$r_{{min}} = {_minimum * 1e6:.3g}\,\mu m$",
    )
    _ax.set_xlabel(r"Distance $r$ ($\mu$m)", fontsize=12)
    _ax.set_ylabel(r"Potential $u(r)$ ($10^{-18}$ J)", fontsize=12)
    _ax.set_title(f"{_label} potential", fontsize=14)
    _ax.set_ylim(-150, 100)
    _ax.grid(True, linestyle=":", alpha=0.7)
    _ax.legend()
    _fig
    return


@app.cell
def _(kernel_cfg):
    if kernel_cfg.kernel_type.value == "hnc":
        _r = np.linspace(0.01 * kernel_cfg.beta, 3 * kernel_cfg.beta, 500)
        _u_eff = effective_morse_potential(
            _r,
            kernel_cfg.a,
            kernel_cfg.b,
            kernel_cfg.c,
            kernel_cfg.alpha,
            kernel_cfg.beta,
            kernel_cfg.gamma,
        )
        _fig, _ax = plt.subplots(figsize=(8, 6))
        _ax.plot(_r * 1e6, _u_eff * 1e18, color="green", linewidth=2)
        _ax.axhline(0, color="black", linewidth=1)
        _ax.set(xlabel=r"Distance $r$ ($\mu$m)", ylabel=r"$u_{\mathrm{eff}}(r)$ ($10^{-18}$ J)", title="HNC effective potential")
        _ax.grid(True, linestyle=":", alpha=0.7)
        _fig
    return


@app.cell
def _(kernel_cfg):
    if kernel_cfg.kernel_type.value == "original":
        plot_pair_distribution(kernel_cfg)
    return


@app.cell
def _(kernel_cfg, kernel_result):
    plot_kernel(kernel_result, kernel_cfg)
    return


@app.cell(hide_code=True)
def _(kernel_result):
    mo.md(rf"""
    ### Numerical Taylor coefficients

    $$
    \begin{{aligned}}
    \nu &= {latex_scientific(kernel_result.nu)}, \\
    \mu &= {latex_scientific(kernel_result.mu)}.
    \end{{aligned}}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. CUDA Export

    Writes the discrete convolution stencil expected by the CUDA code, built from
    the live parameters above.
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
    export_name = mo.ui.text(value="kernel.h5", label="File name")
    export_form = (
        mo.md(
            """
    Export a CUDA-compatible convolution kernel.

    {export_dir}

    {export_name}
    """
        )
        .batch(export_dir=export_dir, export_name=export_name)
        .form(
            submit_button_label="Export CUDA kernel",
            clear_on_submit=False,
            show_clear_button=True,
        )
    )
    export_form
    return (export_form,)


@app.cell
def _(export_form, kernel_cfg, kernel_result):
    if export_form.value is None:
        _result = mo.md("Submit the form to export a CUDA-compatible kernel.")
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
                _written = write_kernel_h5(
                    _path, kernel_result, replace(kernel_cfg, output_path=_path)
                )
            except Exception as _exc:
                _result = mo.md(rf"Export failed: `{_exc}`")
            else:
                _result = mo.vstack(
                    [
                        mo.md(rf"Exported CUDA-compatible kernel to `{_written}`."),
                        mo.md(
                            rf"`kernelN={kernel_cfg.kernel_n}`, "
                            rf"`spacing={kernel_result.fine_dz:.6e}` m, "
                            rf"`closure={kernel_cfg.closure}`, "
                            rf"`pair_distribution={kernel_cfg.pair_distribution}`, "
                            rf"`U={kernel_cfg.U:.6e}` J."
                        ),
                    ]
                )
    _result
    return


if __name__ == "__main__":
    app.run()
