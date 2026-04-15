# /// script
# dependencies = [
#     "h5py==3.16.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "scipy==1.17.1",
#     "wigglystuff==0.3.3",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import h5py
    import marimo as mo
    import numpy as np
    from pathlib import Path
    from scipy.integrate import cumulative_trapezoid
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    from typing import Callable, ClassVar, Dict
    from wigglystuff import CopyToClipboard

    return (
        Callable,
        ClassVar,
        CopyToClipboard,
        Dict,
        Path,
        cumulative_trapezoid,
        dataclass,
        h5py,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kernel

    This notebook constructs the effective one-dimensional interaction kernel
    $K$ that appears in the reduced DDFT model.

    $$
        \partial_t \varphi(\rho,z,t) + \partial_z J_z(\rho,z,t) = 0
    $$

    with flux

    $$
        J_z(\rho,z,t) = \Gamma\,\varphi(\rho,z,t) \left( \frac{1}{V} \int_0^L \psi(z',t)\,K(z-z')\,d z' - \partial_z u_{\mathrm{ext}}(\rho,z,t) \right).
    $$

    The total volume fraction is

    $$
        \psi(z,t) := \int_M \varphi(\rho,z,t)\, d\rho.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The reduced model admits two common closures:

    1. Potential closure

    $$
        K(x) = 2\pi\,x\,g(|x|)\,u(|x|)
    $$

    2. Force closure
        $$
            K(x) = 2\pi\,x\int_{|x|}^{\infty} g(R)\,f(R)\,d R,
        $$
        where for conservative forces $f(R) = -u'(R)$.

    In the mean-field case $g=1$, both closures coincide. Otherwise they can
    produce visibly different kernels.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pair Potential

    For now we use a Lennard-Jones pair potential

    $$
        u(r) = 4U \left(\frac{\sigma^{12}}{r^{12}} - \frac{\sigma^6}{r^6}\right),
    $$

    with $\sigma = 5.6\,\mu\mathrm{m}$ and $U = 111.15 \times 10^{-18}\,\mathrm{J}$.

    The pair distribution function $g$ models correlations between cells. The
    closure then determines how $u$, $g$, and the pair force enter the kernel.
    """)
    return


@app.cell
def _(np):
    SIGMA = 5.6e-6  # 5.6 micrometers converted to meters
    U = 111.15e-18  # 111.15 * 10^-18 Joules
    V = 90e-18  # 90 fL in m^3

    def lennard_jones_potential(r: np.ndarray):
        return 4 * U * ((SIGMA / r) ** 12 - (SIGMA / r) ** 6)

    return SIGMA, U, V, lennard_jones_potential


@app.cell
def _(SIGMA, lennard_jones_potential, mo, np, plt):
    _r = np.linspace(0.95 * SIGMA, 3 * SIGMA, 500)
    u = lennard_jones_potential(_r)

    # 4. Convert units for cleaner axis labels
    # Convert r to micrometers (um) and u to 10^-18 Joules (aJ)
    r_um = _r * 1e6
    u_aJ = u * 1e18

    # 5. Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(r_um, u_aJ, color="blue", linewidth=2, label="Lennard-Jones potential")

    # Add a horizontal line at y=0 and vertical line at r=sigma
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(SIGMA * 1e6, color="red", linestyle="--", label=r"$\sigma = 5.6 \mu m$")

    # 6. Add labels, title, and limits
    plt.xlabel(r"Distance $r$ ($\mu$m)", fontsize=12)
    plt.ylabel(r"Potential $u(r)$ ($10^{-18}$ J)", fontsize=12)
    plt.title("Lennard-Jones potential", fontsize=14)

    # Restrict the y-axis so the steep curve doesn't squash the potential well
    plt.ylim(-150, 100)

    # 7. Add grid and legend
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.axhline(0, color="white", linewidth=1, linestyle="-")

    ax = mo.ui.matplotlib(plt.gca())
    ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pair Distribution Function

    The pair distribution function $g(r)$ captures how strongly nearby RBCs are
    correlated. Use the tabs to choose one of the available approximations.
    """)
    return


@app.cell
def _(Callable, ClassVar, Dict, U, dataclass, lennard_jones_potential, mo, np):
    @dataclass
    class PairDistributionObject:
        key: str
        markdown: mo.Html
        func: Callable[[np.ndarray], np.ndarray]
        registry: ClassVar[Dict[str, "PairDistributionObject"]] = {}

        def __post_init__(self):
            self.registry[self.key] = self

    # Pair Distribution Objects
    _ = PairDistributionObject(
        key="Mean Field",
        markdown=mo.md(
            r"""
    The mean-field approximation assumes no positional correlations.

    $$
        g(x) = 1
    $$
    """
        ),
        func=lambda x: np.ones_like(x),
    )

    G0 = 4.0e7
    SIGMA_C = 0.5e-6
    EQ_DIST = 6.585467201064237091254725819933213415424688719213008880615234375e-06

    _ = PairDistributionObject(
        key="Nearest Neighbor",
        markdown=mo.md(
            r"""
    This approximation concentrates the weight around one preferred RBC spacing.

    $$
        g(x) = g_0 \exp\left(-\frac{(r-d)^2}{2\sigma_C^2}\right)
    $$
    """
        ),
        func=lambda x: G0 * np.exp(-((x - EQ_DIST) ** 2) / (2 * (SIGMA_C**2))),
    )

    LAMBDA = 1

    _ = PairDistributionObject(
        key="Exponential",
        markdown=mo.md(
            r"""
    This ansatz reuses the pair potential itself to suppress strongly repulsive configurations.

    $$
        g(x) = \exp \left( -\lambda \frac{u(x)}{U} \right)
    $$
    """
        ),
        func=lambda x: np.exp(-LAMBDA * (lennard_jones_potential(x) / U)),
    )

    # TODO: Makr Custom PDF?
    return (PairDistributionObject,)


@app.cell
def _(PairDistributionObject, mo):
    pair_distribution_tabs = mo.ui.tabs(
        {key: node.markdown for key, node in PairDistributionObject.registry.items()},
        value="Nearest Neighbor",
    )

    pair_distribution_tabs
    return (pair_distribution_tabs,)


@app.cell
def _(PairDistributionObject, mo, np, pair_distribution_tabs, plt):
    active_pair_distribution = PairDistributionObject.registry[
        pair_distribution_tabs.value
    ]

    r_plot = np.linspace(1e-9, 5e-5, 400)

    plt.figure(figsize=(8, 6))
    plt.plot(
        r_plot * 1e6,
        active_pair_distribution.func(r_plot),
        color="blue",
        linewidth=2,
    )

    plt.xlabel(r"Distance $r$ ($\mu$m)", fontsize=12)
    plt.ylabel(r"Pair distribution $g(r)$", fontsize=12)
    plt.title("Pair distribution function", fontsize=14)

    plt.grid(True, linestyle=":", alpha=0.7)

    mo.ui.matplotlib(plt.gca())
    return (active_pair_distribution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Closure Options

    Depending on where you introduce the Pair-Distribution-Function $g$, you get different closures and therefor different Kernels.

    1. If you choose $g$ to be the closure of the two-body-density, and assume that it is independent of the one-body-density you get
        $$
        u_{eff} = g \cdot u
        $$
        which is the *Potential Closure of the Kernel*.
    2. If you choose $g$ to describe, the effective interaction force $f_{eff}$ in terms of the "normal" interaction force $f$

        $$
        f_{eff} = g \cdot f
        $$

        you get the *Force Closure of the Kernel*.
    """)
    return


@app.cell
def _(mo):
    closure_tabs = mo.ui.tabs(
        {
            "Potential closure": mo.md(
                r"""
                $$
                K(x) = 2\pi\,x\,g(|x|)\,u(|x|)
                $$
                """
            ),
            "Force closure": mo.md(
                r"""
                $$
                K(x) = 2\pi\,x\int_{|x|}^{\infty} g(R)\,f(R)\, d R,
                \qquad f(R) = -u'(R)
                $$
                """
            ),
        },
        value="Potential closure",
    )
    closure_tabs
    return (closure_tabs,)


@app.cell
def _(
    SIGMA,
    active_pair_distribution,
    closure_tabs,
    cumulative_trapezoid,
    lennard_jones_potential,
    mo,
    np,
    plt,
):
    x_max = 5e-5
    r_min = 0.95 * SIGMA
    r = np.linspace(r_min, x_max, 2000)
    x = np.linspace(-x_max, x_max, 2001)

    g_r = active_pair_distribution.func(r)
    u_r = lennard_jones_potential(r)
    force_r = -np.gradient(u_r, r)

    def sample_kernel(sample_x: np.ndarray) -> np.ndarray:
        if closure_tabs.value == "Potential closure":
            radial_factor = g_r * u_r
            return 2 * np.pi * sample_x * np.interp(np.abs(sample_x), r, radial_factor)

        force_integrand = g_r * force_r
        tail_integral = -cumulative_trapezoid(
            force_integrand[::-1],
            r[::-1],
            initial=0,
        )[::-1]
        return 2 * np.pi * sample_x * np.interp(np.abs(sample_x), r, tail_integral)

    closure_name = closure_tabs.value
    kernel_values = sample_kernel(x)
    ylabel = r"Kernel $K(x)$ ($\mathrm{J}\,\mathrm{m}$)"

    plt.figure(figsize=(8, 6))
    plt.plot(x * 1e6, kernel_values, color="blue", linewidth=2)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel(r"Offset $x$ ($\mu$m)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(closure_name, fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.7)

    kernel_plot = mo.ui.matplotlib(plt.gca())
    kernel_plot
    return closure_name, kernel_values, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CUDA Export

    This export form writes the **discrete convolution stencil** expected by the
    CUDA code. It matches the legacy `genConvKernel()` convention instead of the
    plotted notebook kernel, so the active notebook closure and pair-distribution
    tabs are ignored here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How The Simulation Samples The Kernel

    The CUDA simulation does not consume a continuous kernel $K(x)$. Instead, it
    uses a discrete odd-length stencil

    $$
    \,\{K_i\}_{i=-(N_K-1)/2}^{(N_K-1)/2}
    $$

    with `kernelN = N_K` samples and spacing

    $$
    \Delta z_K = \frac{DZ}{\text{subDiv}}.
    $$

    The sampled offsets are therefore

    $$
    x_i = i\,\Delta z_K,
    \qquad
    i = -\frac{N_K-1}{2}, \ldots, \frac{N_K-1}{2}.
    $$

    so the total sampled support is

    $$
    L_K = (N_K - 1)\,\Delta z_K.
    $$

    During the convolution step, CUDA multiplies these discrete kernel entries by
    the interpolated density values and then applies one additional factor of

    $$
    \Delta z_K = \frac{DZ}{\text{subDiv}}
    $$

    outside the summation. That means the exported `/kernel/values` dataset must
    already be sampled on this exact grid to reproduce the legacy simulation
    behavior.
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
    export_name = mo.ui.text(value="kernel.h5", label="File name")
    export_kernel_n = mo.ui.number(
        start=3, stop=10001, step=2, value=31, label="Export kernelN"
    )
    export_dz = mo.ui.number(
        start=1e-12,
        step=1e-7,
        value=256 * 1.041412353515625e-6,
        label="DZ",
    )
    export_sub_div = mo.ui.number(
        start=1,
        step=1,
        value=256,
        label="subDiv",
    )

    export_form = (
        mo.md(
            """
    Export a legacy CUDA-compatible convolution kernel.

    {export_dir}

    {export_name}

    {export_kernel_n}

    {export_dz}

    {export_sub_div}
    """
        )
        .batch(
            export_dir=export_dir,
            export_name=export_name,
            export_kernel_n=export_kernel_n,
            export_dz=export_dz,
            export_sub_div=export_sub_div,
        )
        .form(
            submit_button_label="Export CUDA kernel",
            clear_on_submit=False,
            show_clear_button=True,
        )
    )
    export_form
    return (export_form,)


@app.cell
def _(CopyToClipboard, Path, U, export_form, h5py, mo, np):
    def _build_legacy_cuda_kernel(kernel_n: int, dz: float, sub_div: int):
        sigma = 5.6e-6
        sigma_c = 0.5e-6
        eq_dist = 6.585467201064237091254725819933213415424688719213008880615234375e-06
        sub_res = 10000.0

        kernel_l = (kernel_n - 1) * dz / sub_div
        kernel_dz = kernel_l / (kernel_n - 1)
        fine_res = int(sub_res * ((kernel_n + 1) / 2))
        fine_dr = kernel_dz / sub_res
        kernel_fine = np.zeros(fine_res, dtype=np.float64)

        force_sum = 0.0
        for i in range(1, fine_res):
            fine_r = i * fine_dr
            force = 4 * U * (12 * sigma**12 / fine_r**13 - 6 * sigma**6 / fine_r**7)
            gpdf = 4e7 * np.exp(-((fine_r - eq_dist) ** 2) / (2 * sigma_c**2))
            if fine_r < 1e-8:
                gpdf = 0.0
            kernel_fine[i] = force_sum
            force_sum += fine_dr * force * gpdf

        kernel_fine = kernel_fine[-1] - kernel_fine
        kernel_values = np.zeros(kernel_n, dtype=np.float64)
        center = (kernel_n - 1) // 2
        kernel_values[center] = 0.0

        for i in range((kernel_n + 1) // 2, kernel_n):
            kernel_z = i * kernel_dz - kernel_l / 2
            fine_idx = int((i + 1 - (kernel_n + 1) / 2) * sub_res)
            kernel_values[i] = kernel_z * kernel_fine[fine_idx]
            kernel_values[kernel_n - 1 - i] = -kernel_values[i]

        x = (np.arange(kernel_n, dtype=np.float64) - center) * kernel_dz
        return x, kernel_values, kernel_dz

    if export_form.value is None:
        _result = mo.md(
            "Submit the form to export the legacy CUDA-compatible convolution kernel."
        )
    else:
        _export_dir_entries = export_form.value.get("export_dir") or []
        _file_name = str(export_form.value.get("export_name", "")).strip()
        _kernel_n = int(export_form.value.get("export_kernel_n", 0))
        _dz = float(export_form.value.get("export_dz", 0.0))
        _sub_div = int(export_form.value.get("export_sub_div", 0))

        if not _export_dir_entries:
            _result = mo.md("Please select a directory before exporting.")
        elif not _file_name:
            _result = mo.md("Please enter a file name before exporting.")
        elif _kernel_n < 3 or _kernel_n % 2 == 0:
            _result = mo.md(
                "`kernelN` must be an odd integer greater than or equal to `3`."
            )
        elif _dz <= 0.0:
            _result = mo.md("`DZ` must be positive.")
        elif _sub_div < 1:
            _result = mo.md("`subDiv` must be a positive integer.")
        else:
            _selected_dir = Path(_export_dir_entries[0].path)
            _path = _selected_dir / _file_name
            _x, _kernel_values, _kernel_dz = _build_legacy_cuda_kernel(
                _kernel_n,
                _dz,
                _sub_div,
            )

            try:
                _path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(_path, "w") as _f:
                    _kernel_group = _f.create_group("kernel")
                    _kernel_group.create_dataset(
                        "values", data=np.asarray(_kernel_values, dtype=np.float64)
                    )
                    _kernel_group.create_dataset(
                        "x", data=np.asarray(_x, dtype=np.float64)
                    )
                    _kernel_group.attrs["kernelN"] = int(_kernel_values.shape[0])
                    _kernel_group.attrs["spacing"] = float(_kernel_dz)
                    _kernel_group.attrs["DZ"] = float(_dz)
                    _kernel_group.attrs["subDiv"] = int(_sub_div)
                    _kernel_group.attrs["closure"] = "legacy_force_closure"
                    _kernel_group.attrs["pair_distribution"] = "legacy_nearest_neighbor"
                    _kernel_group.attrs["cuda_compatible"] = 1
                    _kernel_group.attrs["generated_by"] = "analysis/kernel.py"
            except Exception as _exc:
                _result = mo.md(rf"Export failed: `{_exc}`")
            else:
                _clipboard = mo.ui.anywidget(CopyToClipboard(text_to_copy=str(_path)))
                _result = mo.vstack(
                    [
                        mo.md(
                            rf"Exported CUDA-compatible convolution kernel to `{_path}`."
                        ),
                        mo.md(
                            rf"Stored `{_kernel_n}` samples with `kernelDZ = {_kernel_dz:.6e}` m on `/kernel/values`."
                        ),
                        mo.hstack(
                            [mo.md("Copy exported path:"), _clipboard], justify="start"
                        ),
                    ]
                )

    _result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Taylor Approximation

    We approximate

    $$
        I(z,t) = \frac{1}{V} \int_{0}^{L} d z' \, \psi(z', t) K(z - z')
    $$

    by expanding $\psi(z',t)$ around $z$ and introducing $\eta = z-z'$. This gives

    $$
        I(z,t)
        = -\frac{1}{V}\sum_{n=0}^{\infty}
        \frac{\partial_z^{2n+1}\psi(z,t)}{(2n+1)!}
        \int_{-\infty}^{+\infty} d\eta \, K(\eta)\eta^{2n+1},
    $$

    where the odd-kernel symmetry removes all even orders on a symmetric domain, giving us

    $$
        I(z,t) \approx -\frac{1}{V}\partial_z \psi(z, t) \int_{-\infty}^{+\infty} d{\eta}\, K(\eta)\eta - \frac{1}{V}\frac{1}{3!}\partial_z^3 \psi(z, t) \int_{-\infty}^{+\infty} d{\eta}\, K(\eta)\eta^3 + \ldots
    $$

    with moments

    $$
    \begin{aligned}
        \nu &=- \frac{1}{V} \int_{-\infty}^{+\infty} d\eta \, K(\eta)\eta, \\
        \mu &=- \frac{1}{V} \frac{1}{3!}\int_{-\infty}^{+\infty} d\eta \, K(\eta)\eta^3.
    \end{aligned}
    $$

    so that

    $$
        I(z,t) \approx \nu \, \partial_z \psi(z,t) + \mu \, \partial_z^3 \psi(z,t).
    $$
    """)
    return


@app.cell
def _(V, closure_name, kernel_values, mo, np, x):
    moment_1 = np.trapezoid(kernel_values * x, x)
    moment_3 = np.trapezoid(kernel_values * x**3, x) / 6

    nu = moment_1 / V
    mu = moment_3 / V
    nu_legacy = moment_1 / (2 * np.pi)
    mu_legacy = moment_3 / (2 * np.pi)

    def latex_scientific(value: float) -> str:
        if value == 0:
            return "0"
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10**exponent)
        return rf"{mantissa:.6f} \cdot 10^{{{exponent}}}"

    mo.md(
        rf"""
    ### Numerical Coefficients

    For the current kernel choice (**{closure_name}**) and $V = 90\,\mathrm{{fL}}$,

    $$
    \begin{{aligned}}
    \nu &= {latex_scientific(nu)}\,\mathrm{{J}}, \\
    \mu &= {latex_scientific(mu)}\,\mathrm{{J\,m^2}}.
    \end{{aligned}}
    $$

    Removing the explicit $2\pi$ and $V$ factors from the notebook convention gives

    $$
    \begin{{aligned}}
    \mathrm{{NU}}_{{legacy}} &= {latex_scientific(nu_legacy)}\,\mathrm{{J\,m^3}}, \\
    \mathrm{{MU}}_{{legacy}} &= {latex_scientific(mu_legacy)}\,\mathrm{{J\,m^5}}.
    \end{{aligned}}
    $$
    """
    )
    return


if __name__ == "__main__":
    app.run()
