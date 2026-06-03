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

__generated_with = "0.23.8"
app = marimo.App()

with app.setup:
    import numpy as np
    import h5py
    from pathlib import Path
    from red_patterns import Array1F

    # Potential
    SIGMA = 5.6e-6  # 5.6 micrometers converted to meters
    # U = 111.15e-18  # 111.15 * 10^-18 Joules
    V = 90e-18

    # Pair Distribution Function
    G0 = 4.0e7
    SIGMA_C = 0.5e-6
    EQ_DIST = 6.585467201064237091254725819933213415424688719213008880615234375e-06


@app.function
def latex_scientific(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)
    return rf"{mantissa:.6f} \cdot 10^{{{exponent}}}"


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    from typing import Callable, ClassVar, Dict
    from wigglystuff import CopyToClipboard

    return Callable, ClassVar, CopyToClipboard, Dict, dataclass, mo, plt


@app.cell(hide_code=True)
def _(mo):
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
def _(mo):
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
def _(mo):
    mo.md(r"""
    This notebook also calculates the coefficients used in the taylor approximation

    ### 0.1 Taylor Approximation

    We approximate

    $$
        I(z,t) = \frac{2 \pi}{V} \int_{0}^{L} d z' \, \psi(z', t) K(z - z')
    $$

    by expanding $\psi(z',t)$ around $z$ and introducing $\eta = z-z'$. This gives

    $$
        I(z,t)
        = -\frac{2 \pi}{V}\sum_{n=0}^{\infty}
        \frac{\partial_z^{2n+1}\psi(z,t)}{(2n+1)!}
        \int_{-\infty}^{+\infty} d\eta \, K(\eta)\eta^{2n+1},
    $$

    where the odd-kernel symmetry removes all even orders on a symmetric domain, giving us

    $$
        I(z,t) \approx -\frac{2 \pi}{V}\partial_z \psi(z, t) \int_{-\infty}^{+\infty} d{\eta}\, K(\eta)\eta - \frac{2 \pi}{V}\frac{1}{3!}\partial_z^3 \psi(z, t) \int_{-\infty}^{+\infty} d{\eta}\, K(\eta)\eta^3 + \ldots
    $$

    with moments

    $$
    \begin{aligned}
        \nu &= \int_{-\infty}^{+\infty} d\eta \, K(\eta)\eta, \\
        \mu &= \frac{1}{3!}\int_{-\infty}^{+\infty} d\eta \, K(\eta)\eta^3.
    \end{aligned}
    $$

    so that

    $$
        \frac{V}{2 \pi} \cdot I(z,t) \approx -\nu \, \partial_z \psi(z,t) - \mu \, \partial_z^3 \psi(z,t).
    $$
    """)
    return


@app.cell(hide_code=True)
def _(U, mo):
    mo.md(rf"""
    ## 1. Pick your Pair Potential

    For now we use a Lennard-Jones pair potential

    $$
        u(r) = 4U \left(\frac{{\sigma^{{12}} }}{{r^{{12}}}} - \frac{{\sigma^6}}{{r^6}}\right),
    $$

    with $\sigma = {latex_scientific(SIGMA)}\mathrm{{m}}$ and $U = {latex_scientific(U)}\mathrm{{J}}$.
    """)
    return


@app.cell
def _(mo):
    ui_pair_potential_U = mo.ui.number(
        start=0.0, stop=1000, step=0.05, value=100.0, label="$U \\; [10^{-18} J]$"
    )
    ui_pair_potential_sigma = mo.ui.number(
        start=0.0, stop=10, step=0.05, value=5.6, label="$\\sigma \\; [10^{-6} m]$"
    )
    mo.vstack([ui_pair_potential_U, ui_pair_potential_sigma])
    return ui_pair_potential_U, ui_pair_potential_sigma


@app.function
def lj_potential(r: Array1F, U: np.floating, sigma: np.floating):
    """
    Lennard Jones Potential
    $$
        u(r) = 4U \\left(\\frac{\\sigma^{12} }{r^{12}} - \\frac{\\sigma^6}{r^6}\\right),
    $$
    """
    return 4 * U * ((sigma / r) ** 12 - (sigma / r) ** 6)


@app.function
def lj_derivative(r: Array1F, U: np.floating, sigma: np.floating) -> Array1F:
    """
    Analytical Derivative of Lennard Jones Potential. With
    $$
        u(r) = 4U \\left(\\frac{\\sigma^{12} }{r^{12}} - \\frac{\\sigma^6}{r^6}\\right),
    $$
    this function calculates
    $$
        \\frac{d u}{d r}(r) = \\frac{4U}{r} \\left(-12 \\frac{\\sigma^{12} }{r^{12}} +6 \\frac{\\sigma^6}{r^6}\\right),
    $$
    """
    sr6 = (sigma / r) ** 6
    sr12 = sr6**2
    return (4 * U / r) * (-12 * sr12 + 6 * sr6)


@app.cell
def _(mo, plt, ui_pair_potential_U, ui_pair_potential_sigma):
    _U = ui_pair_potential_U.value * 1e-18
    _sigma = ui_pair_potential_sigma.value * 1e-6
    _r = np.linspace(0.95 * _sigma, 3 * _sigma, 500)
    u = lj_potential(_r, _U, _sigma)

    # 4. Convert units for cleaner axis labels
    # Convert r to micrometers (um) and u to 10^-18 Joules (aJ)
    r_um = _r * 1e6
    u_aJ = u * 1e18

    # 5. Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(r_um, u_aJ, color="blue", linewidth=2, label="Lennard-Jones potential")

    # Add a horizontal line at y=0 and vertical line at r=sigma
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(
        _sigma * 1e6,
        color="red",
        linestyle="--",
        label=rf"$\sigma = {_sigma * 1e6} \mu m$",
    )

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

    _lj_plot = mo.ui.matplotlib(plt.gca())
    _lj_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Closure Options

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
    POTENTIAL_CLOSURE = "Potential closure"
    FORCE_CLOSURE = "Force closure"
    ui_closure_type = mo.ui.tabs(
        {
            POTENTIAL_CLOSURE: mo.md(
                r"""
                $$
                K(x) = -x\,g(|x|)\,u(|x|)
                $$
                """
            ),
            FORCE_CLOSURE: mo.md(
                r"""
                $$
                K(x) = x\int_{|x|}^{\infty} g(R)\,f(R)\, d R,
                \qquad f(R) = -u'(R)
                $$
                """
            ),
        },
        value="Force closure",
    )
    ui_closure_type
    return FORCE_CLOSURE, ui_closure_type


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Pick Pair Distribution Function

    The pair distribution function $g(r)$ captures how strongly nearby RBCs are
    correlated. Use the tabs to choose one of the available approximations.
    """)
    return


@app.cell
def _(PairDistributionObject, mo):
    ui_pair_distribution_type = mo.ui.tabs(
        {key: node.markdown for key, node in PairDistributionObject.registry.items()},
        value="Nearest Neighbor",
    )

    ui_pair_distribution_type
    return (ui_pair_distribution_type,)


@app.function
def pdf_mean_field(x: Array1F):
    """
    $$
        g(x) = 1
    $$
    """
    return np.ones_like(x)


@app.cell
def _(mo):
    ui_pdf_mf_md = mo.md(
        r"""
    The mean-field approximation assumes no positional correlations.

    $$
        g(x) = 1
    $$
    """
    )
    return (ui_pdf_mf_md,)


@app.function
def pdf_nearest_neighbor(
    x: Array1F, g0: np.floating, d: np.floating, sigma: np.floating
):
    """
    $$
        g(x) = g_0 \\exp\\left(-\\frac{(r-d)^2}{2\\sigma_C^2}\\right)
    $$
    """
    return g0 * np.exp(-((x - d) ** 2) / (2 * (sigma**2)))


@app.cell
def _(mo):
    ui_pdf_nn_g0 = mo.ui.number(
        start=0.0, stop=10.0, step=0.1, value=4.0, label="$g_0 [m]$"
    )
    ui_pdf_nn_d = mo.ui.number(
        start=0.0, stop=10.0, step=0.00000001, value=6.585467, label="$d [10^{-6} m]$"
    )
    ui_pdf_nn_sigma = mo.ui.number(
        start=0.0, stop=1.0, step=0.01, value=0.5, label="$\\sigma_C [10^{-6} m]$"
    )
    ui_pdf_nn_md = mo.md(
        r"""
    This approximation concentrates the weight around one preferred RBC spacing.

    $$
        g(x) = g_0 \exp\left(-\frac{(r-d)^2}{2\sigma_C^2}\right)
    $$
    """
    )
    return ui_pdf_nn_d, ui_pdf_nn_g0, ui_pdf_nn_md, ui_pdf_nn_sigma


@app.function
def pdf_exponential(x: Array1F, U: np.floating, sigma: np.floating) -> Array1F:
    """
    $$
        g(x) = \\exp \\left( -\\sigma \\frac{u(x)}{U} \\right)
    $$
    """
    return np.exp(-sigma * (lj_potential(x, U, sigma) / U))


@app.cell
def _(mo):
    ui_pdf_ee_lambda = mo.ui.number(
        start=0.0, stop=10.0, step=0.5, value=1.0, label="$\\lambda$"
    )
    ui_pdf_ee_md = mo.md(
        r"""
    This ansatz reuses the pair potential itself to suppress strongly repulsive configurations.

    $$
        g(x) = \exp \left( -\lambda \frac{u(x)}{U} \right)
    $$
    """
    )
    return ui_pdf_ee_lambda, ui_pdf_ee_md


@app.cell
def _(
    Callable,
    ClassVar,
    Dict,
    dataclass,
    mo,
    ui_pair_potential_U,
    ui_pdf_ee_lambda,
    ui_pdf_ee_md,
    ui_pdf_mf_md,
    ui_pdf_nn_d,
    ui_pdf_nn_g0,
    ui_pdf_nn_md,
    ui_pdf_nn_sigma,
):
    PDF_MEAN_FIELD = "Mean Field"
    PDF_NEAREST_NEIGHBOR = "Nearest Neighbor"
    PDF_EXPONENTIAL = "Exponential"

    @dataclass
    class PairDistributionObject:
        key: str
        markdown: mo.Html
        func: Callable[[np.ndarray], np.ndarray]
        registry: ClassVar[Dict[str, "PairDistributionObject"]] = {}

        def __post_init__(self):
            self.registry[self.key] = self

    def _with_guard(
        fn: Callable[[np.ndarray], np.ndarray],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Wrap a pair distribution with the legacy divergence guard.

        The original kernel generator sets g(r)=0 for r<1e-8 to avoid
        numerical issues near the LJ divergence.
        """

        def guarded(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            out = np.asarray(fn(x), dtype=np.float64)
            out = out.copy()
            out[x < 1e-8] = 0.0
            return out

        return guarded

    # Pair Distribution Objects
    _ = PairDistributionObject(
        key=PDF_MEAN_FIELD,
        markdown=ui_pdf_mf_md,
        func=_with_guard(pdf_mean_field),
    )

    _ = PairDistributionObject(
        key=PDF_NEAREST_NEIGHBOR,
        markdown=mo.vstack([ui_pdf_nn_md, ui_pdf_nn_g0, ui_pdf_nn_d, ui_pdf_nn_sigma]),
        func=_with_guard(
            # TODO:
            lambda x: pdf_nearest_neighbor(x, ui_pdf_nn_g0.value * 1e+7, ui_pdf_nn_d.value * 1e-6, ui_pdf_nn_sigma.value * 1e-6)
            # lambda x: pdf_nearest_neighbor(
                # x, ui_pdf_nn_g0.value * 1e7, EQ_DIST, ui_pdf_nn_sigma.value * 1e-6
            # )
        ),
    )

    _ = PairDistributionObject(
        key=PDF_EXPONENTIAL,
        markdown=mo.vstack([ui_pdf_ee_md, ui_pdf_ee_lambda]),
        func=_with_guard(
            lambda x: pdf_exponential(x, ui_pair_potential_U, ui_pdf_ee_lambda)
        ),
    )
    return (PairDistributionObject,)


@app.function
def compute_force_closure_kernel(
    x,
    u_prime_func,
    g_func,
    sub_res: float = 10000.0,
):
    """
    Computes

    $$
        K(x) = - x\\int_{|x|}^\\infty g(R) u'(R) dR
    $$

    ## Example

    ```python
    x_out = np.linspace(-15e-6, 15e-6, 100_001)
    K_out = compute_force_kernel(
        x_out,
        u_prime_func=lambda r: lj_derivative(r, U, SIGMA),
        g_func=lambda r: pdf_nearest_neighbor(r, G0, EQ_DIST, SIGMA_C),
        sub_res=10_000,
    )
    ```
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("`x` must be a 1-D array")
    if sub_res <= 0:
        raise ValueError("`sub_res` must be positive")
    if x.size == 0:
        return np.asarray([], dtype=np.float64)
    if x.size == 1:
        return np.zeros_like(x, dtype=np.float64)

    def _infer_uniform_spacing(sample_x: np.ndarray) -> float:
        spacing = np.diff(sample_x)
        if not np.allclose(spacing, spacing[0]):
            raise ValueError("`x` must be sampled on a uniform grid")
        if spacing[0] <= 0.0:
            raise ValueError("`x` must be strictly increasing")
        return float(spacing[0])

    def _build_force_closure_radial_grid(
        sample_x: np.ndarray, kernel_dz: float, radial_sub_res: float
    ) -> tuple[np.ndarray, float]:
        max_multiple = float(np.max(np.abs(sample_x)) / kernel_dz)
        fine_res = int(radial_sub_res * (max_multiple + 1.0))
        fine_dr = kernel_dz / radial_sub_res
        r = np.arange(fine_res, dtype=np.float64) * fine_dr
        return r, fine_dr

    def _evaluate_force_closure_inputs(
        r: np.ndarray, fine_dr: float
    ) -> tuple[np.ndarray, np.ndarray]:
        r_eval = np.where(r > 0.0, r, fine_dr)
        g_vals = np.asarray(g_func(r_eval), dtype=np.float64)
        u_prime_vals = np.asarray(u_prime_func(r_eval), dtype=np.float64)
        g_vals[0] = 0.0
        u_prime_vals[0] = 0.0
        return g_vals, u_prime_vals

    def _accumulate_force_closure_tail(
        g_vals: np.ndarray, u_prime_vals: np.ndarray, fine_dr: float
    ) -> np.ndarray:
        contributions = fine_dr * u_prime_vals * g_vals
        kernel_fine = np.empty_like(g_vals, dtype=np.float64)
        kernel_fine[0] = 0.0
        # Exclusive left-to-right prefix sum: kernel_fine[k] holds the integral
        # contributions from indices 1..k-1 (contributions[0] is 0 anyway).
        np.cumsum(contributions[:-1], out=kernel_fine[1:])
        return kernel_fine[-1] - kernel_fine

    def _evaluate_force_closure_on_samples(
        sample_x: np.ndarray, tail: np.ndarray, fine_dr: float
    ) -> np.ndarray:
        sample_idx = np.rint(np.abs(sample_x) / fine_dr).astype(np.int64)
        sample_idx = np.clip(sample_idx, 0, tail.size - 1)
        return -(sample_x * tail[sample_idx])

    kernel_dz = _infer_uniform_spacing(x)
    r, fine_dr = _build_force_closure_radial_grid(x, kernel_dz, sub_res)
    g_vals, u_prime_vals = _evaluate_force_closure_inputs(r, fine_dr)
    tail = _accumulate_force_closure_tail(g_vals, u_prime_vals, fine_dr)
    return _evaluate_force_closure_on_samples(x, tail, fine_dr)


@app.function
def compute_potential_closure_kernel(
    x,
    u_func,
    g_func,
):
    """
    Computes the potential-closure kernel

    $$
        K(x) = -x\\,g(|x|)\\,u(|x|)
    $$

    Unlike the force closure, this is a pointwise evaluation — no
    integration is required.

    ## Example

    ```python
    x_out = np.linspace(-15e-6, 15e-6, 100_001)
    K_out = compute_potential_closure_kernel(
        x_out,
        u_func=lambda r: lj_potential(r, U, SIGMA),
        g_func=lambda r: pdf_nearest_neighbor(r, G0, EQ_DIST, SIGMA_C),
    )
    ```
    """
    x = np.asarray(x, dtype=np.float64)
    abs_x = np.abs(x)
    K = np.zeros_like(x, dtype=np.float64)
    nonzero_mask = abs_x > 0.0
    if np.any(nonzero_mask):
        g_vals = np.asarray(g_func(abs_x[nonzero_mask]), dtype=np.float64)
        u_vals = np.asarray(u_func(abs_x[nonzero_mask]), dtype=np.float64)
        K[nonzero_mask] = -(x[nonzero_mask] * g_vals * u_vals)
    return K


@app.cell
def _(PairDistributionObject, ui_pair_distribution_type):
    active_pair_distribution = PairDistributionObject.registry[
        ui_pair_distribution_type.value
    ]
    return (active_pair_distribution,)


@app.cell
def _(active_pair_distribution, mo, plt):
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

    _pdf_plot = mo.ui.matplotlib(plt.gca())
    _pdf_plot
    return


@app.function
def generate_kernel_stencil(kernel_func, kernel_n: np.integer, kernel_dz: np.floating):
    """Sample a continuous kernel on a discrete odd-length stencil.

    Given a callable ``kernel_func`` that evaluates a kernel $K(x)$ on an
    arbitrary array of offsets, this function places an odd-length stencil
    of ``kernel_n`` points with spacing ``kernel_dz`` centered at $x = 0$
    and evaluates the kernel there.

    ## Parameters

    kernel_func : callable
        A function that accepts a 1-D numpy array of offsets $x$ (in
        metres) and returns the kernel values $K(x)$ on that grid.
        Typical examples:

        * Force closure:
          ``lambda x: compute_force_closure_kernel(x, u_prime_func=..., g_func=...)``
        * Potential closure:
          ``lambda x: compute_potential_closure_kernel(x, u_func=..., g_func=...)``

    kernel_n : int
        Number of stencil points.  Must be an odd integer $\\geq 3$
        so that the stencil has a well-defined centre.

    kernel_dz : float
        Spacing between adjacent stencil points, in metres.

    ## Returns

    `numpy.ndarray`
        1-D array of ``kernel_n`` kernel values
        $K(x_i)$ with $x_i = i \\cdot \\text{kernel\\_dz}$,
        $i = -(N-1)/2, \\ldots, (N-1)/2$.

    ## Raises

    AssertionError
        If ``kernel_n`` is even.

    ## Examples

    # Force Closure

    ```python
        x_sample, K_sample = generate_kernel_stencil(
        kernel_func=lambda x: compute_force_closure_kernel(
            x,
            u_prime_func=lambda r: lj_derivative(r, U, SIGMA),
            g_func=lambda r: pdf_nearest_neighbor(r, G0, EQ_DIST, SIGMA_C),
        ),
        kernel_n=31,
        kernel_dz=fine_dz,
    )
    ```

    # Potential closure

    >>> x_sample, K_sample = generate_kernel_stencil(
    ...     kernel_func=lambda x: compute_potential_closure_kernel(
    ...         x,
    ...         u_func=lambda r: lj_potential(r, U, SIGMA),
    ...         g_func=lambda r: pdf_nearest_neighbor(r, G0, EQ_DIST, SIGMA_C),
    ...     ),
    ...     kernel_n=31,
    ...     kernel_dz=fine_dz,
    ... )
    """
    assert kernel_n % 2 != 0, "`kernel_n` needs to be odd!"
    center_idx = (kernel_n - 1) // 2
    x = (np.arange(kernel_n, dtype=np.float64) - center_idx) * kernel_dz
    return x, kernel_func(x)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Inspect Kernel

    First pick parameters for the export
    """)
    return


@app.cell
def _(mo):
    ui_kernel_n = mo.ui.number(start=3, stop=10001, step=2, value=31, label="kernelN")
    ui_dz = mo.ui.number(
        start=1e-12,
        step=1e-8,
        value=256.0 * 1.0455122765372783e-6,
        label="(coarse) $\\Delta z$",
    )
    ui_subDiv = mo.ui.number(start=1, step=1, value=256, label="subDiv")

    mo.vstack([ui_kernel_n, ui_dz, ui_subDiv])
    return ui_dz, ui_kernel_n, ui_subDiv


@app.cell
def _(
    FORCE_CLOSURE,
    active_pair_distribution,
    ui_closure_type,
    ui_dz,
    ui_kernel_n,
    ui_pair_potential_U,
    ui_pair_potential_sigma,
    ui_subDiv,
):
    coarse_dz = float(ui_dz.value)
    sub_div = int(ui_subDiv.value)
    fine_dz = coarse_dz / sub_div
    kernelN = ui_kernel_n.value

    g_func = active_pair_distribution.func

    U = ui_pair_potential_U.value * 1e-18
    sigma = ui_pair_potential_sigma.value * 1e-6
    u_func = lambda r: lj_potential(r, U, sigma)
    u_prime_func = lambda r: lj_derivative(r, U, sigma)

    if ui_closure_type.value == FORCE_CLOSURE:
        kernel_func = lambda x: compute_force_closure_kernel(
            x,
            u_prime_func=u_prime_func,
            g_func=g_func,
        )
    else:
        kernel_func = lambda x: compute_potential_closure_kernel(
            x,
            u_func=u_func,
            g_func=g_func,
        )

    x_sample, K_sample = generate_kernel_stencil(
        kernel_func=kernel_func,
        kernel_n=kernelN,
        kernel_dz=fine_dz,
    )

    # Keep a separate continuum path for inspection/plotting.
    _scale = 101
    x, K = generate_kernel_stencil(
        kernel_func=kernel_func,
        kernel_n=kernelN * _scale,
        kernel_dz=fine_dz / _scale,
    )
    return K, K_sample, U, fine_dz, kernelN, x, x_sample


@app.cell
def _(
    K,
    K_sample,
    active_pair_distribution,
    mo,
    plt,
    ui_closure_type,
    x,
    x_sample,
):
    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.plot(x * 1e6, K, color="blue", linewidth=2, label="dense sampled curve")
    _ax.scatter(
        x_sample * 1e6,
        K_sample,
        s=18,
        color="black",
        label="exported stencil",
        zorder=3,
    )

    _closure_name = ui_closure_type.value
    _pair_key = active_pair_distribution.key

    _ax.axhline(0, color="black", linewidth=1)
    _ax.set_xlabel(r"Offset $x$ ($\mu$m)", fontsize=12)
    _ax.set_ylabel(r"Kernel $K(x)$", fontsize=12)
    _ax.set_title(f"{_closure_name} ({_pair_key})", fontsize=14)
    _ax.grid(True, linestyle=":", alpha=0.7)
    _ax.legend()

    mo.ui.matplotlib(_ax)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CUDA Export

    This export form writes the **discrete convolution stencil** expected by the
    CUDA code. The exported stencil is built from the currently selected closure
    and pair distribution.
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
def _(mo):
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
        .batch(
            export_dir=export_dir,
            export_name=export_name,
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
def _(
    CopyToClipboard,
    K_sample,
    U,
    export_form,
    fine_dz,
    kernelN,
    mo,
    ui_closure_type,
    ui_dz,
    ui_pair_distribution_type,
    ui_subDiv,
    x_sample,
):
    if export_form.value is None:
        _result = mo.md(
            "Submit the form to export a CUDA-compatible convolution kernel."
        )
    else:
        _export_dir_entries = export_form.value.get("export_dir") or []
        _file_name = str(export_form.value.get("export_name", "")).strip()
        # Always export with the same live parameters used for plotting.
        _kernel_n = int(kernelN)
        _dz = float(ui_dz.value)
        _sub_div = int(ui_subDiv.value)
        _u_scale = U

        if not _export_dir_entries:
            _result = mo.md("Please select a directory before exporting.")
        elif not _file_name:
            _result = mo.md("Please enter a file name before exporting.")
        else:
            _selected_dir = Path(_export_dir_entries[0].path)
            _path = _selected_dir / _file_name

            _closure = str(ui_closure_type.value)
            _pair_key = str(ui_pair_distribution_type.value)

            try:
                _path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(_path, "w") as _f:
                    _kernel_group = _f.create_group("kernel")
                    _kernel_group.create_dataset(
                        "values", data=np.asarray(K_sample, dtype=np.float64)
                    )
                    _kernel_group.create_dataset(
                        "x", data=np.asarray(x_sample, dtype=np.float64)
                    )
                    _kernel_group.attrs["kernelN"] = int(kernelN)
                    _kernel_group.attrs["spacing"] = float(fine_dz)
                    _kernel_group.attrs["DZ"] = float(_dz)
                    _kernel_group.attrs["subDiv"] = int(_sub_div)
                    _kernel_group.attrs["closure"] = _closure
                    _kernel_group.attrs["pair_distribution"] = _pair_key
                    _kernel_group.attrs["U"] = float(_u_scale)
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
                            rf"Stored `{_kernel_n}` samples with spacing `{fine_dz:.6e}` m on `/kernel/values`."
                        ),
                        mo.md(
                            rf"Used live plotting parameters: `kernelN={_kernel_n}`, `DZ={_dz:.6e}`, `subDiv={_sub_div}`, `U={_u_scale:.6e}` (J)."
                        ),
                        mo.hstack(
                            [mo.md("Copy exported path:"), _clipboard], justify="start"
                        ),
                    ]
                )

    _result
    return


@app.function
def calculate_nu_mu(x: Array1F, K: Array1F) -> tuple[np.floating, np.floating]:
    """
    Compute the Taylor-expansion moments $\\nu$ and $\\mu$ from a discrete kernel stencil.

    $$
    \\begin{aligned}
        \\nu &= \\int_{-\\infty}^{+\\infty} d\\eta \\, K(\\eta)\\eta, \\\\
        \\mu &= \\frac{1}{3!}\\int_{-\\infty}^{+\\infty} d\\eta \\, K(\\eta)\\eta^3.
    \\end{aligned}
    $$
    """
    x = np.asarray(x, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if x.ndim != 1 or K.ndim != 1 or x.shape != K.shape:
        raise ValueError("`x` and `K` must be 1-D arrays with the same shape")
    if x.size < 2:
        raise ValueError("Need at least two stencil points to infer the spacing")

    dz = float(x[1] - x[0])
    nu = dz * float(np.sum(x * K))
    mu = (dz * float(np.sum((x**3) * K))) / 6.0

    return nu, mu


@app.cell
def _(K, mo, x):
    # _nu, _mu = calculate_nu_mu(x_sample, K_sample)
    _nu, _mu = calculate_nu_mu(x, K)

    mo.md(
        rf"""
    ### Numerical Coefficients

    $$
    \begin{{aligned}}
    \nu &= {latex_scientific(_nu)}, \\
    \mu &= {latex_scientific(_mu)}.
    \end{{aligned}}
    $$
    """
    )
    return


if __name__ == "__main__":
    app.run()
