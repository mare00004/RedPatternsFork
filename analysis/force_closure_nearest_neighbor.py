import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from red_patterns.kernel import (
        EQ_DIST,
        G0,
        SIGMA as DEFAULT_LJ_SIGMA,
        SIGMA_C,
        compute_force_closure_kernel,
        generate_kernel_stencil,
        guard_pair_distribution,
        lj_derivative,
        lj_potential,
        pdf_nearest_neighbor,
    )
    from red_patterns.sweep_jobs import DEFAULT_DZ

    return (
        DEFAULT_LJ_SIGMA,
        DEFAULT_DZ,
        EQ_DIST,
        G0,
        SIGMA_C,
        compute_force_closure_kernel,
        generate_kernel_stencil,
        guard_pair_distribution,
        lj_derivative,
        lj_potential,
        mo,
        np,
        pdf_nearest_neighbor,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Force-closure nearest-neighbor kernel

    This notebook visualizes the Lennard–Jones potential, the nearest-neighbor
    pair distribution, and the force-closure kernel used by RedPatternsFork.

    The Lennard–Jones pair potential and its radial derivative are

    $$
    u(r)=4U\left[\left(\frac{\sigma}{r}\right)^{12}
                     -\left(\frac{\sigma}{r}\right)^6\right],
    \qquad
    u'(r)=\frac{4U}{r}\left[-12\left(\frac{\sigma}{r}\right)^{12}
                              +6\left(\frac{\sigma}{r}\right)^6\right].
    $$

    The nearest-neighbor pair distribution used in the project is the Gaussian

    $$
    g_{\mathrm{NN}}(r)
      =g_0\exp\!\left[-\frac{(r-d)^2}{2\sigma_C^2}\right].
    $$

    With the pair force $f(r)=-u'(r)$, the repository defines the force-closure
    kernel as

    $$
    \boxed{K(x)
      =x\int_{|x|}^{\infty}g_{\mathrm{NN}}(r)f(r)\,dr
      =-x\int_{|x|}^{\infty}g_{\mathrm{NN}}(r)u'(r)\,dr.}
    $$

    The units fix the units of the pair distribution. Since

    $$
    [x]=[r]=[dr]=\mathrm m,
    \qquad [u]=\mathrm J,
    \qquad [u']=\mathrm{J\,m^{-1}},
    $$

    dimensional consistency requires

    $$
    [K]
      =[x]\,[g_{\mathrm{NN}}]\,[u']\,[dr]
      =\mathrm m\,[g_{\mathrm{NN}}]\,
        \mathrm{J\,m^{-1}}\,\mathrm m
      =[g_{\mathrm{NN}}]\,\mathrm{J\,m}.
    $$

    Therefore $[K]=\mathrm{J\,m}$ implies
    $[g_{\mathrm{NN}}]=[g_0]=1$: both the pair distribution and its amplitude
    are dimensionless. The Gaussian parameters $d$ and $\sigma_C$ have units
    of length.

    There is **no $2\pi$ in the stored kernel**. The CUDA model carries that
    geometrical factor in

    $$
    I(z)=\int \psi(z')K(z-z')\,dz',
    \qquad
    \beta=\frac{2\pi}{\zeta V},
    \qquad
    v_{\mathrm{int}}=-\beta I.
    $$

    Thus one could instead absorb $2\pi$ into a differently named kernel,
    $\widetilde K=2\pi K$, but doing that to the exported kernel while retaining
    the CUDA $\beta$ would count $2\pi$ twice.
    """)
    return


@app.cell
def _(DEFAULT_LJ_SIGMA, mo):
    U_control = mo.ui.number(
        start=1.0,
        stop=500.0,
        step=1.0,
        value=100.0,
        label=r"$U$ [$10^{-18}$ J]",
    )
    sigma_control = mo.ui.number(
        start=1.0,
        stop=10.0,
        step=0.1,
        value=DEFAULT_LJ_SIGMA * 1e6,
        label=r"$\sigma$ [$\mu$m]",
    )
    controls = mo.hstack([U_control, sigma_control], justify="start", gap=2.0)
    controls
    return U_control, sigma_control


@app.cell
def _(
    DEFAULT_DZ,
    EQ_DIST,
    G0,
    SIGMA_C,
    U_control,
    compute_force_closure_kernel,
    generate_kernel_stencil,
    guard_pair_distribution,
    lj_derivative,
    lj_potential,
    np,
    pdf_nearest_neighbor,
    sigma_control,
):
    U = float(U_control.value) * 1e-18
    sigma = float(sigma_control.value) * 1e-6

    pair_distribution = guard_pair_distribution(
        lambda radius: pdf_nearest_neighbor(radius, G0, EQ_DIST, SIGMA_C)
    )

    r_potential = np.linspace(0.95 * sigma, 3.0 * sigma, 800)
    potential = lj_potential(r_potential, U, sigma)

    r_pair = np.linspace(0.0, 14e-6, 800)
    pair_values = pair_distribution(r_pair)

    # This support extends more than 18 Gaussian widths beyond the peak, so the
    # omitted tail is negligible for the default nearest-neighbor distribution.
    x = np.linspace(-16e-6, 16e-6, 401)
    kernel = compute_force_closure_kernel(
        x,
        u_prime_func=lambda radius: lj_derivative(radius, U, sigma),
        g_func=pair_distribution,
        sub_res=2_000,
    )

    default_kernel_n = 31
    default_subdiv = 256
    stencil_spacing = DEFAULT_DZ / default_subdiv
    stencil_x, stencil_kernel = generate_kernel_stencil(
        kernel_func=lambda offsets: compute_force_closure_kernel(
            offsets,
            u_prime_func=lambda radius: lj_derivative(radius, U, sigma),
            g_func=pair_distribution,
            sub_res=10_000,
        ),
        kernel_n=default_kernel_n,
        kernel_dz=stencil_spacing,
    )

    kernel_max = float(np.max(np.abs(np.concatenate([kernel, stencil_kernel]))))
    kernel_scale_exponent = (
        int(np.floor(np.log10(kernel_max))) if kernel_max > 0.0 else 0
    )
    kernel_scale = 10.0**kernel_scale_exponent

    return (
        U,
        default_kernel_n,
        default_subdiv,
        kernel,
        kernel_scale,
        kernel_scale_exponent,
        pair_values,
        potential,
        r_pair,
        r_potential,
        sigma,
        stencil_kernel,
        stencil_spacing,
        stencil_x,
        x,
    )


@app.cell(hide_code=True)
def _(EQ_DIST, G0, SIGMA_C, U, mo, sigma):
    mo.md(rf"""
    ## Parameters shown

    $$
    U={U * 1e18:.1f}\times10^{{-18}}\ \mathrm{{J}},\qquad
    \sigma={sigma * 1e6:.2f}\ \mu\mathrm{{m}},
    $$

    $$
    g_0={G0:.3e}\ \text{{(dimensionless)}},\qquad
    d={EQ_DIST * 1e6:.6f}\ \mu\mathrm{{m}},\qquad
    \sigma_C={SIGMA_C * 1e6:.2f}\ \mu\mathrm{{m}}.
    $$

    Here $\sigma$ is the Lennard–Jones zero-crossing length. Its potential
    minimum is at $r_\min=2^{{1/6}}\sigma$ with value $u(r_\min)=-U$.
    """)
    return


@app.cell
def _(U, np, plt, potential, r_potential, sigma):
    _r_min = np.power(2.0, 1.0 / 6.0) * sigma
    _fig, _ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    _ax.plot(r_potential * 1e6, potential * 1e18, linewidth=2.2)
    _ax.axhline(0.0, color="black", linewidth=0.9)
    _ax.axvline(
        _r_min * 1e6,
        color="tab:red",
        linestyle="--",
        label=rf"$r_\min=2^{{1/6}}\sigma={_r_min * 1e6:.3f}\,\mu$m",
    )
    _ax.scatter([_r_min * 1e6], [-U * 1e18], color="tab:red", zorder=3)
    _ax.set(
        xlabel=r"separation $r$ [$\mu$m]",
        ylabel=r"$u(r)$ [$10^{-18}$ J]",
        title="Lennard–Jones potential",
    )
    _ax.grid(True, linestyle=":", alpha=0.65)
    _ax.legend()
    _fig
    return


@app.cell
def _(EQ_DIST, G0, SIGMA_C, pair_values, plt, r_pair):
    _fig, _ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    _ax.plot(r_pair * 1e6, pair_values / 1e7, color="tab:green", linewidth=2.2)
    _ax.axvline(
        EQ_DIST * 1e6,
        color="tab:red",
        linestyle="--",
        label=rf"$d={EQ_DIST * 1e6:.3f}\,\mu$m",
    )
    _ax.scatter([EQ_DIST * 1e6], [G0 / 1e7], color="tab:red", zorder=3)
    _ax.set(
        xlabel=r"separation $r$ [$\mu$m]",
        ylabel=r"$g_{\mathrm{NN}}(r)$ [$10^{-7}$]",
        title=rf"Nearest-neighbor pair distribution ($\sigma_C={SIGMA_C * 1e6:.2f}\,\mu$m)",
    )
    _ax.grid(True, linestyle=":", alpha=0.65)
    _ax.legend()
    _fig
    return


@app.cell
def _(kernel, kernel_scale, kernel_scale_exponent, plt, x):
    _fig, _ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    _ax.plot(x * 1e6, kernel / kernel_scale, color="tab:purple", linewidth=2.2)
    _ax.axhline(0.0, color="black", linewidth=0.9)
    _ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.5)
    _ax.set(
        xlabel=r"offset $x$ [$\mu$m]",
        ylabel=rf"$K(x)$ [$10^{{{kernel_scale_exponent}}}\,\mathrm{{J\,m}}$]",
        title="Force-closure nearest-neighbor kernel",
    )
    _ax.grid(True, linestyle=":", alpha=0.65)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The kernel is odd: $K(-x)=-K(x)$. Numerically, the upper integration limit
    is the edge of the plotted $\pm16\,\mu$m support; at that distance the
    Gaussian nearest-neighbor distribution is negligible.
    """)
    return


@app.cell(hide_code=True)
def _(
    default_kernel_n,
    default_subdiv,
    mo,
    stencil_spacing,
    stencil_x,
):
    mo.md(rf"""
    ## Default 31-point convolution stencil

    The continuous curve above is sampled before it is passed to the CUDA
    convolution. For an odd stencil length $N_K$, the exporter uses offsets

    $$
    x_j=\left(j-\frac{{N_K-1}}{{2}}\right)h,
    \qquad j=0,\ldots,N_K-1,
    \qquad K_j=K(x_j),
    $$

    with fine-grid spacing

    $$
    h=\frac{{\Delta z}}{{\mathrm{{subdiv}}}}.
    $$

    The repository defaults are $N_K={default_kernel_n}$,
    $\Delta z=2.67651\times10^{{-4}}\,\mathrm m$, and
    $\mathrm{{subdiv}}={default_subdiv}$. Hence

    $$
    h={stencil_spacing:.10e}\,\mathrm m
      ={stencil_spacing * 1e6:.6f}\,\mu\mathrm m,
    $$

    and the stencil covers
    $[{stencil_x[0] * 1e6:.6f},\,{stencil_x[-1] * 1e6:.6f}]\,\mu\mathrm m$.
    The markers below are the 31 values stored for convolution; the line is the
    more densely evaluated kernel shown for reference.

    CUDA forms the corresponding Riemann sum using the same spacing,

    $$
    I_i\approx h\sum_{{j=0}}^{{N_K-1}}
      \psi_{{i+j-(N_K-1)/2}}K_j,
    $$

    before multiplying the result by the model coefficient
    $\beta=2\pi/(\zeta V)$.
    """)
    return


@app.cell
def _(
    kernel,
    kernel_scale,
    kernel_scale_exponent,
    plt,
    stencil_kernel,
    stencil_x,
    x,
):
    _fig, _ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    _ax.plot(
        x * 1e6,
        kernel / kernel_scale,
        color="tab:purple",
        linewidth=2.2,
        label="densely evaluated kernel",
    )
    _ax.scatter(
        stencil_x * 1e6,
        stencil_kernel / kernel_scale,
        s=34,
        facecolor="white",
        edgecolor="black",
        linewidth=1.1,
        zorder=3,
        label="default 31-point stencil",
    )
    _ax.axhline(0.0, color="black", linewidth=0.9)
    _ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.5)
    _ax.set(
        xlabel=r"offset $x$ [$\mu$m]",
        ylabel=rf"$K(x)$ [$10^{{{kernel_scale_exponent}}}\,\mathrm{{J\,m}}$]",
        title="Force-closure kernel with exported stencil",
    )
    _ax.grid(True, linestyle=":", alpha=0.65)
    _ax.legend()
    _fig
    return


if __name__ == "__main__":
    app.run()
