# /// script
# dependencies = [
#     "h5py==3.16.0",
#     "altair==6.1.0",
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "pyarrow==24.0.0",
#     "nbformat==5.10.4",
#     "nbconvert==7.17.1",
#     "playwright==1.60.0",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import altair as alt
    import pyarrow  # noqa: F401
    from red_patterns import Array1F
    from kernel import (
        calculate_nu_mu,
        generate_kernel_stencil,
        compute_force_closure_kernel,
        lj_derivative,
        pdf_nearest_neighbor,
        latex_scientific,
    )

    return (
        alt,
        calculate_nu_mu,
        compute_force_closure_kernel,
        generate_kernel_stencil,
        latex_scientific,
        lj_derivative,
        mo,
        np,
        pd,
        pdf_nearest_neighbor,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # $\nu$ & $\mu$ for the force closure, nearest neighbor, kernel
    """)
    return


@app.cell
def _():
    # Pair Distribution Function
    G0 = 4.0e7
    EQ_DIST = 6.587340962e-6 # From `nu_of_d.py`
    SIGMA_C = 0.5e-6
    return EQ_DIST, G0, SIGMA_C


@app.cell
def _(
    EQ_DIST,
    G0,
    SIGMA_C,
    calculate_nu_mu,
    compute_force_closure_kernel,
    generate_kernel_stencil,
    lj_derivative,
    np,
    pdf_nearest_neighbor,
):
    def with_guard(fn):
        def guarded(x):
            x = np.asarray(x, dtype=np.float64)
            out = np.asarray(fn(x), dtype=np.float64)
            out = out.copy()
            out[x < 1e-8] = 0.0
            return out

        return guarded

    def kernel(U: np.floating, sigma: np.floating):
        scale = 101
        x, K = generate_kernel_stencil(
            kernel_func=lambda s: compute_force_closure_kernel(
                s,
                u_prime_func=lambda r: lj_derivative(r, U, sigma),
                g_func=with_guard(
                    lambda r: pdf_nearest_neighbor(r, G0, EQ_DIST, SIGMA_C)
                ),
                sub_res=10_000,
            ),
            kernel_n=31 * scale,
            kernel_dz=1.0455122765372783e-6 / scale,
        )
        return calculate_nu_mu(x, K)

    return kernel, with_guard


@app.cell
def _(np):
    U_start = 80e-18
    U_end = 120e-18
    U_sweep = np.linspace(U_start, U_end, 20)

    sigma_start = 3e-6
    sigma_end = 8e-6
    sigma_sweep = np.linspace(sigma_start, sigma_end, 40)
    return U_sweep, sigma_sweep


@app.cell
def _(U_sweep, kernel, np, sigma_sweep):
    # nu and mu are linear in U (it enters u'(r) as a pure prefactor), so the
    # expensive integral only needs to be evaluated once per sigma at U=1 and
    # then scaled by U. This turns the 2D sweep into a 1D sweep over sigma.
    base_nu_mu = np.array([kernel(1.0, sigma) for sigma in sigma_sweep])  # (n_sigma, 2)
    nu_mu_grid = U_sweep[:, None, None] * base_nu_mu[None, :, :]  # (n_U, n_sigma, 2)
    return base_nu_mu, nu_mu_grid


@app.cell
def _(nu_mu_grid):
    nu_grid = nu_mu_grid[:, :, 0]
    mu_grid = nu_mu_grid[:, :, 1]
    return mu_grid, nu_grid


@app.cell
def _(U_sweep, mu_grid, np, nu_grid, pd, sigma_sweep):
    # Long-form table for plotting: one row per (U, sigma) cell.
    _U_mesh, _sigma_mesh = np.meshgrid(U_sweep, sigma_sweep, indexing="ij")
    sweep_df = pd.DataFrame(
        {
            # Convert to friendly units: U in 1e-18 J, sigma in 1e-6 m.
            "U_aJ": _U_mesh.ravel() * 1e18,
            "sigma_um": _sigma_mesh.ravel() * 1e6,
            "nu": nu_grid.ravel(),
            "mu": mu_grid.ravel(),
        }
    )
    sweep_df["U_label"] = sweep_df["U_aJ"].map(lambda v: f"{v:.1f}")
    sweep_df["sigma_label"] = sweep_df["sigma_um"].map(lambda v: f"{v:.3f}")
    # Rescale to readable units so the color legend isn't all zeros.
    sweep_df["nu_1e24"] = sweep_df["nu"] * 1e24
    sweep_df["mu_1e36"] = sweep_df["mu"] * 1e36
    return (sweep_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. $\nu$ and $\mu$ heatmaps over $(U, \sigma)$
    """)
    return


@app.cell
def _(alt, mo, sweep_df):
    def _heatmap(value: str, color_title: str, chart_title: str):
        return (
            alt.Chart(sweep_df)
            .mark_rect(stroke="black", strokeWidth=0.5)
            .encode(
                x=alt.X(
                    "U_label:O",
                    title="U  [10⁻¹⁸ J]",
                    sort=alt.SortField(field="U_aJ", order="ascending"),
                ),
                y=alt.Y(
                    "sigma_label:O",
                    title="σ  [10⁻⁶ m]",
                    sort=alt.SortField(field="sigma_um", order="descending"),
                ),
                color=alt.Color(
                    f"{value}:Q",
                    title=color_title,
                    scale=alt.Scale(scheme="viridis"),
                ),
                tooltip=[
                    alt.Tooltip("U_aJ:Q", title="U [aJ]", format=".2f"),
                    alt.Tooltip("sigma_um:Q", title="σ [µm]", format=".3f"),
                    alt.Tooltip(f"{value}:Q", title=color_title, format=".3f"),
                ],
            )
            .properties(width=320, height=320, title=chart_title)
        )

    _charts = alt.hconcat(
        _heatmap("nu_1e24", "ν  [10⁻²⁴]", "ν(U, σ)"),
        _heatmap("mu_1e36", "μ  [10⁻³⁶]", "μ(U, σ)"),
    ).resolve_scale(color="independent")
    mo.ui.altair_chart(_charts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. $\nu(U, \sigma)$ and $\mu(U, \sigma)$ surfaces
    """)
    return


@app.cell
def _(U_sweep, mo, mu_grid, np, nu_grid, plt, sigma_sweep):
    # 3D surfaces of nu and mu over the (U, sigma) sweep.
    _U_mesh, _sigma_mesh = np.meshgrid(U_sweep * 1e18, sigma_sweep * 1e6, indexing="ij")

    _fig = plt.figure(figsize=(11, 4.5), constrained_layout=True)
    for _idx, (_grid, _label) in enumerate(
        ((nu_grid, r"$\nu$"), (mu_grid, r"$\mu$")), start=1
    ):
        _ax = _fig.add_subplot(1, 2, _idx, projection="3d")
        _surf = _ax.plot_surface(
            _U_mesh, _sigma_mesh, _grid, cmap="viridis", linewidth=0, antialiased=True
        )
        _ax.set_xlabel(r"$U$  [$10^{-18}$ J]")
        _ax.set_ylabel(r"$\sigma$  [$10^{-6}$ m]")
        _ax.set_zlabel(_label)
        _ax.set_title(rf"{_label}$(U, \sigma)$")
        _fig.colorbar(_surf, ax=_ax, shrink=0.6, pad=0.12)

    mo.ui.matplotlib(_ax)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Analytical $\sigma$-dependence

    ### 3.1. Derivation

    $\sigma$ enters the entire pipeline through exactly one place, the
    Lennard-Jones force `lj_derivative`:

    $$
    u'(R) = \frac{4U}{R}\!\left(-12\frac{\sigma^{12}}{R^{12}} + 6\frac{\sigma^{6}}{R^{6}}\right)
          = 24U\,R^{-7}\,\sigma^{6}
          \;-\; 48U\,R^{-13}\,\sigma^{12}.
    $$

    Everything downstream is **linear** in $u'$ — the inner integral
    $K(x) = -x\int_{|x|}^{\infty} g(R)\,u'(R)\,dR$, and the moments
    $\nu=\int K\eta\,d\eta$, $\mu=\tfrac{1}{3!}\int K\eta^{3}\,d\eta$ — while $g(R)$,
    the integration limits and the grid do **not** depend on $\sigma$. So
    $\sigma^{6}$ and $\sigma^{12}$ pull straight through as scalar prefactors:

    $$
    \boxed{\;\nu(\sigma) = a_6\,\sigma^{6} + a_{12}\,\sigma^{12},
            \qquad
            \mu(\sigma) = b_6\,\sigma^{6} + b_{12}\,\sigma^{12}\;}
    $$

    with $\sigma$-independent coefficients. Writing
    $J_n(x) = x\int_{|x|}^{\infty} g(R)\,R^{-n}\,dR$,

    $$
    \begin{align*}
    a_6 & = -24U\!\int\!\eta\,J_7\,d\eta,\quad
    a_{12} = +48U\!\int\!\eta\,J_{13}\,d\eta,\quad \\
    b_{6} & = \frac{-24U}{6}\!\int\!\eta^{3} J_{7}\,d\eta, \quad
    b_{12} = \frac{+48U}{6}\!\int\!\eta^{3} J_{13}\,d\eta
    \end{align*}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.2. Determining $a_6, a_{12}$ and $b_6, b_{12}$

    #### 3.2.1. by fitting

    The coefficients $a_6, a_{12}$ (for $\nu$) and $b_6, b_{12}$ (for $\mu$) are
    determined by least-squares regression on the $U = 1$ base profiles.
    Because the model is exact (not approximate), the regression residual is
    numerically zero.

    The least-squares problem is solved on a **normalised basis** to avoid
    catastrophic ill-conditioning: raw $\sigma^{6}$ and $\sigma^{12}$
    differ by $\sim 30$ orders of magnitude across the sweep range, so
    an un-scaled system is numerically singular. Defining $s_0 = \bar\sigma$
    (the mean of the sweep values),

    $$
    \min_{c_1,\,c_2} \sum_i \bigl[y_i \;-\; c_1\,(\sigma_i/s_0)^{6} \;-\; c_2\,(\sigma_i/s_0)^{12}\bigr]^2,
    $$

    after which the physical coefficients are recovered by undoing the
    normalisation:

    $$
    a_6 = \frac{c_1}{s_0^{6}}, \qquad a_{12} = \frac{c_2}{s_0^{12}}
    $$

    (and identically for $b_6, b_{12}$). Since $\nu$ and $\mu$ are linear in
    $U$, only the $U = 1$ sweep is needed; the user-chosen $U$ simply scales
    the result:

    $$
    \nu(U, \sigma) = U\,(a_6\,\sigma^{6} + a_{12}\,\sigma^{12}), \qquad
    \mu(U, \sigma) = U\,(b_6\,\sigma^{6} + b_{12}\,\sigma^{12}).
    $$
    """)
    return


@app.cell
def _(base_nu_mu, np, sigma_sweep):
    # nu and mu are *exactly* a6*sigma**6 + a12*sigma**12 (sigma enters only
    # through u', which is linear in sigma**6 and sigma**12). Fit the two
    # coefficients on a normalized basis -- raw sigma**6 and sigma**12 differ by
    # ~30 orders of magnitude, so an unscaled lstsq is hopelessly ill-conditioned.
    _s0 = float(np.mean(sigma_sweep))
    _basis = np.column_stack([(sigma_sweep / _s0) ** 6, (sigma_sweep / _s0) ** 12])

    def _fit(_y):
        # Least squares fit
        _c, *_ = np.linalg.lstsq(_basis, _y, rcond=None)
        return np.array([_c[0] / _s0**6, _c[1] / _s0**12])  # -> physical sigma

    nu_coef = _fit(base_nu_mu[:, 0])  # [a6, a12] for nu at U=1
    mu_coef = _fit(base_nu_mu[:, 1])  # [b6, b12] for mu at U=1
    return mu_coef, nu_coef


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3.2.2. by approximating the integral:

    ...
    """)
    return


@app.cell
def _(
    EQ_DIST,
    G0,
    SIGMA_C,
    compute_force_closure_kernel,
    generate_kernel_stencil,
    np,
    pdf_nearest_neighbor,
    with_guard,
):
    # Compute J_7 from its integral definition
    # J_n(x) = x * integral_{|x|}^{inf} g(R) * R^{-n} dR
    # Method: use generate_kernel_stencil with u' = R^{-7}
    # K_n(x) = -x * integral_{|x|}^{inf} g(R) * R^{-n} dR = -J_n(x)

    _x7, _K7 = generate_kernel_stencil(
        kernel_func=lambda s: compute_force_closure_kernel(
            s,
            u_prime_func=lambda r: r ** (-7),
            g_func=with_guard(
                lambda r: pdf_nearest_neighbor(r, G0, EQ_DIST, SIGMA_C)
            ),
            sub_res=10_000,
        ),
        kernel_n=31 * 101,
        kernel_dz=1.0455122765372783e-6 / 101,
    )

    J7 = -_K7  # J_7(x) = -K_7(x)

    # Compute J_13 the same way J_7 was computed (u' = R^{-13})
    _x13, _K13 = generate_kernel_stencil(
        kernel_func=lambda s: compute_force_closure_kernel(
            s,
            u_prime_func=lambda r: r ** (-13),
            g_func=with_guard(
                lambda r: pdf_nearest_neighbor(r, G0, EQ_DIST, SIGMA_C)
            ),
            sub_res=10_000,
        ),
        kernel_n=31 * 101,
        kernel_dz=1.0455122765372783e-6 / 101,
    )

    J13 = -_K13  # J_13(x) = -K_13(x)

    # a_6 = -24 U * integral(eta * J_7(eta), d_eta),  with U = 1
    _int_eta_J7 = np.trapezoid(_x7 * J7, _x7)
    a6_integral = -24.0 * 1.0 * _int_eta_J7  # a_6 at U = 1

    # a_12 = +48 U * integral(eta * J_13(eta), d_eta),  with U = 1
    _int_eta_J13 = np.trapezoid(_x13 * J13, _x13)
    a12_integral = 48.0 * _int_eta_J13

    # b_6 = (-24 U / 6) * integral(eta^3 * J_7(eta), d_eta),  with U = 1
    _int_eta3_J7 = np.trapezoid(_x7**3 * J7, _x7)
    b6_integral = -24.0 / 6.0 * _int_eta3_J7

    # b_12 = (+48 U / 6) * integral(eta^3 * J_13(eta), d_eta),  with U = 1
    _int_eta3_J13 = np.trapezoid(_x13**3 * J13, _x13)
    b12_integral = 48.0 / 6.0 * _int_eta3_J13
    return a12_integral, a6_integral, b12_integral, b6_integral


@app.cell
def _(
    a12_integral,
    a6_integral,
    b12_integral,
    b6_integral,
    latex_scientific,
    mo,
    mu_coef,
    nu_coef,
):
    # Fitted counterparts (all at U = 1)
    a6_fit = nu_coef[0]
    a12_fit = nu_coef[1]
    b6_fit = mu_coef[0]
    b12_fit = mu_coef[1]

    mo.md(
        rf"""
    ### 3.3. Integral vs. fitted coefficients (at $U = 1$)

    | Coefficient | Integral | Fitted | Relative difference |
    |-------------|----------|--------|---------------------|
    | $a_6$       | ${latex_scientific(a6_integral)}$ | ${latex_scientific(a6_fit)}$ | ${abs(a6_integral - a6_fit) / abs(a6_fit):.2e}$ |
    | $a_{{12}}$  | ${latex_scientific(a12_integral)}$ | ${latex_scientific(a12_fit)}$ | ${abs(a12_integral - a12_fit) / abs(a12_fit):.2e}$ |
    | $b_6$       | ${latex_scientific(b6_integral)}$ | ${latex_scientific(b6_fit)}$ | ${abs(b6_integral - b6_fit) / abs(b6_fit):.2e}$ |
    | $b_{{12}}$  | ${latex_scientific(b12_integral)}$ | ${latex_scientific(b12_fit)}$ | ${abs(b12_integral - b12_fit) / abs(b12_fit):.2e}$ |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.4. Visualization
    """)
    return


@app.cell
def _(latex_scientific, mo, mu_coef, nu_coef, ui_U):
    _U = ui_U.value * 1.0e-18
    mo.md(
    rf"""
    For $U = {latex_scientific(_U)}\;J$ the coefficients are
    $$
    \begin{{align*}}
        &a_6 = {latex_scientific(_U * nu_coef[0])} \qquad a_{{12}} = {latex_scientific(_U * nu_coef[1])} \\
        &b_6 = {latex_scientific(_U * mu_coef[0])} \qquad b_{{12}} = {latex_scientific(_U * mu_coef[1])} \\
    \end{{align*}}
    $$
    """
    )
    return


@app.cell
def _(U_sweep, mo):
    ui_U = mo.ui.slider(
        start=float(U_sweep[0] * 1e18),
        stop=float(U_sweep[-1] * 1e18),
        step=0.5,
        value=float(U_sweep[0] * 1e18),
        label="$U \\; [10^{-18}\\,\\mathrm{J}]$",
        show_value=True,
    )
    ui_U
    return (ui_U,)


@app.cell
def _(base_nu_mu, mo, mu_coef, np, nu_coef, plt, sigma_sweep, ui_U):
    # nu and mu are linear in U: scale the U=1 profiles by the chosen U.
    _U = ui_U.value * 1e-18
    _sigma_um = sigma_sweep * 1e6
    _nu_pts = _U * base_nu_mu[:, 0] * 1e24
    _mu_pts = _U * base_nu_mu[:, 1] * 1e36

    # Analytical two-term power law a6*sigma**6 + a12*sigma**12 on a fine grid.
    _sig = np.linspace(sigma_sweep[0], sigma_sweep[-1], 300)
    _sig_um = _sig * 1e6
    _nu_fit = _U * (nu_coef[0] * _sig**6 + nu_coef[1] * _sig**12) * 1e24
    _mu_fit = _U * (mu_coef[0] * _sig**6 + mu_coef[1] * _sig**12) * 1e36

    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    (_ax_nu, _ax_mu) = _ax
    _ax_nu.plot(_sig_um, _nu_fit, color="tab:blue",
                label=r"$a_6\sigma^6 + a_{12}\sigma^{12}$")
    _ax_nu.plot(_sigma_um, _nu_pts, "o", color="tab:blue", mfc="white",
                label="numerical")
    _ax_nu.set_xlabel(r"$\sigma$  [$10^{-6}$ m]")
    _ax_nu.set_ylabel(r"$\nu$  [$10^{-24}$]")
    _ax_nu.set_title(rf"$\nu(\sigma)$ at $U = {ui_U.value:.1f}\times10^{{-18}}$ J")
    _ax_nu.grid(True, linestyle=":", alpha=0.7)
    _ax_nu.legend()

    _ax_mu.plot(_sig_um, _mu_fit, color="tab:red",
                label=r"$b_6\sigma^6 + b_{12}\sigma^{12}$")
    _ax_mu.plot(_sigma_um, _mu_pts, "o", color="tab:red", mfc="white",
                label="numerical")
    _ax_mu.set_xlabel(r"$\sigma$  [$10^{-6}$ m]")
    _ax_mu.set_ylabel(r"$\mu$  [$10^{-36}$]")
    _ax_mu.set_title(rf"$\mu(\sigma)$ at $U = {ui_U.value:.1f}\times10^{{-18}}$ J")
    _ax_mu.grid(True, linestyle=":", alpha=0.7)
    _ax_mu.legend()

    mo.ui.matplotlib(_ax_nu)
    return


if __name__ == "__main__":
    app.run()
