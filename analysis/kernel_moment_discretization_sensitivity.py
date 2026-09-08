import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Why force-closure Taylor moments are stencil-sensitive

    This notebook isolates the force-closure, nearest-neighbor kernel.  It does
    **not** run the simulation.  Its question is whether the moments obtained
    from the finite kernel stencil exported to the convolution code can provide
    a stable route to the Taylor coefficients \(\nu\) and \(\mu\).

    The comparison is deliberately between two different quantities:

    * a high-accuracy continuum reference evaluated from cancellation-free
      radial integrals; and
    * the rectangular sums used on the sampled export stencil.

    The Taylor pair that best reproduces the convolution simulation is used as
    the calibration target throughout:

    $$
        (\nu_{\mathrm{conv}},\mu_{\mathrm{conv}})
        = (-2.975\times10^{-30},\,-5.213\times10^{-37}).
    $$

    A resolution-dependent discrete moment is not a reliable kernel-to-Taylor
    calibration, even when one particular coarse grid happens to be in the
    desired numerical range.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from red_patterns.kernel import (
        calculate_nu_mu,
        compute_force_closure_kernel,
        generate_kernel_stencil,
        guard_pair_distribution,
        lj_derivative,
        pdf_nearest_neighbor,
    )

    return (
        brentq,
        calculate_nu_mu,
        compute_force_closure_kernel,
        generate_kernel_stencil,
        guard_pair_distribution,
        lj_derivative,
        mo,
        np,
        pdf_nearest_neighbor,
        plt,
        quad,
    )


@app.cell
def _():
    # Physical kernel parameters: the force-closure nearest-neighbor defaults.
    U = 100e-18
    SIGMA = 5.6e-6
    G0 = 4.0e7
    D_BASELINE = 6.585467201064237e-6
    NN_SIGMA = 0.5e-6

    # Taylor pair empirically matched to the convolution result.
    NU_TARGET = -2.975e-30
    MU_TARGET = -5.213e-37

    # KernelSweep's 31-point export stencil: exporter spacing is dz / subdiv.
    COARSE_DZ_31 = 0.000267651
    SUBDIV_31 = 256
    H0 = COARSE_DZ_31 / SUBDIV_31
    HALF_WIDTH = 15.0 * H0
    # This is compute_force_closure_kernel's production default.
    RADIAL_SUB_RES = 10_000
    return (
        COARSE_DZ_31,
        D_BASELINE,
        G0,
        H0,
        MU_TARGET,
        NN_SIGMA,
        NU_TARGET,
        RADIAL_SUB_RES,
        SIGMA,
        SUBDIV_31,
        U,
    )


@app.cell(hide_code=True)
def _(COARSE_DZ_31, H0, SUBDIV_31, mo):
    mo.md(rf"""
    ## Explicit 31-point stencil spacing

    The baseline is the exported stencil from `KernelSweep(kernel_n=[31],
    dz=[{COARSE_DZ_31:.6e}], subdiv=[{SUBDIV_31}])`, not an arbitrary grid.
    Its sample spacing is

    $$
        h_0 = \frac{{\Delta z_{{\rm coarse}}}}{{\mathrm{{subdiv}}}}
            = \frac{{{COARSE_DZ_31:.6e}}}{{{SUBDIV_31}}}
            = {H0:.10e}\;\mathrm m.
    $$

    Refinement level \(m\) keeps the same physical half-width and uses
    \(h_m=h_0/2^m\) with \(N_m=2(15\,2^m)+1\) samples. Thus level zero is
    exactly the original 31-point exported stencil.
    """)
    return


@app.cell
def _(
    G0,
    H0,
    NN_SIGMA,
    RADIAL_SUB_RES,
    SIGMA,
    U,
    calculate_nu_mu,
    compute_force_closure_kernel,
    generate_kernel_stencil,
    guard_pair_distribution,
    lj_derivative,
    np,
    pdf_nearest_neighbor,
    quad,
):
    def force_times_pdf(r, d):
        """f(r) g(r), with f=-u', evaluated away from r=0."""
        r = np.asarray(r, dtype=np.float64)
        return -lj_derivative(r, U, SIGMA) * pdf_nearest_neighbor(r, G0, d, NN_SIGMA)

    def continuum_moments(d, upper=25e-6, epsrel=1e-10):
        """Cancellation-free continuum moments on a tail-negligible interval."""
        # Integrate dimensionless quantities: asking QUAD for an absolute
        # tolerance near 1e-40 otherwise triggers floating-point roundoff.
        # Splitting at the LJ force zero avoids cancellation inside QUAD itself.
        _nu_scale = 1e-27
        _mu_scale = 1e-37
        _force_zero = 2.0 ** (1.0 / 6.0) * SIGMA

        def _split_integral(_integrand):
            _left, _left_error = quad(
                _integrand, 3e-6, _force_zero, epsrel=epsrel, epsabs=1e-12, limit=500
            )
            _right, _right_error = quad(
                _integrand, _force_zero, upper, epsrel=epsrel, epsabs=1e-12, limit=500
            )
            return _left + _right, _left_error + _right_error

        nu_scaled, nu_error_scaled = _split_integral(
            lambda r: (2.0 / 3.0) * r**3 * force_times_pdf(r, d) / _nu_scale
        )
        mu_scaled, mu_error_scaled = _split_integral(
            lambda r: (1.0 / 15.0) * r**5 * force_times_pdf(r, d) / _mu_scale
        )
        return (
            float(nu_scaled * _nu_scale),
            float(mu_scaled * _mu_scale),
            float(nu_error_scaled * _nu_scale),
            float(mu_error_scaled * _mu_scale),
        )

    def stencil_moments(d, refinement):
        """Exactly the exporter-style sampled-kernel rectangular sums.

        Refinement halves h while choosing 2*(15*2**m)+1 points, so every
        stencil has the original physical half-width exactly.
        """
        factor = 2**refinement
        spacing = H0 / factor
        kernel_n = 2 * 15 * factor + 1
        x, kernel = generate_kernel_stencil(
            kernel_func=lambda sample_x: compute_force_closure_kernel(
                sample_x,
                u_prime_func=lambda r: lj_derivative(r, U, SIGMA),
                g_func=guard_pair_distribution(
                    lambda r: pdf_nearest_neighbor(r, G0, d, NN_SIGMA)
                ),
                sub_res=RADIAL_SUB_RES,
            ),
            kernel_n=kernel_n,
            kernel_dz=spacing,
        )
        nu, mu = calculate_nu_mu(x, kernel)
        return float(nu), float(mu), x, kernel

    return continuum_moments, force_times_pdf, stencil_moments


@app.cell(hide_code=True)
def _(mo, stencil_rows):
    _baseline = stencil_rows[0]
    mo.md(
        rf"""
    ## 31-point stencil and power-of-two refinements

    The actual exported 31-point stencil is \(m=0\):

    $$
        h_0={_baseline["spacing"]:.10e}\;\mathrm m,\qquad
        \nu_{{h_0}}={_baseline["nu"]:.6e},\qquad
        \mu_{{h_0}}={_baseline["mu"]:.6e}.
    $$

    It has the desired negative signs and is in the same order of magnitude as
    the convolution-matched Taylor pair. The three additional plotted spacings
    are \(h_0/2\), \(h_0/4\), and \(h_0/8\), each at the same physical support.
    """
    )
    return


@app.cell
def _(D_BASELINE, np, stencil_moments):
    refinement_levels = np.arange(4, dtype=int)
    stencil_rows = []
    for _level in refinement_levels:
        _nu, _mu, _x, _kernel = stencil_moments(D_BASELINE, int(_level))
        stencil_rows.append(
            {
                "level": int(_level),
                "spacing": float(_x[1] - _x[0]),
                "kernel_n": int(_x.size),
                "half_width": float(np.max(np.abs(_x))),
                "nu": _nu,
                "mu": _mu,
            }
        )
    return refinement_levels, stencil_rows


@app.cell
def _(np, stencil_rows):
    spacings = np.array([row["spacing"] for row in stencil_rows])
    nu_stencil = np.array([row["nu"] for row in stencil_rows])
    mu_stencil = np.array([row["mu"] for row in stencil_rows])
    nu_relative_error = np.abs((nu_stencil - nu_stencil[0]) / nu_stencil[0])
    mu_relative_error = np.abs((mu_stencil - mu_stencil[0]) / mu_stencil[0])
    return (
        mu_relative_error,
        mu_stencil,
        nu_relative_error,
        nu_stencil,
        spacings,
    )


@app.cell
def _(
    MU_TARGET,
    NU_TARGET,
    mo,
    mu_relative_error,
    mu_stencil,
    np,
    nu_relative_error,
    nu_stencil,
    plt,
    spacings,
):
    _moment_fig, _moment_axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    for _axis, _values, _target, _label in (
        (_moment_axes[0], nu_stencil, NU_TARGET, r"$\nu$"),
        (_moment_axes[1], mu_stencil, MU_TARGET, r"$\mu$"),
    ):
        _magnitudes = np.abs(_values)
        _axis.plot(spacings * 1e6, _magnitudes, "o-", label="export-stencil magnitude")
        _axis.axhline(
            _magnitudes[0],
            color="black",
            linestyle="--",
            label="31-point stencil magnitude",
        )
        _axis.axhline(abs(_target), color="tab:red", linestyle=":", label="convolution-matched magnitude")
        for _spacing, _magnitude, _value in zip(spacings, _magnitudes, _values, strict=True):
            _axis.annotate(
                "+" if _value >= 0.0 else "−",
                (_spacing * 1e6, _magnitude),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        _axis.set_xscale("log", base=2)
        _axis.set_yscale("log", base=10)
        _axis.set_xlabel(r"stencil spacing $h$ [$\mu$m]")
        _axis.set_ylabel(rf"$|{_label.strip('$')}|$ (log scale)")
        _axis.set_title(rf"Magnitude of {_label}; marker label gives its sign")
        _axis.grid(True, linestyle=":", alpha=0.7)
        _axis.legend(fontsize=8)

    _error_fig, _error_axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    for _axis, _errors, _label in (
        (_error_axes[0], nu_relative_error, r"relative drift from 31-point $\nu$"),
        (_error_axes[1], mu_relative_error, r"relative drift from 31-point $\mu$"),
    ):
        _axis.loglog(spacings * 1e6, _errors, "o-")
        _axis.set_xlabel(r"stencil spacing $h$ [$\mu$m]")
        _axis.set_ylabel(_label)
        _axis.grid(True, which="both", linestyle=":", alpha=0.7)
    mo.ui.matplotlib(_moment_axes[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sensitivity to the nearest-neighbor peak \(d\)

    The same force-closure kernel is also sensitive to its physical
    nearest-neighbor parameter \(d\). The curves below use the cancellation-free
    radial moment formulas, rather than a sampled stencil, so they isolate the
    physical \(d\)-dependence. In particular, \(\nu(d)\) changes sign near the
    Lennard–Jones force zero because repulsive and attractive contributions
    nearly cancel.
    """)
    return


@app.cell
def _(D_BASELINE, SIGMA, continuum_moments, np):
    d_sensitivity_grid = np.linspace(6.0e-6, 7.2e-6, 241)
    _moments_by_d = np.array([continuum_moments(_d)[:2] for _d in d_sensitivity_grid])
    nu_d_sweep = _moments_by_d[:, 0]
    mu_d_sweep = _moments_by_d[:, 1]
    r_force_zero = 2.0 ** (1.0 / 6.0) * SIGMA

    def _zero_crossing(_values):
        _crossings = np.flatnonzero(np.diff(np.sign(_values)) != 0)
        if not _crossings.size:
            return float("nan")
        _index = _crossings[0]
        return d_sensitivity_grid[_index] - _values[_index] * (
            d_sensitivity_grid[_index + 1] - d_sensitivity_grid[_index]
        ) / (_values[_index + 1] - _values[_index])

    d_zero_nu = _zero_crossing(nu_d_sweep)
    d_zero_mu = _zero_crossing(mu_d_sweep)
    return d_sensitivity_grid, d_zero_mu, d_zero_nu, mu_d_sweep, nu_d_sweep, r_force_zero


@app.cell
def _(
    D_BASELINE,
    MU_TARGET,
    NU_TARGET,
    d_sensitivity_grid,
    d_zero_mu,
    d_zero_nu,
    mo,
    mu_d_sweep,
    np,
    nu_d_sweep,
    plt,
    r_force_zero,
):
    _d_fig, _d_axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for _axis, _values, _target, _zero, _label in (
        (_d_axes[0], nu_d_sweep, NU_TARGET, d_zero_nu, r"$\nu$"),
        (_d_axes[1], mu_d_sweep, MU_TARGET, d_zero_mu, r"$\mu$"),
    ):
        _axis.plot(d_sensitivity_grid * 1e6, _values, linewidth=2, label=rf"{_label}$(d)$")
        _axis.axhline(0.0, color="black", linewidth=0.8)
        _axis.axhline(_target, color="tab:red", linestyle=":", label="convolution-matched target")
        _axis.axvline(D_BASELINE * 1e6, color="tab:red", linestyle="--", label=rf"KernelSweep $d={D_BASELINE * 1e6:.4f}\,\mu$m")
        _axis.axvline(r_force_zero * 1e6, color="tab:green", linestyle="--", label=rf"LJ force zero $={r_force_zero * 1e6:.4f}\,\mu$m")
        if np.isfinite(_zero):
            _axis.axvline(_zero * 1e6, color="tab:purple", linestyle="-.", label=rf"{_label}$=0$ at ${_zero * 1e6:.4f}\,\mu$m")
        _axis.set_xlabel(r"nearest-neighbor peak $d$ [$\mu$m]")
        _axis.set_ylabel(_label)
        _axis.set_title(rf"{_label} versus nearest-neighbor peak")
        _axis.grid(True, linestyle=":", alpha=0.7)
        _axis.legend(fontsize=7)
    mo.ui.matplotlib(_d_axes[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two-part conclusion

    ### 1. The 31-point stencil has valid *discretization-specific* effective moments

    The convolution simulation operates on the exported 31-point kernel, not
    directly on the analytical force-closure function. For that discrete
    operator, the Riemann sums

    $$
        \nu_{31}=h_0\sum_j x_j K_j,
        \qquad
        \mu_{31}=\frac{h_0}{6}\sum_j x_j^3K_j
    $$

    are its effective long-wavelength Taylor moments. Therefore it is expected
    that \((\nu_{31},\mu_{31})\) has the correct signs and roughly the correct
    scale to reproduce the 31-point convolution simulation. This is a useful
    mapping from a **specific exported stencil** to a Taylor model.

    ### 2. They are not grid-independent moments of the analytical kernel

    Refining only the representation of the same physical kernel changes the
    recovered \(\nu\) and \(\mu\). Thus the relation is not a unique,
    resolution-independent map from the analytical force-closure,
    nearest-neighbor kernel to the Taylor pair. In particular, the apparent
    agreement at 31 points should be interpreted as a property of that
    numerical stencil, not as a converged derivation of the continuum moments.

    The separate \(\nu(d)\) analysis strengthens this conclusion: at any fixed
    stencil, small changes in the nearest-neighbor peak \(d\) can produce large
    changes in \(\nu\) because attractive and repulsive contributions nearly
    cancel. Both effects make an unqualified kernel-to-Taylor calibration
    fragile.
    """)
    return


@app.cell
def _(D_BASELINE, force_times_pdf, np):
    radial_grid = np.linspace(1e-9, 20e-6, 200_001)
    nu_integrand = (2.0 / 3.0) * radial_grid**3 * force_times_pdf(radial_grid, D_BASELINE)
    mu_integrand = (1.0 / 15.0) * radial_grid**5 * force_times_pdf(radial_grid, D_BASELINE)
    positive_nu = float(np.trapezoid(np.maximum(nu_integrand, 0.0), x=radial_grid))
    negative_nu = float(np.trapezoid(np.minimum(nu_integrand, 0.0), x=radial_grid))
    cancellation_ratio = (positive_nu + abs(negative_nu)) / abs(positive_nu + negative_nu)
    positive_mu = float(np.trapezoid(np.maximum(mu_integrand, 0.0), x=radial_grid))
    negative_mu = float(np.trapezoid(np.minimum(mu_integrand, 0.0), x=radial_grid))
    mu_cancellation_ratio = (positive_mu + abs(negative_mu)) / abs(positive_mu + negative_mu)
    return (
        cancellation_ratio,
        negative_nu,
        negative_mu,
        mu_cancellation_ratio,
        mu_integrand,
        nu_integrand,
        positive_mu,
        positive_nu,
        radial_grid,
    )


@app.cell
def _(
    cancellation_ratio,
    mo,
    negative_nu,
    nu_integrand,
    plt,
    positive_nu,
    radial_grid,
):
    _cancellation_fig, _cancellation_axis = plt.subplots(figsize=(9, 4.5))
    _cancellation_axis.plot(radial_grid * 1e6, nu_integrand, color="tab:purple", label=r"$(2/3)R^3g(R)f(R)$")
    _cancellation_axis.fill_between(radial_grid * 1e6, nu_integrand, 0.0, where=nu_integrand > 0, alpha=0.25, label="repulsive contribution")
    _cancellation_axis.fill_between(radial_grid * 1e6, nu_integrand, 0.0, where=nu_integrand < 0, alpha=0.25, label="attractive contribution")
    _cancellation_axis.axhline(0.0, color="black", linewidth=0.8)
    _cancellation_axis.set_xlabel(r"radial distance $R$ [$\mu$m]")
    _cancellation_axis.set_ylabel(r"contribution to $\nu$")
    _cancellation_axis.set_title(r"$\nu$ is the residual of large, oppositely signed contributions")
    _cancellation_axis.grid(True, linestyle=":", alpha=0.7)
    _cancellation_axis.legend()

    _cancellation_text = mo.md(
        rf"""
    Positive contribution: \({positive_nu:.6e}\); negative contribution:
    \({negative_nu:.6e}\).  Their cancellation ratio is
    \({cancellation_ratio:.2f}\).  Thus a small sampling displacement of the
    force-zero region can change the residual \(\nu\) by orders of magnitude,
    even though the kernel itself changes smoothly.
    """
    )
    mo.vstack([mo.ui.matplotlib(_cancellation_axis), _cancellation_text])
    return


@app.cell
def _(
    mo,
    mu_cancellation_ratio,
    mu_integrand,
    negative_mu,
    plt,
    positive_mu,
    radial_grid,
):
    _mu_cancellation_fig, _mu_cancellation_axis = plt.subplots(figsize=(9, 4.5))
    _mu_cancellation_axis.plot(
        radial_grid * 1e6,
        mu_integrand,
        color="darkorange",
        label=r"$(1/15)R^5g(R)f(R)$",
    )
    _mu_cancellation_axis.fill_between(
        radial_grid * 1e6,
        mu_integrand,
        0.0,
        where=mu_integrand > 0,
        alpha=0.25,
        label="repulsive contribution",
    )
    _mu_cancellation_axis.fill_between(
        radial_grid * 1e6,
        mu_integrand,
        0.0,
        where=mu_integrand < 0,
        alpha=0.25,
        label="attractive contribution",
    )
    _mu_cancellation_axis.axhline(0.0, color="black", linewidth=0.8)
    _mu_cancellation_axis.set_xlabel(r"radial distance $R$ [$\mu$m]")
    _mu_cancellation_axis.set_ylabel(r"contribution to $\mu$")
    _mu_cancellation_axis.set_title(r"$\mu$ is the residual of signed contributions")
    _mu_cancellation_axis.grid(True, linestyle=":", alpha=0.7)
    _mu_cancellation_axis.legend()

    _mu_cancellation_text = mo.md(
        rf"""
    Positive contribution: \({positive_mu:.6e}\); negative contribution:
    \({negative_mu:.6e}\). Their cancellation ratio is
    \({mu_cancellation_ratio:.2f}\). The additional \(R^5\) weighting changes
    the balance of attractive and repulsive ranges relative to \(\nu\).
    """
    )
    mo.vstack([mo.ui.matplotlib(_mu_cancellation_axis), _mu_cancellation_text])
    return


@app.cell
def _(
    MU_TARGET,
    NU_TARGET,
    brentq,
    continuum_moments,
    refinement_levels,
    stencil_moments,
):
    def solve_d(moment, target, refinement=None):
        if refinement is None:
            evaluator = lambda d: continuum_moments(d)[0 if moment == "nu" else 1]
        else:
            evaluator = lambda d: stencil_moments(d, refinement)[0 if moment == "nu" else 1]
        _lo = 6.0e-6
        _hi = 7.2e-6
        _f_lo = evaluator(_lo) - target
        _f_hi = evaluator(_hi) - target
        if _f_lo == 0.0:
            return _lo
        if _f_hi == 0.0:
            return _hi
        if _f_lo * _f_hi > 0.0:
            return float("nan")
        return brentq(lambda d: evaluator(d) - target, _lo, _hi, xtol=1e-14)

    calibration_rows = []
    for _level in refinement_levels:
        _d_nu = solve_d("nu", NU_TARGET, int(_level))
        _d_mu = solve_d("mu", MU_TARGET, int(_level))
        calibration_rows.append(
            {
                "level": int(_level),
                "d_for_nu": _d_nu,
                "d_for_mu": _d_mu,
                "gap": abs(_d_nu - _d_mu),
            }
        )
    continuum_d_nu = solve_d("nu", NU_TARGET)
    continuum_d_mu = solve_d("mu", MU_TARGET)
    return calibration_rows, continuum_d_mu, continuum_d_nu


@app.cell
def _(calibration_rows, continuum_d_mu, continuum_d_nu, mo, np, plt):
    _levels = np.array([row["level"] for row in calibration_rows])
    _d_nu = np.array([row["d_for_nu"] for row in calibration_rows])
    _d_mu = np.array([row["d_for_mu"] for row in calibration_rows])
    _gaps = np.array([row["gap"] for row in calibration_rows])

    _calibration_fig, _calibration_axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    _calibration_axes[0].plot(_levels, _d_nu * 1e6, "o-", label=r"$d$ matching target $\nu$")
    _calibration_axes[0].plot(_levels, _d_mu * 1e6, "s-", label=r"$d$ matching target $\mu$")
    _calibration_axes[0].axhline(continuum_d_nu * 1e6, color="tab:blue", linestyle=":")
    _calibration_axes[0].axhline(continuum_d_mu * 1e6, color="tab:orange", linestyle=":")
    _calibration_axes[0].set_xlabel("refinement level")
    _calibration_axes[0].set_ylabel(r"inferred nearest-neighbor peak $d$ [$\mu$m]")
    _calibration_axes[0].set_title("Target-derived kernel parameter drifts with stencil resolution")
    _calibration_axes[0].grid(True, linestyle=":", alpha=0.7)
    _calibration_axes[0].legend(fontsize=8)

    _calibration_axes[1].plot(_levels, _gaps * 1e6, "o-", color="tab:red")
    _calibration_axes[1].axhline(abs(continuum_d_nu - continuum_d_mu) * 1e6, color="black", linestyle=":", label="continuum gap")
    _calibration_axes[1].set_xlabel("refinement level")
    _calibration_axes[1].set_ylabel(r"$|d_\nu-d_\mu|$ [$\mu$m]")
    _calibration_axes[1].set_title("Gap where both target-derived roots exist")
    _calibration_axes[1].grid(True, linestyle=":", alpha=0.7)
    _calibration_axes[1].legend()

    _interpretation = mo.md(
        rf"""
    ## Interpretation

    In the continuum limit, the target \(\nu\) requires
    \(d={continuum_d_nu * 1e6:.6f}\,\mu\mathrm m\), whereas target \(\mu\)
    requires \(d={continuum_d_mu * 1e6:.6f}\,\mu\mathrm m\).  Their separation
    is \({abs(continuum_d_nu - continuum_d_mu) * 1e6:.4f}\,\mu\mathrm m\).

    Thus this fixed force-closure nearest-neighbor family has no stable
    one-parameter \(d \mapsto (\nu,\mu)\) calibration for the required pair.
    A missing coarse-grid marker means that target is not attainable within the
    chosen physical \(d\) bracket at that resolution. Plausible coarse-stencil
    values are not invariant under a representation change that leaves the
    physical kernel unchanged.
    """
    )
    mo.vstack([mo.ui.matplotlib(_calibration_axes[0]), _interpretation])
    return


if __name__ == "__main__":
    app.run()
