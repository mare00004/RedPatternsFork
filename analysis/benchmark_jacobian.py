# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo>=0.23.6", "numpy", "scipy", "matplotlib", "pandas", "h5py", "pydantic"]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="wide")


with app.setup:
    import json
    import sys
    from dataclasses import replace
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # uv's PEP 723 script environment does not install this repository package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from red_patterns.jacobian import (
        TaylorFVM, continuum_rates, dense_jacobian, derivative_sweep,
        discrete_rates, fixed_mass_basis, homogeneous_matrix, homogeneous_state,
        ridge_direction, ridge_rates, ridge_state,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Jacobian benchmark: homogeneous state and discrete ridge

    This notebook differentiates a float64 replica of the **actual Taylor face
    flux**, with clamped ghost cells and zero boundary fluxes. It compares
    independent matrix eigenvalues, analytic spectra and finite differences of
    the nonlinear RHS. The source is `src/cuda_kernel.cu`; the derivation and
    interpretation are in `docs/jacobian_benchmark.md`.

    The state is the stored population $U_{ji}=\Delta\rho\,\varphi_{ji}$, so
    $\psi_i=\sum_j U_{ji}$. The full domain length is $\ell=0.07$ m, not 0.06 m.
    Defaults: $\bar\psi=0.02$, $\nu=-4\times10^{-30}$,
    $\mu=-4\times10^{-37}$, $\beta=7.4\times10^{23}$,
    and ridge $\alpha=2\times10^{-5}$.

    All spectra below are **semidiscrete rates in s⁻¹**. Euler uses amplification
    $1+\Delta t\lambda$ and envelope rate $\log|1+\Delta t\lambda|/\Delta t$.
    """)
    return


@app.cell
def _():
    grid = mo.ui.dropdown([32, 64, 128, 256, 512], value=256, label="N (space and density)")
    concentration = mo.ui.number(start=0.001, stop=0.1, step=0.001, value=0.02, label="Mean ψ")
    nu_factor = mo.ui.slider(0.25, 2, step=0.25, value=1, label="Multiplier of ν = −4e−30", show_value=True)
    mo.hstack([grid, concentration, nu_factor])
    return concentration, grid, nu_factor


@app.cell
def _(concentration, grid, nu_factor):
    psi0 = concentration.value
    unforced = TaylorFVM(n=grid.value, nu=-4e-30*nu_factor.value)
    forced = replace(unforced, alpha=2e-5)
    modes = np.arange(unforced.n)
    theory = continuum_rates(unforced, psi0)
    exact = discrete_rates(unforced, psi0)
    scalar_matrix = homogeneous_matrix(unforced, psi0)
    scalar_eigenvalues = np.linalg.eigvalsh(scalar_matrix)
    cosines = np.cos(np.pi * (np.arange(unforced.n)[:, None]+0.5)*modes/unforced.n)
    cosine_action = scalar_matrix @ cosines
    numerical_mode_rates = np.sum(cosines*cosine_action, axis=0) / np.sum(cosines**2, axis=0)
    spectrum_table = pd.DataFrame({
        "mode": modes, "k_rad_m": modes*np.pi/unforced.length,
        "continuum_s^-1": theory, "discrete_formula_s^-1": exact,
        "jacobian_cosine_s^-1": numerical_mode_rates,
        "continuum_ridge_s^-1": continuum_rates(forced, psi0, ridge=True),
    })
    return (
        cosine_action, cosines, exact, forced, modes, numerical_mode_rates,
        psi0, scalar_eigenvalues, scalar_matrix, spectrum_table, theory, unforced,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For a spatially uniform unforced base, $v_0=0$ and adjacent populations
    agree. Thus the upwind switch has a unique first derivative:
    $\delta J_{jf}=-\beta U_{0,j}(T\delta\psi)_f$.
    Summing over density gives $\dot{\delta\psi}=\beta\bar\psi BT\delta\psi$.
    Cell-centered cosines diagonalize this matrix exactly:

    $$\lambda_m=\beta\bar\psi(-\nu k_m^2+\mu k_m^4),\qquad k_m=m\pi/\ell,$$
    $$\lambda_{h,m}=\beta\bar\psi\left[-\nu\widehat k_m^2
    \left(1+\frac{h^2\widehat k_m^2}{24}\right)+\mu\widehat k_m^4\right],
    \qquad\widehat k_m=\frac{2}{h}\sin\frac{m\pi}{2N}.$$

    The $h^2\widehat k^2/24$ term belongs to the implemented first derivative.
    The full $N^2$ matrix also has $N^2-N+1$ neutral modes generically; these
    include changes of composition invisible in $\psi$. The scalar eigenvalues
    shown here describe the marginal branch.
    """)
    return


@app.cell
def _(exact, modes, numerical_mode_rates, theory, unforced):
    homogeneous_figure, _axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for _ax in _axes:
        _ax.plot(modes, theory, label="Continuum")
        _ax.plot(modes, exact, "--", label="Exact discrete formula")
        _take = np.unique(np.linspace(0, unforced.n-1, min(unforced.n, 80), dtype=int))
        _ax.plot(modes[_take], numerical_mode_rates[_take], ".", label="Jacobian on cosines")
        _ax.axhline(0, color="grey", lw=0.7)
        _ax.set(xlabel="Cosine mode m", ylabel="Growth rate [s⁻¹]")
    _axes[0].set_title("All resolved modes")
    _cutoff = np.sqrt(unforced.nu/unforced.mu)*unforced.length/np.pi
    _axes[1].set(xlim=(0, min(unforced.n-1, 1.2*_cutoff)),
                 ylim=(-0.3*max(exact), 1.3*max(exact)), title="Instability band")
    _axes[1].legend()
    homogeneous_figure
    return (homogeneous_figure,)


@app.cell
def _(forced, psi0):
    ridge = ridge_state(forced, psi0)
    ridge_velocity = forced.velocity(ridge)
    transport_rates = ridge_rates(forced)
    constrained_rates = transport_rates[transport_rates < 0]
    # Numerically assemble and diagonalize the COMPLETE density-resolved matrix.
    small_forced = replace(forced, n=12)
    small_ridge = ridge_state(small_forced, psi0)
    full_ridge_matrix = dense_jacobian(small_forced, small_ridge)
    mass_basis = fixed_mass_basis(small_forced.n)
    reduced_ridge_matrix = mass_basis.T @ full_ridge_matrix @ mass_basis
    small_eigenvalues = np.linalg.eigvals(full_ridge_matrix)
    constrained_eigenvalues = np.linalg.eigvals(reduced_ridge_matrix)
    ridge_theory = continuum_rates(forced, psi0, ridge=True)[1:]
    return (
        constrained_eigenvalues, constrained_rates, full_ridge_matrix, mass_basis,
        reduced_ridge_matrix, ridge, ridge_theory, ridge_velocity,
        small_eigenvalues, small_forced, transport_rates,
    )


@app.cell
def _(psi0, unforced):
    _small_unforced = replace(unforced, n=8)
    _base = homogeneous_state(_small_unforced, psi0)
    _full = dense_jacobian(_small_unforced, _base)
    _expected = np.r_[discrete_rates(_small_unforced, psi0)[1:], np.zeros(57)]
    _observed = np.linalg.eigvals(_full)
    full_homogeneous_error = float(max(abs(np.sort(_observed.real)-np.sort(_expected))))
    return (full_homogeneous_error,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Forced one-bin ridge

    The production initializer puts density class $j$ into cell $r=N-1-j$.
    Its marginal is constant. At every interior face the upstream base
    population is zero and the velocity is nonzero, so
    $$\delta J=\underbrace{U_{\rm up,0}\delta v}_{0}+v_0\delta U_{\rm up}.$$
    Interactions disappear from this Jacobian. Each density block transports
    toward its occupied cell; ordering by distance to that cell makes it triangular.
    There is one mass-changing zero eigenvalue per class, and all remaining
    eigenvalues are negative. The fixed-mass spectral abscissa is exactly
    $$\max\operatorname{Re}\lambda=-\frac{30\alpha}{\ell(N-1)}.$$

    The continuum **displacement branch** instead predicts
    $$\sigma_m=-30\alpha/\ell+\beta\bar\psi(-\nu k_m^2+\mu k_m^4).$$
    These are different perturbation limits. Discrete transport eigenvalues
    have no one-to-one cosine-mode label. The plot compares their ranges and
    stability signs, not paired dispersion curves. The gap closes as $N$ grows;
    fixed-grid stability does not prove stability of the continuum limit.
    """)
    return


@app.cell
def _(constrained_rates, forced, ridge_theory):
    ridge_figure, _axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    _axes[0].hist(constrained_rates, bins=70)
    _axes[0].axvline(0, color="black", lw=1)
    _axes[0].set(xlabel="Eigenvalue [s⁻¹]", ylabel="Multiplicity", title="Full discrete ridge: fixed class masses")
    _axes[1].plot(np.arange(1, len(ridge_theory)+1), ridge_theory)
    _axes[1].axhline(0, color="black", lw=1)
    _axes[1].set(xlabel="Displacement mode m", ylabel="Growth rate [s⁻¹]", title="Continuum ridge displacement branch")
    _axes[1].set_xlim(0, min(forced.n-1, 1.2*np.sqrt(forced.nu/forced.mu)*forced.length/np.pi))
    _axes[1].set_ylim(-0.3*max(abs(ridge_theory.max()), 0.01), max(0.01, ridge_theory.max()*1.2))
    ridge_figure
    return (ridge_figure,)


@app.cell
def _(forced, psi0, unforced):
    # Keep finite-difference experiment at N=64 to expose a visible epsilon window.
    fd_model = replace(unforced, n=64)
    fd_base = homogeneous_state(fd_model, psi0)
    _q = np.random.default_rng(7).normal(size=fd_base.shape)
    _q -= _q.mean(axis=1, keepdims=True)
    _q /= np.max(np.abs(_q))
    _amplitudes = psi0 * np.logspace(-1, -12, 23)
    homogeneous_fd = pd.DataFrame(derivative_sweep(fd_model, fd_base, _q, _amplitudes))
    fd_forced = replace(forced, n=64)
    ridge_fd = pd.DataFrame(derivative_sweep(fd_forced, ridge_state(fd_forced, psi0),
                                            ridge_direction(fd_forced), _amplitudes))
    fd_figure, _axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for _ax, _table, _title in zip(_axes, [homogeneous_fd, ridge_fd], ["Homogeneous", "Ridge"]):
        _ax.loglog(_table.epsilon, _table.forward_relative_error, "o-", label="Forward difference")
        _ax.loglog(_table.epsilon, np.maximum(_table.centered_relative_error, 1e-18), "s-", label="Centered difference")
        _ax.set(xlabel="ε in U = U₀ + εq (max |q| = 1)", ylabel="Relative Jacobian-action error", title=_title+" (N=64)")
        _ax.legend()
    mo.vstack([fd_figure, mo.md("Ridge forward perturbations are nonnegative and conserve each class mass. "
        "Centered differences use a signed algebraic extension into negative empty-cell populations. "
        "Once face signs stay fixed, the ridge RHS is quadratic, so centered differences cancel its "
        "quadratic term exactly up to roundoff. The forward error is the more informative convergence check.")])
    return fd_figure, homogeneous_fd, ridge_fd


@app.cell
def _(forced, psi0, unforced):
    _rows = []
    for _n in [64, 128, 256, 512, 1024]:
        _model = replace(unforced, n=_n)
        _resolved = np.arange(1, 61)
        _continuum = continuum_rates(_model, psi0, _resolved)
        _discrete = discrete_rates(_model, psi0, _resolved)
        _rows.append({
            "N": _n, "max_abs_error_modes_1_60_s^-1": float(max(abs(_discrete-_continuum))),
            "mode_40_error_s^-1": float(abs(_discrete[39]-_continuum[39])),
            "discrete_fastest_mode": int(np.argmax(discrete_rates(_model, psi0))),
            "discrete_max_s^-1": float(max(discrete_rates(_model, psi0))),
            "ridge_fixed_mass_max_s^-1": -forced.alpha*30/(forced.length*(_n-1)),
        })
    refinement = pd.DataFrame(_rows)
    mo.vstack([mo.md("### Refinement at fixed physical modes\nErrors converge only when the physical mode is held fixed. "
                       "Modes near the grid cutoff need not agree with continuum theory."), mo.ui.table(refinement)])
    return (refinement,)


@app.cell
def _(
    constrained_eigenvalues, constrained_rates, cosine_action, cosines, exact,
    forced, full_homogeneous_error, full_ridge_matrix, homogeneous_fd, mass_basis, psi0,
    reduced_ridge_matrix, ridge, ridge_fd, ridge_theory, ridge_velocity,
    scalar_eigenvalues, small_eigenvalues, small_forced, unforced,
):
    _scale = max(abs(exact))
    _up = np.where(ridge_velocity > 0, ridge[:, :-1], ridge[:, 1:])
    _expected_small = np.sort(ridge_rates(small_forced).ravel())
    diagnostics = {
        "N": unforced.n, "psi": psi0, "nu": unforced.nu, "mu": unforced.mu,
        "homogeneous_eigenvalue_max_abs_error_s^-1": float(max(abs(np.sort(exact)-scalar_eigenvalues))),
        "homogeneous_cosine_residual_scaled": float(np.linalg.norm(cosine_action-cosines*exact)/(_scale*np.linalg.norm(cosines))),
        "homogeneous_small_grid_full_eigenvalue_max_abs_error_s^-1": full_homogeneous_error,
        "unforced_continuum_max_s^-1": float(max(continuum_rates(unforced, psi0))),
        "unforced_discrete_max_s^-1": float(max(exact)),
        "unforced_discrete_fastest_mode": int(np.argmax(exact)),
        "ridge_rhs_max_abs": float(np.max(np.abs(forced.rhs(ridge)))),
        "ridge_upstream_population_max_abs": float(np.max(np.abs(_up))),
        "ridge_min_interior_speed_m_s": float(np.min(np.abs(ridge_velocity))),
        "ridge_small_grid_full_eigenvalue_max_abs_error_s^-1": float(max(abs(np.sort(small_eigenvalues.real)-_expected_small))),
        "ridge_small_grid_constrained_eigenvalue_max_abs_error_s^-1": float(max(abs(np.sort(constrained_eigenvalues.real)-_expected_small[_expected_small < 0]))),
        "ridge_small_grid_max_imaginary_eigenvalue": float(max(abs(constrained_eigenvalues.imag))),
        "ridge_fixed_mass_invariance_residual": float(np.linalg.norm(full_ridge_matrix@mass_basis-mass_basis@reduced_ridge_matrix)),
        "ridge_constrained_max_s^-1": float(max(constrained_rates)),
        "ridge_continuum_max_s^-1": float(max(ridge_theory)),
        "ridge_continuum_fastest_mode": int(np.argmax(ridge_theory)+1),
        "homogeneous_best_forward_fd_relative_error": float(homogeneous_fd.forward_relative_error.min()),
        "ridge_best_forward_fd_relative_error": float(ridge_fd.forward_relative_error.min()),
    }
    mo.ui.table(pd.DataFrame([{"check": _k, "value": _v} for _k, _v in diagnostics.items()]))
    return (diagnostics,)


@app.cell
def _(diagnostics, fd_figure, homogeneous_fd, homogeneous_figure, refinement, ridge_fd, ridge_figure, spectrum_table):
    if mo.app_meta().mode == "script":
        _out = Path(__file__).resolve().parents[1] / "data" / "jacobian_benchmark"
        _out.mkdir(parents=True, exist_ok=True)
        for _name, _figure in [("homogeneous", homogeneous_figure), ("ridge", ridge_figure), ("finite_differences", fd_figure)]:
            _figure.savefig(_out / f"{_name}.png", dpi=180)
            _figure.savefig(_out / f"{_name}.pdf")
        for _name, _table in [("spectra", spectrum_table), ("refinement", refinement),
                              ("homogeneous_fd", homogeneous_fd), ("ridge_fd", ridge_fd)]:
            _table.to_csv(_out / f"{_name}.csv", index=False)
        (_out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
        print(json.dumps(diagnostics, indent=2))
        print(f"Saved benchmark figures, tables and diagnostics to {_out}")
    return


if __name__ == "__main__":
    app.run()
