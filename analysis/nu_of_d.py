# /// script
# dependencies = [
#     "h5py==3.16.0",
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "scipy==1.17.1",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Leading Taylor moment $\nu$ vs. correlation spacing $d$

    The force-closure kernel gives a leading moment

    $$
        \nu = \int_{-\infty}^{+\infty} x\,K(x)\,dx
            = \frac{2}{3}\int_{0}^{\infty} R^{3}\,g(R)\,f(R)\,dR,
        \qquad f(R) = -u'(R),
    $$

    where the second (single-integral) form is obtained by swapping the order
    of integration in $K(x)=x\int_{|x|}^{\infty} g\,f\,dR$. It is mathematically
    identical to $\int x K\,dx$ but **avoids the ~100x catastrophic cancellation**
    you hit when Riemann-summing the discrete stencil — so it stays accurate even
    near the sign flip.

    The sign of $\nu$ controls the large-scale (small-$k$) interaction stability:
    $\sigma(k)=C(-\nu k^2+\mu k^4)$, so $\nu<0$ permits clumping while $\nu>0$ is
    diffusive. That sign is set by where the correlation peak $d$ sits relative to
    the LJ force zero $r_{\min}=2^{1/6}\sigma$.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import simpson
    from scipy.optimize import brentq
    from red_patterns import Array1F
    from kernel import (
        lj_derivative,
        pdf_nearest_neighbor,
    )

    return brentq, lj_derivative, mo, np, pdf_nearest_neighbor, plt, simpson


@app.cell
def _():
    # Pair potential (matches config/params.txt: U = 1.1115e-16 J, sigma = 5.6 um)
    U = 1.1115e-16
    SIGMA = 5.6e-6

    # Nearest-neighbor pair distribution g(r) = g0 exp(-(r-d)^2 / (2 sigma_c^2))
    G0 = 4.0e7
    SIGMA_C = 0.5e-6
    EQ_DIST = 6.585467201064237091254725819933213415424688719213008880615234375e-06
    return EQ_DIST, G0, SIGMA, SIGMA_C, U


@app.cell
def _(G0, SIGMA, SIGMA_C, U, lj_derivative, np, pdf_nearest_neighbor, simpson):
    def nu_of_d(d: float, n: int = 300_001, lo: float = 3.0e-6, hi: float = 1.2e-5):
        """Accurate leading moment nu(d) via the cancellation-free single integral.

        g is a sharp Gaussian of width SIGMA_C centered at d, so the fixed grid
        [lo, hi] (here 3..12 um, ~16000 points per SIGMA_C) fully resolves the peak
        while g is negligible at both ends for d in the plotted range.
        """
        R = np.linspace(lo, hi, n)
        g = pdf_nearest_neighbor(R, G0, d, SIGMA_C)
        f = -lj_derivative(R, U, SIGMA)  # interaction force = -u'(R)
        return (2.0 / 3.0) * float(simpson(R**3 * g * f, x=R))

    return (nu_of_d,)


@app.cell
def _(G0, SIGMA, SIGMA_C, U, lj_derivative, np, pdf_nearest_neighbor, simpson):
    def mu_of_d(d: float, n: int = 300_001, lo: float = 3.0e-6, hi: float = 1.2e-5):
        """Accurate third-order moment mu(d) via the cancellation-free single integral

            mu = (1/6) int x^3 K(x) dx = (1/15) int_0^inf R^5 g(R) f(R) dR.

        Same R^5-weighted version of nu_of_d; mu must stay negative for the
        sigma(k) = C(-nu k^2 + mu k^4) high-k term to be stabilizing.
        """
        R = np.linspace(lo, hi, n)
        g = pdf_nearest_neighbor(R, G0, d, SIGMA_C)
        f = -lj_derivative(R, U, SIGMA)  # interaction force = -u'(R)
        return (1.0 / 15.0) * float(simpson(R**5 * g * f, x=R))

    return (mu_of_d,)


@app.cell
def _(SIGMA, mu_of_d, np, nu_of_d):
    R_MIN = 2.0 ** (1.0 / 6.0) * SIGMA  # LJ force zero / potential minimum

    d_sweep = np.linspace(6.0e-6, 7.2e-6, 241)
    nu_sweep = np.array([nu_of_d(d) for d in d_sweep])
    mu_sweep = np.array([mu_of_d(d) for d in d_sweep])

    def _zero_crossing(values):
        """First nu/mu = 0 location in d_sweep by linear interpolation (or None)."""
        cross = np.where(np.diff(np.sign(values)) != 0)[0]
        if not cross.size:
            return None
        i = cross[0]
        return d_sweep[i] - values[i] * (d_sweep[i + 1] - d_sweep[i]) / (
            values[i + 1] - values[i]
        )

    d_zero = _zero_crossing(nu_sweep)
    d_zero_mu = _zero_crossing(mu_sweep)
    return R_MIN, d_sweep, d_zero, d_zero_mu, mu_sweep, nu_sweep


@app.cell
def _(EQ_DIST, R_MIN, d_sweep, d_zero, mo, nu_sweep, plt):
    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.plot(d_sweep * 1e6, nu_sweep, color="blue", linewidth=2, label=r"$\nu(d)$")
    _ax.axhline(0.0, color="black", linewidth=1)

    # Shade the clumping-permitting region (nu < 0).
    _ax.fill_between(
        d_sweep * 1e6,
        nu_sweep,
        0.0,
        where=(nu_sweep < 0),
        color="red",
        alpha=0.12,
        label=r"$\nu<0$: clumping permitted",
    )

    _ax.axvline(
        R_MIN * 1e6,
        color="green",
        linestyle="--",
        label=rf"$r_{{\min}}=2^{{1/6}}\sigma={R_MIN * 1e6:.3f}\,\mu$m",
    )
    _ax.axvline(
        EQ_DIST * 1e6,
        color="red",
        linestyle=":",
        label=rf"current $d={EQ_DIST * 1e6:.3f}\,\mu$m",
    )
    if d_zero is not None:
        _ax.axvline(
            d_zero * 1e6,
            color="purple",
            linestyle="-.",
            label=rf"$\nu=0$ at $d={d_zero * 1e6:.4f}\,\mu$m",
        )

    _ax.set_xlabel(r"correlation peak $d$ ($\mu$m)", fontsize=12)
    _ax.set_ylabel(r"$\nu(d)$", fontsize=12)
    _ax.set_title(r"Leading Taylor moment $\nu$ vs. correlation spacing $d$", fontsize=14)
    _ax.grid(True, linestyle=":", alpha=0.7)
    _ax.legend()

    mo.ui.matplotlib(_ax)
    return


@app.cell
def _(EQ_DIST, R_MIN, d_sweep, d_zero_mu, mo, mu_sweep, plt):
    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.plot(d_sweep * 1e6, mu_sweep, color="darkorange", linewidth=2, label=r"$\mu(d)$")
    _ax.axhline(0.0, color="black", linewidth=1)

    # Shade the region where mu > 0: the high-k term +mu*k^4 then destabilizes.
    _ax.fill_between(
        d_sweep * 1e6,
        mu_sweep,
        0.0,
        where=(mu_sweep > 0),
        color="red",
        alpha=0.12,
        label=r"$\mu>0$: high-$k$ unstable",
    )

    _ax.axvline(
        R_MIN * 1e6,
        color="green",
        linestyle="--",
        label=rf"$r_{{\min}}=2^{{1/6}}\sigma={R_MIN * 1e6:.3f}\,\mu$m",
    )
    _ax.axvline(
        EQ_DIST * 1e6,
        color="red",
        linestyle=":",
        label=rf"current $d={EQ_DIST * 1e6:.3f}\,\mu$m",
    )
    if d_zero_mu is not None:
        _ax.axvline(
            d_zero_mu * 1e6,
            color="purple",
            linestyle="-.",
            label=rf"$\mu=0$ at $d={d_zero_mu * 1e6:.4f}\,\mu$m",
        )

    _ax.set_xlabel(r"correlation peak $d$ ($\mu$m)", fontsize=12)
    _ax.set_ylabel(r"$\mu(d)$", fontsize=12)
    _ax.set_title(
        r"Third-order Taylor moment $\mu$ vs. correlation spacing $d$", fontsize=14
    )
    _ax.grid(True, linestyle=":", alpha=0.7)
    _ax.legend()

    mo.ui.matplotlib(_ax)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Which $d$ reproduces the `config/params.txt` moments?

    The config sweep uses the coarse-stencil values $\nu=-2.55\times10^{-30}$ and
    $\mu=-4.47\times10^{-37}$. Solve $\nu(d)=\nu_{\text{target}}$ and
    $\mu(d)=\mu_{\text{target}}$ **independently** (each moment depends on $d$ alone),
    then check whether a single $d$ can satisfy both. Because $\nu$ is a ~100x
    cancellation its target sits a hair below the $\nu=0$ crossing and is extremely
    $d$-sensitive; $\mu$ is well-conditioned, so its $d$ is robust. If the two $d$
    values disagree, no physical spacing reproduces the config pair — confirming the
    config $\nu$ is fitted, not derived.
    """)
    return


@app.cell
def _(brentq, mu_of_d, np, nu_of_d):
    NU_TARGET = -2.55e-30
    MU_TARGET = -4.47e-37

    def _solve_for_target(func, target, lo=6.0e-6, hi=7.2e-6, n=601):
        """All d in [lo, hi] with func(d) == target, via sign-change bracketing + brentq."""
        d_grid = np.linspace(lo, hi, n)
        vals = np.array([func(d) for d in d_grid]) - target
        brackets = np.where(np.diff(np.sign(vals)) != 0)[0]
        roots = []
        for i in brackets:
            roots.append(
                brentq(lambda d: func(d) - target, d_grid[i], d_grid[i + 1], xtol=1e-13)
            )
        return roots

    d_for_nu = _solve_for_target(nu_of_d, NU_TARGET)
    d_for_mu = _solve_for_target(mu_of_d, MU_TARGET)
    return MU_TARGET, NU_TARGET, d_for_mu, d_for_nu


@app.cell
def _(MU_TARGET, NU_TARGET, d_for_mu, d_for_nu, mo, mu_of_d, nu_of_d):
    def _fmt(roots):
        if not roots:
            return "  (no d in [6.0, 7.2] um reproduces this target)\n"
        out = ""
        for d in roots:
            out += (
                f"  d = {d * 1e6:.9f} um  ->  "
                f"nu = {nu_of_d(d):+.4e},  mu = {mu_of_d(d):+.4e}\n"
            )
        return out

    _msg = (
        f"Target nu = {NU_TARGET:+.4e}:\n"
        + _fmt(d_for_nu)
        + f"\nTarget mu = {MU_TARGET:+.4e}:\n"
        + _fmt(d_for_mu)
    )

    if d_for_nu and d_for_mu:
        _gap = abs(d_for_nu[0] - d_for_mu[0]) * 1e6
        _msg += (
            f"\nGap between the nu- and mu-derived d: {_gap:.4f} um.\n"
            "If this gap is non-negligible vs SIGMA_C = 0.5 um, no single physical d\n"
            "produces the config (nu, mu) pair -- i.e. the config nu is a fit."
        )

    mo.md(f"```\n{_msg}\n```")
    return


if __name__ == "__main__":
    app.run()
