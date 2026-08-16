# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.0",
#     "numpy==2.5.1",
#     "scipy==1.18.0",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import numpy as np
    from numpy.linalg import solve, cond
    import matplotlib.pyplot as plt
    from scipy.optimize import least_squares
    from scipy.integrate import quad

    def latex_scientific(value: float) -> str:
        if value == 0:
            return "0"
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10**exponent)
        return rf"{mantissa:.6f} \cdot 10^{{{exponent}}}"

    return cond, latex_scientific, mo, np, plt, quad, solve


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hypernetted Chain Model (HNC)

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model equation for the effective potential is

    $$
        u_\text{eff}(r) = a (e^{-\frac{u(r)}{b}} - 1) - c u(r)
    $$

    with the parameters $a, b, c \in \mathbb{R}$. And isolated potential $u(r)$ of a single, isolated, RBC.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## $\nu$- and $\mu$-Calculation

    As always

    $$
        \begin{align*}
        &\nu = \int_{-\infty}^{+\infty} K(\eta) \eta \; d \eta\\
        &\mu = \frac{1}{3!}\int_{-\infty}^{+\infty} K(\eta)\eta^3 \; d \eta
        \end{align*}
    $$
    """)
    return


@app.cell
def _(kernel_len, kernel_mor, np, x):
    nu_len = np.trapezoid(kernel_len * x, x)
    mu_len = (1 / 6) * np.trapezoid(kernel_len * np.power(x, 3), x)

    nu_mor = np.trapezoid(kernel_mor * x, x)
    mu_mor = (1 / 6) * np.trapezoid(kernel_mor * np.power(x, 3), x)
    return mu_len, mu_mor, nu_len, nu_mor


@app.cell(hide_code=True)
def _(latex_scientific, mo, mu_len, mu_mor, nu_len, nu_mor):
    mo.md(rf"""
    For the Lennard-Jones potential we get

    $$
        \nu = {latex_scientific(nu_len)}, \quad \mu = {latex_scientific(mu_len)}
    $$

    For the Morse potential we get

    $$
        \nu = {latex_scientific(nu_mor)}, \quad \mu = {latex_scientific(mu_mor)}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deriving parameters from known $\nu$ and $\mu$

    First of all we have to choose physically meaningfull parameters for the Lennard-Jones- and Morse-Potential.

    For the Lennard-Jones Potentail we already now that, if written in the form

    $$
        u(r) = 4 U \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]
    $$
    """)
    return


@app.cell(hide_code=True)
def _(SIGMA, U, latex_scientific, mo):
    mo.md(rf"""
    the parameters have to be $U = {latex_scientific(U)} \text{{J}}$ and $\sigma = {latex_scientific(SIGMA)} \text{{m}}$. The relation between those parameters and our parameters are $\alpha = U$ and $\beta = \sqrt[6]{2} \sigma$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the Morse-Potential we require that its minimum coincides with that of the Leannard-Jones Potential. This fixes $\alpha_\text{Morse} = \alpha_\text{LJ}$ and $\beta_\text{Morse} = \beta_\text{LJ}$ and leaves $\gamma_\text{Morse}$ free.
    """)
    return


@app.cell
def _(np):
    U = 100e-18 # in J
    SIGMA = 5.6e-6 # in m

    a_len = U
    b_len = np.power(2, 1 / 6) * SIGMA

    a_mor = a_len
    c_mor = b_len
    b_mor = 1
    return SIGMA, U, a_len, b_len


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Idea now is to start with a known set of $\nu$ and $\mu$ values and try to fix the free parameters $b_\text{Morse}, a, b, c$ of the effective potential, in such a way that using the derived Kernel to calcualte $\nu$ and $\mu$ gives the same result.
    """)
    return


@app.cell
def _(a_len, b_len, kernel, np, pot_lennard_jones, quad):
    def compute_nu_mu(a, b, c, limit=5000):
        _u_len = lambda t: pot_lennard_jones(t, a_len, b_len)
        _K = lambda t: kernel(t, _u_len, a, b, c)

        # Integrals are symmetrical
        nu, _ = quad(lambda eta: _K(eta) * eta, 0, np.inf, limit=limit)
        mu, _ = quad(lambda eta: _K(eta) * eta**3, 0, np.inf, limit=limit)

        nu *= 2
        mu /= 3

        return nu, mu

    def residuals(params, nu_target, mu_target):
        a, b, c = params
        try:
            nu, mu = compute_nu_mu(a, b, c)
            return [nu - nu_target, mu - mu_target]
        except:
            return [1e100, 1e100]

    return


@app.cell
def _(np):
    NU = -2.832638e-30 / (2 * np.pi)
    MU = -4.468455e-37 / (2 * np.pi)
    return MU, NU


@app.cell
def _(MU, NU, a_len, b_len, cond, np, quad, solve):
    def q_of_y(y, beta):
        """
        q(y) = exp(-u(y)/beta)
        with r = b_len * y -> Stability
        u(y) = a_len * phi(y)
        """
        def phi(y):
            if y == 0:
                return np.inf
            inv = 1.0 / y
            return inv**12 - 2.0 * inv**6

        if y == 0:
            return 0.0

        z = -a_len * phi(y) / beta

        if z < -745:
            return 0.0
        if z > 700:
            return np.exp(700)

        return np.exp(z)

    def integrate_y(f, limit=500):
        y0 = 2 ** (-1 / 6)

        val1, _ = quad(f, 0, y0, limit=limit)
        val2, _ = quad(f, y0, 1, limit=limit)
        val3, _ = quad(f, 1, np.inf, limit=limit)

        return val1 + val2 + val3


    def moment_matrix(beta):
        """
        Returns the 2x2 matrix M(beta) such that

            [NU] = M [alpha]
            [MU]     [d    ]

        where d = beta * gamma.
        """

        L = b_len

        def q(y):
            return q_of_y(y, beta)

        A_nu = L**3 * integrate_y(
            lambda y: y**2 * (q(y)**2 - 1.0)
        )

        D_nu = 2 * L**3 * integrate_y(
            lambda y: y**2 * (q(y) - 1.0)
        )

        A_mu = (L**5 / 6) * integrate_y(
            lambda y: y**4 * (q(y)**2 - 1.0)
        )

        D_mu = (L**5 / 3) * integrate_y(
            lambda y: y**4 * (q(y) - 1.0)
        )

        return np.array([
            [A_nu, D_nu],
            [A_mu, D_mu]
        ])

    def initial_guess_from_beta(beta):
        M = moment_matrix(beta)
        rhs = np.array([NU, MU])

        alpha, d = solve(M, rhs)
        gamma = d / beta

        return alpha, beta, gamma, cond(M)

    return (initial_guess_from_beta,)


@app.cell
def _(
    a_len,
    b_len,
    initial_guess_from_beta,
    kernel,
    latex_scientific,
    mo,
    np,
    plt,
    pot_effective,
    pot_lennard_jones,
):
    alpha, beta, gamma, _ = initial_guess_from_beta(2e-6)

    _x = np.linspace(-5e-6, 5e-6, 500)
    _u_len = lambda t: pot_lennard_jones(t, a_len, b_len)
    _K = lambda t: kernel(t, _u_len, alpha, beta, gamma)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 8))

    _ax1.plot(_x, _K(_x), "g-", label="Kernel")

    _x_pos = np.linspace(0.5 * b_len, 1e-6, 500)
    _u_eff = pot_effective(_u_len(_x_pos), alpha, beta, gamma)
    _ax2.plot(_x_pos, _u_eff, "g-", label="Kernel")

    _md = mo.md(rf"$$\alpha = {latex_scientific(alpha)}, \quad \beta = {latex_scientific(beta)}, \quad \gamma = {latex_scientific(gamma)}$$")

    print(alpha, beta, gamma)

    # mo.vstack([mo.ui.matplotlib(_ax1), _md], align="center")
    mo.ui.matplotlib(_ax1)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pair Potential

    The potentials were defined in a way that the minima align.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lennard-Jones Potential

    The Lennard-Jones potential is given by

    $$
        u(r) = \alpha \left[ \left( \frac{\beta}{r} \right)^{12} - 2 \left( \frac{\beta}{r} \right)^{6}\right]
    $$

    with $\alpha, \beta \in \mathbb{R}$. The minimum is at $r_\text{min} = \beta$ with the value $u(r_\text{min}) = -\alpha$. If you write the Lennard-Jones potential in its usual form $u(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^{6}\right]$, you get $\alpha = \varepsilon$ and $\beta = \sqrt[6]{2} \sigma$.
    """)
    return


@app.cell
def _(np):
    def pot_lennard_jones(
        r: np.ndarray, a: np.floating, b: np.floating
    ) -> np.ndarray:
        return a * (np.power(b / r, 12) - 2 * np.power(b / r, 6))

    return (pot_lennard_jones,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Morse Potential

    The Morse potential is given by

    $$
        u(r) = \alpha (1 - e^{-\gamma(r - \beta)})^2 - \alpha
    $$

    with $\alpha, \beta, \gamma \in \mathbb{R}$. The minimum is at $r_\text{min} = \beta$ with the value $u(r_\text{min}) = -\alpha$.
    """)
    return


@app.cell
def _(np):
    def pot_morse(r: np.ndarray, a: np.floating, b: np.floating, c: np.floating) -> np.ndarray:
        return a * np.power(1 - np.exp(-c * (r - b)), 2) - a

    return (pot_morse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparision

    Below you can see both of the pair potential as well as the pair distributions that we would get from $u(r) = - \varepsilon \ln g(r)$. With $\varepsilon = 1$.
    """)
    return


@app.cell
def _(mo):
    ui_alpha = mo.ui.slider(-10, 10, 0.1, value=2, label=r"$\alpha$", show_value=True)
    ui_beta = mo.ui.slider(-10, 10, 0.1, value=2, label=r"$\beta$", show_value=True)
    ui_gamma = mo.ui.slider(-10, 10, 0.1, value=2, label=r"$\gamma$", show_value=True)
    return ui_alpha, ui_beta, ui_gamma


@app.cell
def _(mo, ui_alpha, ui_beta, ui_gamma):
    mo.hstack([ui_alpha, ui_beta, ui_gamma])
    return


@app.cell
def _(pot_lennard_jones, pot_morse, ui_alpha, ui_beta, ui_gamma):
    # Pair Potential Functions Helpers with all relevant parameters set to the corresponding ui values
    pot_len_ui = lambda t: pot_lennard_jones(t, ui_alpha.value, ui_beta.value)
    pot_mor_ui = lambda t: pot_morse(t, ui_alpha.value, ui_beta.value, ui_gamma.value)
    return (pot_mor_ui,)


@app.cell
def _(mo, np, plt, pot_lennard_jones, pot_morse, ui_alpha, ui_beta, ui_gamma):
    fig, (ax_pot, ax_g) = plt.subplots(1, 2, figsize=(10,5))

    r = np.linspace(1, 10, 1000)

    _alpha = ui_alpha.value
    _beta = ui_beta.value
    _gamma = ui_gamma.value

    pot_len = pot_lennard_jones(r, _alpha, _beta)
    pot_mor = pot_morse(r, _alpha, _beta, _gamma)

    u_len = lambda t: pot_lennard_jones(t, _alpha, _beta)
    u_mor = lambda t: pot_morse(t, _alpha, _beta, _gamma)

    u_mor_max = _alpha * ((1 - np.exp(_beta * _gamma))**2) - _alpha

    ##############################

    ax_pot.plot(r, pot_len, "r-", label="Lennard-Jones Potential")
    ax_pot.plot(r, pot_mor, "b-", label="Morse Potential")

    ax_pot.set_xlim([1, 5])
    ax_pot.set_ylim([-2 * _alpha, 2*_alpha])
    ax_pot.hlines([0], xmin=0, xmax=8, colors="grey", linestyles="dashed")
    # ax_pot.vlines([0], ymin=-2*_alpha, ymax=+2*_alpha, colors="red", linestyles="dashed")
    ax_pot.legend()
    ax_pot.set_title("Pair Potential")
    ax_pot.set_xlabel("r")
    ax_pot.set_xlabel("u(r)")
    ax_pot.set_box_aspect(1)


    #############################

    ep = 1

    g_len = np.exp(- pot_len / ep)
    g_mor = np.exp(- pot_mor / ep)


    ax_g.plot(r, g_len, "r-", label="PDF (Lennard-Jones)")
    ax_g.plot(r, g_mor, "b-", label="PDF (Morse)")

    ax_g.hlines([1], xmin=r[0], xmax=r[-1], colors="grey", linestyles="dashed")

    ax_g.set_title("Pair Distribution Functions")
    ax_g.legend()

    ax_g.set_xlabel("r")
    ax_g.set_ylabel("g(r)")
    ax_g.set_box_aspect(1)

    mo.ui.matplotlib(ax_pot)
    return pot_len, pot_mor, r


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effective Potential and resulting Kernel

    As stated above the effective potential will be modeled as

    $$
        u_\text{eff}[u](r) = a (e^{-\frac{u(r)}{b}} - 1) - c \, u(r)
    $$

    with the parameters $a, b, c \in \mathbb{R}$. See `Obsidian > RedPatternsHNC.md`.
    """)
    return


@app.cell
def _(np):
    def pot_effective(
        pot: np.ndarray, a: np.floating, b: np.floating, c: np.floating
    ) -> np.ndarray:
        return a * (np.exp(-pot / b) - 1) - c * pot

    return (pot_effective,)


@app.cell
def _(mo):
    ui_a = mo.ui.slider(-10, 10, 0.1, value=2, label="$a$", show_value=True)
    ui_b = mo.ui.slider(-10, 10, 0.1, value=2, label="$b$", show_value=True)
    ui_c = mo.ui.slider(-10, 10, 0.1, value=2, label="$c$", show_value=True)
    return ui_a, ui_b, ui_c


@app.cell
def _(mo, ui_a, ui_b, ui_c):
    mo.hstack([ui_a, ui_b, ui_c])
    return


@app.cell
def _(
    mo,
    np,
    plt,
    pot_effective,
    pot_len,
    pot_mor,
    r,
    ui_a,
    ui_alpha,
    ui_b,
    ui_beta,
    ui_c,
):
    _fig, _ax = plt.subplots(figsize=(8, 6))

    _pot_eff_len = pot_effective(pot_len, ui_a.value, ui_b.value, ui_c.value)
    _pot_eff_mor = pot_effective(pot_mor, ui_a.value, ui_b.value, ui_c.value)


    _eff_max = ui_a.value * (np.exp(ui_alpha.value / ui_b.value) - 1) + ui_c.value * ui_alpha.value
    _ylim = 1.2 * _eff_max

    _ax.plot(r, _pot_eff_len, "r-", label="Effective Potential (Lennard-Jones)")
    _ax.plot(r, _pot_eff_mor, "b-", label="Effective Potential (Morse)")

    _ax.hlines([0], xmin=0, xmax=6, color="grey", linestyle="dashed")
    _ax.vlines([ui_beta.value], ymin=-_ylim, ymax=_ylim, colors="grey", linestyles="dashed")

    _ax.set_xlim([0, 6])
    _ax.set_ylim([-_ylim, _ylim])
    _ax.legend()

    ###########
    # Add $\beta$ x tick
    # See: https://matplotlib.org/stable/users/explain/axes/axes_ticks.html#tick-objects
    _beta = ui_beta.value
    _xticks = _ax.get_xticks()

    try:
        i = _xticks.index(_beta)[0]
    except:
        i = int(np.where(_beta >= _xticks)[0][-1]) + 1
        _xticks = np.concatenate([_xticks[:i], [_beta], _xticks[i:]])

    # Build label list — format numeric ticks, use LaTeX for the beta tick
    _labels = [f"{t:.1f}" for t in _xticks]
    _labels[i] = r"$\beta$"

    _ax.set_xticks(_xticks)
    _ax.set_xticklabels(_labels)
    ###########

    mo.ui.matplotlib(_ax)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kernel

    The interaction kernel will be,

    $$
        K(x) = x \; u_{\text{eff}}(|x|)
    $$
    """)
    return


@app.cell
def _(np):
    def kernel(x: np.ndarray, pot: np.ufunc):
        return x * pot(np.abs(x))

    return (kernel,)


@app.cell
def _(mo, ui_a, ui_alpha, ui_b, ui_beta, ui_c, ui_gamma):
    mo.vstack(
        [
            mo.hstack([ui_a, ui_b, ui_c]),
            mo.hstack([ui_alpha, ui_beta, ui_gamma]),
        ]
    )
    return


@app.cell
def _(
    kernel,
    mo,
    np,
    plt,
    pot_effective,
    pot_lennard_jones,
    pot_morse,
    ui_a,
    ui_alpha,
    ui_b,
    ui_beta,
    ui_c,
    ui_gamma,
):
    _fig, _ax = plt.subplots(figsize=(8, 6))

    x = np.linspace(-10, 10, 2000) 

    fn_pot_len = lambda t: pot_lennard_jones(t, ui_alpha.value, ui_beta.value)
    fn_pot_mor = lambda t: pot_morse(t, ui_alpha.value, ui_beta.value, ui_gamma.value)

    fn_eff_pot_len = lambda t: pot_effective(fn_pot_len(t), ui_a.value, ui_b.value, ui_c.value)
    fn_eff_pot_mor = lambda t: pot_effective(fn_pot_mor(t), ui_a.value, ui_b.value, ui_c.value)

    kernel_len = kernel(x, fn_eff_pot_len)
    kernel_mor = kernel(x, fn_eff_pot_mor)

    _ax.plot(x, kernel_len, "r-", label="Kernel (Lennard-Jones)")
    _ax.plot(x, kernel_mor, "b-", label="Kernel (Morse)")

    _ax.hlines([0], xmin=-10, xmax=10, color="grey", linestyle="dashed")

    # _xmax = 2 * ui_beta.value
    _xmax = 10
    _ymax = 1.2 * np.max(kernel_mor)

    _ax.set_xlim([-_xmax , +_xmax])
    _ax.set_ylim([-_ymax, _ymax])
    _ax.legend()

    mo.ui.matplotlib(_ax)
    return kernel_len, kernel_mor, x


@app.cell(column=3)
def _(mo):
    mo.md(r"""
    ## Derive $\nu$ and $\mu$

    Given the Morse potential $u(r) = \alpha (1 - e^{-\gamma(r - \beta)})^2 - \alpha$ and a fixed effective potential parameter $b$, we can solve for $a$ and $c$ analytically via a linear system.

    The moments (including the factor of 2 from the symmetric $(-\infty, +\infty)$ integration) are:

    $$
        M_n^{(e)}(b) = 2 \int_0^\infty dr \, r^n \left( e^{-u(r)/b} - 1 \right), \qquad
        M_n^{(u)} = 2 \int_0^\infty dr \, r^n \, u(r)
    $$

    The linear system to solve is:

    $$
        \begin{pmatrix} \nu \\ 6\mu \end{pmatrix} =
        \begin{pmatrix} M_2^{(e)}(b) & -M_2^{(u)} \\ M_4^{(e)}(b) & -M_4^{(u)} \end{pmatrix}
        \begin{pmatrix} a \\ c \end{pmatrix}
    $$
    """)
    return


@app.cell
def _(mo):
    ui_b_derive = mo.ui.text(label="$b$", value="1e-10")
    ui_nu_derive = mo.ui.text(label=r"$\nu$", value="-3e-30")
    ui_mu_derive = mo.ui.text(label=r"$\mu$", value="-4e-37")

    mo.hstack([
        ui_b_derive,
        ui_nu_derive,
        ui_mu_derive
    ])
    return ui_b_derive, ui_mu_derive, ui_nu_derive


@app.cell
def _(ui_b_derive, ui_mu_derive, ui_nu_derive):
    b_derive = float(ui_b_derive.value)
    nu_derive = float(ui_nu_derive.value)
    mu_derive = float(ui_mu_derive.value)
    return b_derive, mu_derive, nu_derive


@app.cell
def _(
    b_derive,
    latex_scientific,
    mo,
    mu_derive,
    np,
    nu_derive,
    solve_morse_system,
):
    # Use physical Morse parameters matching the LJ potential
    # alpha = a_len = U = 100e-18 J (depth)
    # beta  = b_len = 2^(1/6) * SIGMA ~ 6.29e-6 m (minimum position)
    # gamma is free — choose to match LJ curvature at the minimum:
    #   For LJ:  u''(r_min) = 72 * alpha / beta^2
    #   For Morse: u''(r_min) = 2 * alpha * gamma^2
    #   => gamma = sqrt(36 / beta^2) = 6 / beta

    alpha_test = 1e-16
    beta_test = np.pow(2, 1/6) * 5.6e-6
    gamma_test = 6.0 / beta_test

    a_sol, c_sol, M_mat, cond_num = solve_morse_system(
        alpha_test, beta_test, gamma_test, b_derive, nu_derive, mu_derive
    )

    mo.md(rf"""
    **Solved parameters** (Morse potential with $\gamma = 6/\beta$):

    $$
        a = {latex_scientific(a_sol)}, \quad c = {latex_scientific(c_sol)}
    $$

    **Moment matrix:**

    $$
        M = \begin{{pmatrix}}
        {latex_scientific(M_mat[0,0])} & {latex_scientific(M_mat[0,1])} \\
        {latex_scientific(M_mat[1,0])} & {latex_scientific(M_mat[1,1])}
        \end{{pmatrix}}
    $$

    **Condition number:** $\kappa = {latex_scientific(cond_num)}$
    """)
    return a_sol, alpha_test, beta_test, c_sol, gamma_test


@app.cell
def _(
    a_sol,
    alpha_test,
    b_derive,
    beta_test,
    c_sol,
    gamma_test,
    mo,
    plot_morse_analysis,
):
    _fig = plot_morse_analysis(
        alpha_m=alpha_test,
        beta_m=beta_test,
        gamma_m=gamma_test,
        a_eff=a_sol,
        b_eff=b_derive,
        c_eff=c_sol,
        eps=1e-16,
    )

    mo.ui.matplotlib(_fig.gca())
    return


@app.cell
def _(np, plt, pot_effective, pot_morse):
    def plot_morse_analysis(
        alpha_m, beta_m, gamma_m, a_eff, b_eff, c_eff,
        eps=1.0, r_max=None, x_max=None
    ):
        """
        Plot the Morse potential, pair distribution function,
        effective potential, and kernel for the given parameters.

        Parameters
        ----------
        alpha_m, beta_m, gamma_m : float
            Morse potential parameters.
        a_eff, b_eff, c_eff : float
            Effective potential parameters (a, b, c).
        eps : float
            Energy scale for g(r) = exp(-u(r)/eps). Default 1.0.
        r_max : float, optional
            Upper r limit. Defaults to 3 * beta_m.
        x_max : float, optional
            Upper eta limit for kernel. Defaults to 3 * beta_m.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if r_max is None:
            r_max = 3 * beta_m
        if x_max is None:
            x_max = 3 * beta_m

        r_pos = np.linspace(0.01 * beta_m, r_max, 2000)
        x_full = np.linspace(-x_max, x_max, 4000)

        # 1. Morse potential
        u_r = pot_morse(r_pos, alpha_m, beta_m, gamma_m)

        # 2. Pair distribution function
        g_r = np.exp(-u_r / eps)

        # 3. Effective potential
        u_eff_r = pot_effective(u_r, a_eff, b_eff, c_eff)

        # 4. Kernel: K(eta) = eta * u_eff(|eta|)
        u_eff_abs = pot_effective(
            pot_morse(np.abs(x_full), alpha_m, beta_m, gamma_m),
            a_eff, b_eff, c_eff,
        )
        K_x = x_full * u_eff_abs

        _fig, ((_ax1, _ax2), (_ax3, _ax4)) = plt.subplots(
            2, 2, figsize=(12, 10)
        )

        # --- Morse potential ---
        _ax1.plot(r_pos, u_r, "b-")
        _ax1.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        _ax1.axvline(beta_m, color="grey", linestyle="--", linewidth=0.8)
        _ax1.set_xlabel(r"$r$")
        _ax1.set_ylabel(r"$u(r)$")
        _ax1.set_title("Morse Potential")
        _ax1.set_ylim([-2 * alpha_m, 2 * alpha_m])
        _ax1.set_box_aspect(1)

        # --- Pair distribution ---
        _ax2.plot(r_pos, g_r, "r-")
        _ax2.axhline(1, color="grey", linestyle="--", linewidth=0.8)
        _ax2.set_xlabel(r"$r$")
        _ax2.set_ylabel(r"$g(r)$")
        _ax2.set_title(r"Pair Distribution ($\varepsilon = %g$)" % eps)
        #_ax2.set_ylim([np.exp(-alpha_m), np.exp(-alpha_m)].sort())
        # _ax2.set_ylim([1-1e-14, 1+1e-14])
        _ax2.set_box_aspect(1)

        # --- Effective potential ---
        # max = a * (exp(alpha / b) - 1) + c * alpha
        _max = a_eff * (np.exp(alpha_m / b_eff)) + c_eff * alpha_m
        _ylim = 1.2 * _max
        _ax3.plot(r_pos, u_eff_r, "g-")
        _ax3.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        _ax3.axvline(beta_m, color="grey", linestyle="--", linewidth=0.8)
        _ax3.set_xlabel(r"$r$")
        _ax3.set_ylabel(r"$u_{\mathrm{eff}}(r)$")
        _ax3.set_title("Effective Potential")
        _ax3.set_ylim([-_ylim, _ylim])
        _ax3.set_box_aspect(1)

        # --- Kernel ---
        _ax4.plot(x_full, K_x, "m-")
        _ax4.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        _ax4.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        _ax4.set_xlabel(r"$\eta$")
        _ax4.set_ylabel(r"$K(\eta)$")
        _ax4.set_title("Kernel")
        _ax4.set_box_aspect(1)

        _fig.tight_layout()
        return _fig

    return (plot_morse_analysis,)


@app.cell
def _(cond, np, pot_morse, quad, solve):
    def solve_morse_system(alpha_m, beta_m, gamma_m, b_eff, nu_target, mu_target):
        """
        Given Morse potential parameters (alpha_m, beta_m, gamma_m) and the
        effective potential parameter b_eff, solve the linear system

            [nu    ]   [M2_e(b)  -M2_u] [a]
            [6*mu  ] = [M4_e(b)  -M4_u] [c]

        for a and c, using the Morse potential
            u(r) = alpha_m * (1 - exp(-gamma_m * (r - beta_m)))^2 - alpha_m

        Returns (a, c, M_mat, cond_num) where M_mat is the moment matrix.
        """
        def u(r):
            return pot_morse(r, alpha_m, beta_m, gamma_m)

        def safe_exp(z):
            if z > 700:
                return np.exp(700)
            if z < -745:
                return 0.0
            return np.exp(z)

        def integrate_half(f, limit=500):
            """Integrate f from 0 to inf, splitting at the Morse minimum for stability."""
            val1, _ = quad(f, 0, beta_m, limit=limit)
            val2, _ = quad(f, beta_m, np.inf, limit=limit)
            return val1 + val2

        # Compute moments (factor of 2 for symmetric integration from -inf to +inf)
        M2_e = 2 * integrate_half(lambda r: r**2 * (safe_exp(-u(r) / b_eff) - 1))
        M2_u = 2 * integrate_half(lambda r: r**2 * u(r))
        M4_e = 2 * integrate_half(lambda r: r**4 * (safe_exp(-u(r) / b_eff) - 1))
        M4_u = 2 * integrate_half(lambda r: r**4 * u(r))

        # Assemble and solve the linear system
        M_mat = np.array([
            [M2_e, -M2_u],
            [M4_e, -M4_u]
        ])

        rhs = np.array([nu_target, 6 * mu_target])

        a_sol, c_sol = solve(M_mat, rhs)

        return a_sol, c_sol, M_mat, cond(M_mat)

    return (solve_morse_system,)


@app.cell(column=4, hide_code=True)
def _(mo):
    mo.md(r"""
    ## HNC Pair Distribution Function
    """)
    return


@app.cell
def _(np):
    from scipy.fft import dst


    def radial_fourier_3d(f, dr):
        """
        3D Fourier transform of a radial function f(r).

        Convention:
            F(k) = 4*pi * integral_0^inf dr
                   r^2 f(r) sin(kr)/(kr)

        Grid:
            r_j = j * dr,  j = 1, ..., N
            R   = (N + 1) * dr
            k_m = m * pi / R,  m = 1, ..., N

        Parameters
        ----------
        f : ndarray, shape (N,)
            Radial function f(r_j).
        dr : float
            Radial grid spacing.

        Returns
        -------
        k : ndarray
            Wave-number grid.
        F : ndarray
            3D radial Fourier transform.
        """

        N = len(f)

        R = (N + 1) * dr

        r = np.arange(1, N + 1) * dr
        k = np.arange(1, N + 1) * np.pi / R

        # scipy DST-I:
        #
        # dst(x)_m = 2 sum_j x_j sin(k_m r_j)
        #
        # Therefore:
        #
        # integral dr r f(r) sin(kr)
        # ~= dr/2 * dst(r*f)

        F = (
            2.0
            * np.pi
            * dr
            * dst(r * f, type=1)
            / k
        )

        return k, F


    def radial_inverse_fourier_3d(F, dr):
        """
        Inverse 3D Fourier transform of a radial function F(k).

        Convention:
            f(r) = 1/(2*pi^2)
                   integral_0^inf dk
                   k^2 F(k) sin(kr)/(kr)
        """

        N = len(F)

        R = (N + 1) * dr

        r = np.arange(1, N + 1) * dr
        k = np.arange(1, N + 1) * np.pi / R

        dk = np.pi / R

        f = (
            dk
            * dst(k * F, type=1)
            / (4.0 * np.pi**2 * r)
        )

        return f

    return radial_fourier_3d, radial_inverse_fourier_3d


@app.cell
def _(np, radial_fourier_3d, radial_inverse_fourier_3d):
    def solve_hnc_3d(
        beta_u,
        rho,
        dr,
        mix=0.05,
        tol=1e-8,
        max_iter=20_000,
        verbose=True,
    ):
        """
        Solve the homogeneous isotropic 3D Ornstein-Zernike
        equation using the HNC closure.

        Parameters
        ----------
        beta_u : ndarray
            beta * u(r) evaluated on the radial grid.
            Must be dimensionless.

        rho : float
            3D NUMBER density.
            If r is measured in meters:
                rho has units 1/m^3.
            If r is measured in units of sigma:
                rho is the corresponding reduced density.

        dr : float
            Radial grid spacing.

        mix : float
            Linear Picard mixing parameter.
            Typical values: 0.01 -- 0.2.

        tol : float
            Convergence tolerance.

        max_iter : int
            Maximum number of iterations.

        Returns
        -------
        results : dict
            Contains r, k, g, h, c, gamma, S and convergence
            information.
        """

        beta_u = np.asarray(beta_u, dtype=float)

        N = len(beta_u)

        R = (N + 1) * dr
        r = np.arange(1, N + 1) * dr

        # At rho -> 0, gamma -> 0,
        # so this is a natural initial guess.
        gamma = np.zeros(N)

        converged = False

        for iteration in range(max_iter):

            # --------------------------------
            # 1. HNC closure in real space
            # --------------------------------

            exponent = -beta_u + gamma

            # Large negative exponent is harmless:
            # exp(-large) -> 0.
            #
            # Very large positive exponents indicate
            # a numerical/physical problem.
            finite_exponent = exponent[np.isfinite(exponent)]

            if (
                len(finite_exponent) > 0
                and np.max(finite_exponent) > 500
            ):
                raise RuntimeError(
                    "HNC exponent became extremely large. "
                    "Try smaller rho, smaller mixing, or inspect u(r)."
                )

            boltzmann = np.exp(
                np.clip(exponent, -745.0, 500.0)
            )

            c = boltzmann - 1.0 - gamma

            # --------------------------------
            # 2. Fourier transform c(r)
            # --------------------------------

            k, C = radial_fourier_3d(c, dr)

            # --------------------------------
            # 3. Ornstein-Zernike equation
            # --------------------------------

            denominator = 1.0 - rho * C

            min_denominator = np.min(
                np.abs(denominator)
            )

            if min_denominator < 1e-10:
                raise RuntimeError(
                    "1 - rho*C(k) is almost zero. "
                    "The OZ equation is approaching a singularity."
                )

            Gamma = (
                rho * C**2
                / denominator
            )

            # --------------------------------
            # 4. Back to real space
            # --------------------------------

            gamma_new = radial_inverse_fourier_3d(
                Gamma,
                dr,
            )

            # --------------------------------
            # 5. Check convergence
            # --------------------------------

            residual = np.max(
                np.abs(gamma_new - gamma)
            )

            # --------------------------------
            # 6. Linear mixing
            # --------------------------------

            gamma = (
                (1.0 - mix) * gamma
                + mix * gamma_new
            )

            if verbose and iteration % 100 == 0:
                print(
                    f"{iteration:6d}: "
                    f"residual = {residual:.3e}, "
                    f"min|1-rho*C| = {min_denominator:.3e}"
                )

            if residual < tol:
                converged = True

                if verbose:
                    print(
                        f"Converged after {iteration + 1} iterations "
                        f"with residual {residual:.3e}"
                    )

                break

        if not converged:
            raise RuntimeError(
                f"HNC did not converge after {max_iter} iterations. "
                f"Last residual = {residual:.3e}"
            )

        # ------------------------------------
        # Construct final correlation functions
        # ------------------------------------

        g = np.exp(
            np.clip(
                -beta_u + gamma,
                -745.0,
                500.0,
            )
        )

        h = g - 1.0

        c = h - gamma

        # Final transform
        k, C = radial_fourier_3d(c, dr)

        H = C / (1.0 - rho * C)

        # Static structure factor
        S = 1.0 + rho * H

        # Equivalently:
        #
        # S = 1 / (1 - rho*C)

        # Check consistency between HNC and OZ
        h_oz = radial_inverse_fourier_3d(
            H,
            dr,
        )

        oz_error = np.max(
            np.abs(h - h_oz)
        )

        if verbose:
            print(
                f"OZ consistency error: {oz_error:.3e}"
            )

        return {
            "r": r,
            "k": k,
            "g": g,
            "h": h,
            "c": c,
            "gamma": gamma,
            "S": S.real,
            "H": H.real,
            "C": C.real,
            "iterations": iteration + 1,
            "residual": residual,
            "oz_error": oz_error,
        }

    return (solve_hnc_3d,)


@app.cell
def _(np, plt, pot_mor_ui, solve_hnc_3d, ui_beta):
    def _():
        # ---------------------------------
        # Numerical radial grid
        # ---------------------------------

        N = 4096

        sigma = (1 / np.pow(2, 1/6)) * ui_beta.value

        r_max = 20.0 * sigma

        # Notice the N+1 here.
        #
        # The actual numerical points are:
        #
        # dr, 2dr, ..., N dr
        #
        # while r = 0 and r = r_max are not
        # explicitly included.
        dr = r_max / (N + 1)

        r = np.arange(1, N + 1) * dr


        # ---------------------------------
        # Lennard-Jones potential
        # ---------------------------------

        beta_epsilon = 1e-1

        beta_u = beta_epsilon * pot_mor_ui(r)


        # ---------------------------------
        # Number density
        # ---------------------------------

        # Reduced density rho*sigma^3 = 0.3
        rho_star = 1e-1

        rho = rho_star / sigma**3


        # ---------------------------------
        # Solve HNC
        # ---------------------------------

        result = solve_hnc_3d(
            beta_u=beta_u,
            rho=rho,
            dr=dr,
            mix=0.11,
            tol=1e-8,
        )


        # ---------------------------------
        # Extract result
        # ---------------------------------

        r = result["r"]
        g = result["g"]


        # ---------------------------------
        # Plot g(r)
        # ---------------------------------

        mask = r < 8.0 * sigma

        plt.plot(
            r[mask] / sigma,
            g[mask],
        )

        plt.axhline(
            1.0,
            linestyle="--",
        )

        plt.xlabel(r"$r/\sigma$")
        plt.ylabel(r"$g(r)$")

        plt.tight_layout()
        return plt.gca()


    _()
    return


@app.cell
def _():
    return


@app.cell(column=5)
def _():
    return


if __name__ == "__main__":
    app.run()
