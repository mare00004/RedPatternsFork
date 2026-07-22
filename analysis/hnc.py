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

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
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
    ## Potentials

    The potentials were defined in a way that the minima align.

    > Note that the $a, b, c$ parameters of the effective potential have nothing to do with the $a,b,c$ parameters of the Lennard-Jones and Morse-Potential.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lennard-Jones Potential

    The Lennard-Jones potential is given by

    $$
        u(r) = a \left[ \left( \frac{b}{r} \right)^{12} - 2 \left( \frac{b}{r} \right)^{6}\right]
    $$

    with $a, b \in \mathbb{R}$. The minimum is at $r_\text{min} = b$ with the value $u(r_\text{min}) = -a$.
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
        u(r) = a (1 - e^{-b(r - c)})^2 - a
    $$

    with $a, b, c \in \mathbb{R}$. The minimum is at $r_\text{min} = c$ with the value $u(r_\text{min}) = -a$.
    """)
    return


@app.cell
def _(np):
    def pot_morse(r: np.ndarray, a: np.floating, b: np.floating, c: np.floating) -> np.ndarray:
        return a * np.power(1 - np.exp(-b * (r - c)), 2) - a

    return (pot_morse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparision

    Below you can see both of the potential plotted with similar coefficients....
    """)
    return


@app.cell
def _(mo, np, plt, pot_lennard_jones, pot_morse):
    fig, ax = plt.subplots(figsize=(8, 6))

    r = np.linspace(1, 10, 1000)

    a = 2
    b = 2
    c = b

    pot_len = pot_lennard_jones(r, a, b)
    pot_mor = pot_morse(r, a, b, c)

    u_len = lambda t: pot_lennard_jones(t, a, b)
    u_mor = lambda t: pot_morse(t, a, b, c)

    ax.plot(r, pot_len, "r-", label="Lennard-Jones Potential")

    ax.plot(r, pot_mor, "b-", label="Morse Potential")

    ax.set_xlim([1, 5])
    ax.set_ylim([-3, 5])
    ax.hlines([0], xmin=0, xmax=8, colors="grey", linestyles="dashed")
    ax.legend()

    mo.ui.matplotlib(ax)
    return pot_len, pot_mor, r, u_len, u_mor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effective Potential

    As stated above the effective potential will be modeled as

    $$
        u_\text{eff}[u](r) = a (e^{-\frac{u(r)}{b}} - 1) - c \, u(r)
    $$

    with the parameters $a, b, c \in \mathbb{R}$.
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
    ui_a = mo.ui.slider(-10, 10, 0.1, value=1, label="$a$", show_value=True)
    ui_b = mo.ui.slider(-10, 10, 0.1, value=1, label="$b$", show_value=True)
    ui_c = mo.ui.slider(-10, 10, 0.1, value=1, label="$c$", show_value=True)

    mo.hstack([ui_a, ui_b, ui_c])
    return ui_a, ui_b, ui_c


@app.cell
def _(mo, plt, pot_effective, pot_len, pot_mor, r, ui_a, ui_b, ui_c):
    _fig, _ax = plt.subplots(figsize=(8, 6))

    _pot_eff_len = pot_effective(pot_len, ui_a.value, ui_b.value, ui_c.value)
    _pot_eff_mor = pot_effective(pot_mor, ui_a.value, ui_b.value, ui_c.value)

    _ax.plot(r, _pot_eff_len, "b-", label="Effective Potential (Lennard-Jones)")
    _ax.plot(r, _pot_eff_mor, "r-", label="Effective Potential (Morse)")

    _ax.set_xlim([0, 6])
    _ax.set_ylim([-10, 10])
    _ax.legend()

    mo.ui.matplotlib(_ax)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kernel

    The interaction kernel will be, the "force closure", of the effective potential, i.e.

    $$
        K(x) = - x \int_{|x|}^{\infty} g(r) u_{\text{eff}}'(r) \;d r
    $$

    which can be simplified to

    $$
        K(x) = x \left(e^{-\frac{u(|x|)}{b}} - 1\right) \left[\frac{a}{2} \left(e^{- \frac{u(|x|)}{b}} + 1\right) + bc\right]
    $$

    Where $a, b, c$ are the parameters of the effective potential.
    """)
    return


@app.cell
def _(np):
    # def kernel(x: np.ndarray, pot: np.ndarray, a, b, c):
    #     """
    #     x has to be positive and start at 0
    #     """
    #     x_pos = x[x >= 0]
    #     k_pos = x_pos * (np.exp(- pot / b) - 1) * ( (a / 2) * (np.exp(- pot / b) + 1) + b * c) 
    #     return np.concat([-k_pos[::-1], k_pos[0::]])

    def kernel(x: np.ndarray, pot: np.ufunc, a, b, c):
        return x * (np.exp(- pot(np.abs(x)) / b) - 1) * ( (a / 2) * (np.exp(- pot(np.abs(x)) / b) + 1) + b * c) 

    return (kernel,)


@app.cell
def _(mo, ui_a, ui_b, ui_c):
    mo.hstack([
        ui_a, ui_b, ui_c
    ])
    return


@app.cell
def _(kernel, mo, np, plt, u_len, u_mor, ui_a, ui_b, ui_c):
    _fig, _ax = plt.subplots(figsize=(8, 6))

    x = np.linspace(-10, 10, 2000) 

    kernel_len = kernel(x, u_len, ui_a.value, ui_b.value, ui_c.value)
    kernel_mor = kernel(x, u_mor, ui_a.value, ui_b.value, ui_c.value)

    _ax.plot(x, kernel_len, "b-", label="Kernel (Lennard-Jones)")
    _ax.plot(x, kernel_mor, "r-", label="Kernel (Morse)")

    _ax.set_xlim([-10, 10])
    _ax.set_ylim([-50, 50])
    _ax.legend()

    mo.ui.matplotlib(_ax)
    return kernel_len, kernel_mor, x


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
    the parameters have to be $U = {latex_scientific(U)} \text{{J}}$ and $\sigma = {latex_scientific(SIGMA)} \text{{m}}$. The relation between those parameters and our parameters are $a_\text{{LJ}} = U$ and $b_\text{{LJ}} = \sqrt[6]{2} \sigma$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the Morse-Potential we require that its minimum coincides with that of the Leannard-Jones Potential. This fixes $a_\text{Morse} = a_\text{LJ}$ and $c_\text{Morse} = b_\text{LJ}$ and leaves $b_\text{Morse}$ free.
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


if __name__ == "__main__":
    app.run()
