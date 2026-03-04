# /// script
# dependencies = [
#     "animatplot==0.4.3",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.1",
# ]
# ///

import marimo

__generated_with = "0.19.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    return FuncAnimation, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Band pattern formation of Red Blood Cells

    ## Density Gradients

    The density of the percoll suspension is described by a continuous function $p(z,t)$, where $z$ is the height inside the tube and $t$ is the time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (Inverse) Sigmoidal Gradient

    The traditional self-forming sigmoidal gradient can be modeled by the following function

    $$
        p(z,t) = p_0 + \delta(t) \frac{\chi(z)}{\left(1 - |\chi(z)|^{\mu(t)}\right)^{1/\mu(t)}}
    $$

    , where $\delta(t) = \delta_1 t^{\delta_2}$, $\mu(t) = \mu_1 t + \mu_2$ and $\chi(z) = (z - z_0) \lambda^{-1}$. Furthermore, $p_0 = 1.1 \; g ml^{-1}$ is the average density **of what??** at the center $z_0$ of the tube. The rest of the parameters $(\lambda, \delta_1, \delta_2, \mu_1, \mu_2)$ can be measured.
    """)
    return


@app.cell
def _():
    sys_length = 0.06798 # 0.06  # in m
    z_0 = sys_length / 2.0
    p_0 = 1100.0  # in g/l
    lam = 0.0338  # lambda in m
    delta_1 = 3.1773e-4  # in m^3 kg^-1 s^-delta_2
    delta_2 = 1.52
    mu_1 = 1.1012e-3
    mu_2 = 0.6
    return delta_1, delta_2, lam, mu_1, mu_2, p_0, sys_length, z_0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One can easily see that for all $t$ the function $p(z,t)$ has a point symmetry [🔗](https://de.wikipedia.org/wiki/Punktsymmetrie#%C3%9Cberblick) around $(z_0, 0)$, because

    $$
        p(z_0 + z, t) = - p(z_0 - z, t)
    $$

    In the following you can see a plot of the function
    """)
    return


@app.cell
def _(delta_1, delta_2, lam, mu_1, mu_2, np, p_0, sys_length, z_0):
    z = np.linspace(0, sys_length, 500)

    def p(z, t):
        delta = delta_1 * t**delta_2
        chi = (z - z_0) / lam
        mu = mu_1 * t + mu_2
        denom = (1 - np.abs(chi) ** mu) ** (1 / mu)
        return p_0 + delta * (chi / denom)
    return p, z


@app.cell
def _():
    # plt.plot(p(z, 1))
    return


@app.cell
def _(FuncAnimation, mo, np, p, plt, sys_length, z):
    T = 1200

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlabel("z")
    ax.set_ylabel("p(z,t)")

    ax.set_xlim(0, sys_length)
    ax.set_ylim(1000, 1200)

    def update(t):
        line.set_data(z, p(z, t))
        ax.set_title(f"Wave at t = {t:.2f} s")
        return line,

    ani = FuncAnimation(fig, update, frames=np.linspace(0, T, 100), interval=50)
    mo.Html(ani.to_html5_video())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Implementation

    Since the simulation has no boundary conditions we enlarge our system and fit two parabolas to the edeges.
    """)
    return


@app.cell
def _(np, p, plt, sys_length):
    _wing_size = 0.005 # in m
    _sys_length = sys_length + 2 * _wing_size
    _DZ = 1.0e-4
    _z = np.arange(0, _sys_length + _DZ, _DZ)

    def _grad_wing(p, z, t):
        wingL = _wing_size / _DZ
        r3 = (p(_wing_size, t) - p(_wing_size - _DZ, t)) / _DZ
        r2 = p(_wing_size, t);
        r1 = r2 - 50;
        x1 = 12;
        x2 = wingL;
        a = (r1 - r2 + r3 * (x2 - x1)) / ((x1 - x2) * (x1 - x2))
        b = r3 - 2 * a * x2
        c = r2 - r3 * x2 + x2 * x2 * a

        wing_grid = z[z <= _wing_size]

        return a * wing_grid**2 + b * wing_grid + c

    _grad_wing(p, _z, 1)

    plt.plot(_z)
    return


if __name__ == "__main__":
    app.run()
