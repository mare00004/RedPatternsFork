# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.0",
#     "mcp==1.28.1",
#     "numpy==2.5.1",
#     "pydantic-ai==2.5.1",
#     "pydantic-ai-slim[groq]>=2.5.1",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(slider):
    N = 512
    L = 0.06
    w_L = 0.005
    offset = 0.002

    L_eff = L + (2 * w_L)

    DZ = L_eff / N
    P = slider.value
    R_min = -30.0
    R_max = +30.0

    P_0 = 1100.0
    return DZ, L, L_eff, N, P, P_0, R_max, R_min, offset, w_L


@app.cell
def _(L, P, R_min, offset, w_L):
    def p(x):
        return (P / L) * (x - (L / 2))

    def l(x):
        t = offset - w_L
        a = ((P / 2) + R_min - ((P / L) * t)) / (t**2)
        return a * ((x - w_L) ** 2) +  (P / L) * (x - w_L) - (P / 2)


    return l, p


@app.cell
def _(L_eff, R_max, R_min, l, np, offset, p, w_L):
    def p_eff(x):
        condlist = [
            x <= offset,
            (offset < x) & (x <= w_L),
            (w_L < x) & (x < (L_eff - w_L)),
            (L_eff - w_L <= x) & (x < (L_eff - offset)),
            x >= (L_eff - offset),
        ]

        funclist = [
            lambda x: R_min,
            lambda x: l(x),
            lambda x: p(x - w_L),
            lambda x: -l(L_eff - x),
            lambda x: R_max,
        ]

        return np.piecewise(x, condlist, funclist)

    return (p_eff,)


@app.cell
def _(DZ, N, P_0, np, p_eff, plt):
    z = (np.arange(N) + 0.5) * DZ
    y = p_eff(z) + P_0

    plt.plot(z, y)
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(start=0.0, stop=20.0, step=1.0, value=8.0)
    slider
    return (slider,)


if __name__ == "__main__":
    app.run()
