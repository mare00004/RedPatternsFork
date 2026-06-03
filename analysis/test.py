# /// script
# dependencies = [
#     "marimo",
#     "numpy==2.4.6",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test Notebook

    ## Run Tests
    """)
    return


@app.cell
def _(np):
    PATH = "/home/max/projects/RedPatternsFork/data/conv_const_linear/run.npz"
    data = np.load(PATH)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate Tests
    """)
    return


if __name__ == "__main__":
    app.run()
