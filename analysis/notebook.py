# /// script
# dependencies = ["marimo"]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    print("Hi")
    return


@app.cell
def _():
    print("Hi again!")
    return


if __name__ == "__main__":
    app.run()
