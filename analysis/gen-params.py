# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.19.6",
#     "numpy==2.4.1",
#     "pandas==3.0.0",
#     "wigglystuff==0.2.18",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium", sql_output="native")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generate Parameter Sweeps

    This notebook lets you generate paramter sweeps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example Usage

    This is a sample sweep:

    ```python
    nu = -9.6217e-30
    mu = -1.5194205e-36

    nu_sweep = np.geomspace(nu * 0.01, nu * 100, num=10)
    mu_sweep = np.geomspace(mu * 0.01, mu * 100, num=10)

    tayl_sweep = TaylSweep(
        T=[1800.0],
        DT=[5.0e-04],
        NO=[3000],
        gradient=[Gradient.SIGMOID],
        U=[111.15e-18],
        PSI=[0.02],
        gamma=[1.8e-10],
        delta=[1.0e-11],
        kappa=[0.0],
        NU=nu_sweep,
        MU=mu_sweep,
    )

    conv_sweep = ConvSweep(
        T=[1800.0],
        DT=[5.0e-04],
        NO=[3000],
        gradient=[Gradient.SIGMOID],
        U=[111.15e-18],
        PSI=[0.02],
        gamma=[1.8e-10],
        delta=[1.0e-11],
        kappa=[0.0],
    )

    sweep = combine_sweeps(tayl_sweep, conv_sweep)
    ```

    Now you have the runs for your entire sweep inside the `sweep` variable. To generate the CLI arguments simply call `gen_cli_args` like so

    ```python
    args = gen_cli_args(meeting)
    args
    ```

    Then you can copy the values from the display pandas dataframe (as `.csv`) and paste them into a `params.txt` file on the GPU cluster.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Try it yourself!

    Implement the sweeps you want to run and store them in the variable `sweep`. Then Inspect your data in the dataframe below and copy the arguments to the *Clipboard*.

    > Note that the CLI arguments are automatically generated for you if you use the Code Editor below.
    """)
    return


@app.cell
def _(mo):
    initial_code = """# Default Values
    T = 1800.0
    DT = 5.0e-04
    NO = 3000
    U = 111.15e-18
    PSI = 0.02
    gamma = 1.8e-10
    delta = 1.0e-11
    kappa = 0.0

    gradient = Gradient.SIGMOID

    nu = -9.6217e-30
    mu = -1.5194205e-36

    nu_sweep = np.geomspace(nu * 0.01, nu * 100, num=10)
    mu_sweep = np.geomspace(mu * 0.01, mu * 100, num=10)

    # 1. Implement your sweeps

    tayl_sweep = TaylSweep(
        T=[1800.0],
        DT=[5.0e-04],
        NO=[3000],
        gradient=[Gradient.SIGMOID],
        U=[111.15e-18],
        PSI=[0.02],
        gamma=[1.8e-10],
        delta=[1.0e-11],
        kappa=[0.0],
        NU=nu_sweep,
        MU=mu_sweep,
    )

    conv_sweep = ConvSweep(
        T=[1800.0],
        DT=[5.0e-04],
        NO=[3000],
        gradient=[Gradient.SIGMOID],
        U=[111.15e-18],
        PSI=[0.02],
        gamma=[1.8e-10],
        delta=[1.0e-11],
        kappa=[0.0],
    )

    # 2. Combine them into a single sweep

    sweep = combine_sweeps(tayl_sweep, conv_sweep)"""

    editor = mo.ui.code_editor(value=initial_code, language="python", label="Store your sweeps in the `sweep` variable!").form()

    editor
    # code_form = mo.ui.form(
        # element=editor,
        # submit_button_label="Run Code ▶️",
        # submit_button_tooltip="Click to execute the python code above",
        # bordered=False
    # )
    # 
    # code_form
    return (editor,)


@app.cell
def _(clipboard, mo):
    mo.md(f"""
    Inspect your sweep below and if your done copy the CLI Arguments to your Clipboard:

    {clipboard}
    """)
    return


@app.cell
def _(editor, extract_user_code, mo):
    if editor.value:
        result_data, error_msg = extract_user_code(editor.value, "sweep")

        if error_msg:
            output = mo.callout(error_msg, kind="danger")
        else:
            output = result_data
    else:
        output = mo.md("*Waiting for user to run code...*")

    output
    return (output,)


@app.cell
def _(CopyToClipboard, gen_cli_args, mo, output):
    # For some reasong the .to_csv includes the header which is just a '0' and then a \n so i just remove the first two chars
    clipboard = mo.ui.anywidget(CopyToClipboard(text_to_copy=gen_cli_args(output).to_csv(index=False)[2:]))
    return (clipboard,)


@app.cell
def _(
    ConvSweep,
    Gradient,
    Model,
    SimpleNamespace,
    TaylSweep,
    combine_sweeps,
    np,
):
    def extract_user_code(code_string, target_var_name):
        allowed_builtins = {
            "print": print,
            "range": range,
            "len": len,
            "int": int,
            "float": float,
            "list": list,
            "str": str,
        }

        safe_np = SimpleNamespace(
            linspace=np.linspace,
            geomspace=np.geomspace,
            array=np.array,
        )

        custom_tools = {
            "TaylSweep": TaylSweep,
            "ConvSweep": ConvSweep,
            "Gradient": Gradient,
            "MODEL": Model,
            "combine_sweeps": combine_sweeps,
            "np": safe_np,
        }

        execution_globals = {"__builtins__": allowed_builtins, **custom_tools}

        local_scope = {}

        try:
            exec(code_string, execution_globals, local_scope)

            if target_var_name in local_scope:
                return local_scope[target_var_name], None
            return None, f"Variable '{target_var_name}' was not defined."

        except Exception as e:
            return None, f"Error: {e}"
    return (extract_user_code,)


@app.cell(hide_code=True)
def _(ConvSweep, Gradient, TaylSweep, combine_sweeps, np):
    nu = -9.6217e-30
    mu = -1.5194205e-36

    nu_sweep = np.geomspace(nu * 0.01, nu * 100, num=10)
    mu_sweep = np.geomspace(mu * 0.01, mu * 100, num=10)

    meeting_conv_sweep = ConvSweep(
        T=[1800.0],
        DT=[5.0e-04],
        NO=[3000],
        gradient=[Gradient.SIGMOID],
        U=[111.15e-18],
        PSI=[0.02],
        gamma=[1.8e-10],
        delta=[1.0e-11],
        kappa=[0.0],
    )

    meeting_tayl_sweep_normal = TaylSweep(
        T=[1800.0],
        DT=[5.0e-04],
        NO=[3000],
        gradient=[Gradient.SIGMOID],
        U=[111.15e-18],
        PSI=[0.02],
        gamma=[1.8e-10],
        delta=[1.0e-11],
        kappa=[0.0],
        NU=nu_sweep,
        MU=mu_sweep,
    )

    meeting_tayl_sweep_small = TaylSweep(
        T=[1800.0],
        DT=[5.0e-05],
        NO=[30000],
        gradient=[Gradient.SIGMOID],
        U=[111.15e-18],
        PSI=[0.02],
        gamma=[1.8e-10],
        delta=[1.0e-11],
        kappa=[0.0],
        NU=nu_sweep,
        MU=mu_sweep,
    )

    meeting_sweep = combine_sweeps(meeting_conv_sweep, meeting_tayl_sweep_normal, meeting_tayl_sweep_small)
    return (meeting_sweep,)


@app.cell(hide_code=True)
def _(gen_cli_args, meeting_sweep):
    meeting_args = gen_cli_args(meeting_sweep)
    # meeting_args
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Code:
    """)
    return


@app.cell
def _(Enum, dataclass, np):
    EPS = 1e-12


    @dataclass(frozen=True)
    class Range:
        lo: float
        hi: float
        step: float  # grid spacing

        def values(self) -> np.ndarray:
            if self.step <= 0:
                raise ValueError("step must be > 0")
            vals = np.arange(
                self.lo, self.hi + self.step / 2.0, self.step, dtype=float
            )
            vals = vals[vals <= self.hi + EPS]
            return vals


    class Model(str, Enum):
        CONV = "convolution"
        TAYL = "taylor"


    class Gradient(str, Enum):
        LINEAR = "linear"
        SIGMOID = "sigmoid"


    @dataclass(frozen=True)
    class SweepConfig:
        model: list[Model]
        T: list[float]
        DT: list[float]
        NO: list[int]
        gradient: list[Gradient]
        U: list[float]
        PSI: list[float]
        gamma: list[float]
        delta: list[float]
        kappa: list[float]
        # NU, MU?


    sample_config = SweepConfig(
        model=[Model.CONV, Model.TAYL],
        T=[1000.0, 1200.0, 1400.0, 1600.0],
        DT=[5.0e-05, 5.0e-04, 5.0e-03],
        NO=[1000],
        gradient=[Gradient.LINEAR, Gradient.SIGMOID],
        U=[111.15e-18],
        PSI=[0.01, 0.02, 0.03],
        gamma=[1.0e-12, 1.0e-11, 1.0e-10, 1.0e-09],
        delta=[1.0e-12, 1.0e-11, 1.0e-10, 1.0e-09],
        kappa=[1.0e-12, 1.0e-11, 1.0e-10, 1.0e-09],
    )
    return Gradient, Model


@app.cell
def _(Gradient, Model, dataclass, gen_cli_args, pd, product):
    from abc import ABC, abstractmethod
    from typing import Sequence


    @dataclass(frozen=True, kw_only=True)
    class SimulationSweep(ABC):
        T: Sequence[float]
        DT: Sequence[float]
        NO: Sequence[int]
        gradient: Sequence[Gradient]
        U: Sequence[float]
        PSI: Sequence[float]
        gamma: Sequence[float]
        delta: Sequence[float]
        kappa: Sequence[float]

        @property
        @abstractmethod
        def model_name(self) -> Model:
            raise NotImplementedError

        @abstractmethod
        def model_rows(self) -> list[dict]:
            raise NotImplementedError

        def to_dataframe(self) -> pd.DataFrame:
            """Return DataFrame for Parameter Sweep with one Row per Run."""
            simulation_parameter_products = product(
                self.T,
                self.DT,
                self.NO,
                self.gradient,
                self.U,
                self.PSI,
                self.gamma,
                self.delta,
                self.kappa,
            )

            model_parameter_rows = self.model_rows()

            rows = []
            for (
                T,
                DT,
                NO,
                gradient,
                U,
                PSI,
                gamma,
                delta,
                kappa,
            ) in simulation_parameter_products:
                for m in model_parameter_rows:
                    rows.append(
                        {
                            "model": self.model_name,
                            "T": T,
                            "DT": DT,
                            "NO": NO,
                            "gradient": gradient.value,
                            "U": U,
                            "PSI": PSI,
                            "gamma": gamma,
                            "delta": delta,
                            "kappa": kappa,
                            **m,
                        }
                    )
            return pd.DataFrame(rows)

        def to_cli_args(self):
            return gen_cli_args(self.to_dataframe())


    @dataclass(frozen=True, kw_only=True)
    class TaylSweep(SimulationSweep):
        NU: Sequence[float]
        MU: Sequence[float]

        @property
        def model_name(self) -> Model:
            return Model.TAYL

        def model_rows(self) -> list[dict]:
            return [{"NU": NU, "MU": MU} for NU, MU in product(self.NU, self.MU)]


    @dataclass(frozen=True, kw_only=True)
    class ConvSweep(SimulationSweep):
        @property
        def model_name(self) -> Model:
            return Model.CONV

        def model_rows(self) -> list[dict]:
            return [{}]
    return ConvSweep, TaylSweep


@app.cell
def _(ConvSweep, TaylSweep, pd):
    def gen_sweep(
        T,
        DT,
        NO,
        gradient,
        U,
        PSI,
        gamma,
        delta,
        kappa,
    ):
        """Generates a Sweep for both model types, with NU and MU fixed to the default value."""

        args = {
            "T": T,
            "DT": DT,
            "NO": NO,
            "gradient": gradient,
            "U": U,
            "PSI": PSI,
            "gamma": gamma,
            "delta": delta,
            "kappa": kappa,
        }

        tayl_sweep = TaylSweep(
            **args,
            NU=[-1.0e-30, -1.0e-29, -1.0e-28],
            MU=[-1.0e-37, -1.0e-36, -1.0e-35],
        )
        conv_sweep = ConvSweep(**args)

        return pd.concat(
            [tayl_sweep.to_dataframe(), conv_sweep.to_dataframe()],
            ignore_index=True,
        )
    return (gen_sweep,)


@app.cell
def _(Gradient, gen_sweep):
    multi_sweep = gen_sweep(
        T=[1800.0],
        DT=[5.0e-04],
        NO=[3000],
        gradient=[Gradient.LINEAR, Gradient.SIGMOID],
        U=[111.15e-18],
        PSI=[0.02],
        gamma=[1.8e-10],
        delta=[1.0e-11],
        kappa=[0.0, 1.0e-25],
    )

    # multi_sweep
    return


@app.cell
def _(np):
    def gen_cli_args(df):
        def row_to_cli_args(row):
            def cli(arg: str):
                try:
                    if np.isnan(row[arg]):
                        return ""
                    else:
                        return f"""--{arg}={row[arg]}"""
                except TypeError:
                    return f"""--{arg}={row[arg]}"""

            args = []
            for col_name in df:
                if col_name == "model":
                    args.append(f"""--use-{row[col_name].value}""")
                else:
                    args.append(cli(col_name))
            return " ".join(args)

        return df.apply(row_to_cli_args, axis=1)
    return (gen_cli_args,)


@app.cell
def _(pd):
    def combine_sweeps(*sweeps):
        return pd.concat(
            [sweep.to_dataframe() for sweep in sweeps],
            ignore_index=True,
        )
    return (combine_sweeps,)


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import pandas as pd
    from dataclasses import dataclass
    from enum import Enum
    from wigglystuff import CopyToClipboard
    from itertools import product
    from types import SimpleNamespace
    return (
        CopyToClipboard,
        Enum,
        SimpleNamespace,
        dataclass,
        mo,
        np,
        pd,
        product,
    )


if __name__ == "__main__":
    app.run()
