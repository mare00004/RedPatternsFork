# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "h5py==3.16.0",
#     "marimo>=0.19.6",
#     "numpy==2.4.1",
#     "pandas==3.0.0",
#     "pydantic==2.13.4",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", sql_output="native")

with app.setup:
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import marimo as mo
    import numpy as np
    import pandas as pd

    NOTEBOOK_FILE = (
        Path(__file__).resolve()
        if "__file__" in globals()
        else (Path.cwd() / "analysis" / "gen-params.py").resolve()
    )
    ANALYSIS_DIR = NOTEBOOK_FILE.parent
    if str(ANALYSIS_DIR) not in sys.path:
        sys.path.insert(0, str(ANALYSIS_DIR))

    from red_patterns.kernel import ClosureType, PDFType
    from red_patterns.phi import PhiType
    from red_patterns.sweep_jobs import (
        ConvSweep,
        Gradient,
        KernelSweep,
        PhiSweep,
        Range,
        TaylSweep,
        combine_sweeps,
        normalize_runs,
        write_sweep_export,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Generate Parameter Sweeps

    This notebook builds structured sweep payloads and exports a sweep directory
    containing `runs.jsonl` and `run_ids.txt`. Each line in `runs.jsonl` is one
    run object with a sequential `run_id`, runtime parameters, and embedded
    phi/kernel generation config.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example Usage

    ```python
    nu = -2.832638e-30
    mu = -4.468445e-37

    nu_sweep = np.geomspace(nu * 0.01, nu * 100, num=10)
    mu_sweep = np.geomspace(mu * 0.01, mu * 100, num=10)

    phi = PhiSweep(
        psi_avg=[0.02],
        phi_type=[PhiType.GAUSSIAN],
        gaussian_mu=[1100.0],
        gaussian_sigma=[4.0],
    )

    kernel = KernelSweep(
        closure=[ClosureType.FORCE],
        pair_distribution=[PDFType.NEAREST_NEIGHBOR],
        U=[111.15e-18],
    )

    tayl_sweep = TaylSweep(
        N=[256],
        T=[1800.0],
        DT=[5.0e-04],
        storeTime=[3000.0],
        gradient=[Gradient.SIGMOID],
        phi=phi,
        NU=nu_sweep,
        MU=mu_sweep,
    )

    conv_sweep = ConvSweep(
        N=[256],
        T=[1800.0],
        DT=[5.0e-04],
        storeTime=[3000.0],
        gradient=[Gradient.SIGMOID],
        phi=phi,
        kernel=kernel,
    )

    sweep = combine_sweeps(tayl_sweep, conv_sweep)
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Build Your Sweep

    Store the final value in the variable `sweep`. The preview table below is derived
    directly from the exported JSON payloads, so what you inspect is exactly what
    ends up in the exported `runs.jsonl`.
    """)
    return


@app.cell
def _():
    initial_code = """phi = PhiSweep(
        psi_avg=[0.02],
        phi_type=[PhiType.HOMOGENEOUS],
        N=[256],
        wing=[32],
    )

    kernel = KernelSweep(
        closure=[ClosureType.FORCE],
        pair_distribution=[PDFType.NEAREST_NEIGHBOR],
        U=[111.15e-18],
    )

    nu = -2.832638e-30
    mu = -4.468445e-37
    nu_sweep = np.geomspace(nu * 0.01, nu * 100, num=5)
    mu_sweep = np.geomspace(mu * 0.01, mu * 100, num=5)

    tayl_sweep = TaylSweep(
        N=[256],
        T=[1000.0],
        DT=[1e-3],
        storeTime=[1],
        gradient=[Gradient.LINEAR],
        phi=phi,
        NU=nu_sweep,
        MU=mu_sweep,
    )

    conv_sweep = ConvSweep(
        N=[256],
        T=[1000.0],
        DT=[1e-3],
        storeTime=[1],
        gradient=[Gradient.SIGMOID],
        phi=phi,
        kernel=kernel,
    )

    sweep = combine_sweeps(tayl_sweep, conv_sweep)"""

    editor = mo.ui.code_editor(
        value=initial_code,
        language="python",
        label="Store your exported sweep in the `sweep` variable.",
    ).form()
    editor
    return (editor,)


@app.function
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
        "ClosureType": ClosureType,
        "ConvSweep": ConvSweep,
        "Gradient": Gradient,
        "KernelSweep": KernelSweep,
        "PDFType": PDFType,
        "PhiSweep": PhiSweep,
        "PhiType": PhiType,
        "Range": Range,
        "TaylSweep": TaylSweep,
        "combine_sweeps": combine_sweeps,
        "np": safe_np,
    }

    execution_globals = {"__builtins__": allowed_builtins, **custom_tools}
    local_scope: dict[str, object] = {}

    try:
        exec(code_string, execution_globals, local_scope)
        if target_var_name not in local_scope:
            return None, f"Variable `{target_var_name}` was not defined."
        sweep_runs = normalize_runs(local_scope[target_var_name])
        return sweep_runs, None
    except Exception as exc:
        return None, f"Error: {exc}"


@app.cell
def _(editor):
    mo.stop(not editor.value, mo.md("No Editor Value Found!"))
    return


@app.cell
def _(editor):
    sweep_runs, error_msg = extract_user_code(editor.value, "sweep")
    mo.stop(error_msg is not None, mo.callout(error_msg, kind="danger"))
    mo.stop(sweep_runs is None, mo.md("*Waiting for user to run code...*"))
    mo.md(f"Prepared **{len(sweep_runs)}** runs for export.")
    preview = pd.json_normalize([run.model_dump(mode="json") for run in sweep_runs])
    preview
    return error_msg, sweep_runs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Export Sweep Directory

    Export the sweep to disk. The export writes both `runs.jsonl` and
    `run_ids.txt` into the selected directory.
    """)
    return


@app.cell
def _():
    export_dir = mo.ui.file_browser(
        initial_path=Path.cwd(),
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="directory",
        label="Sweep directory",
    )
    export_form = (
        mo.md(
            """
    {export_dir}
    """
        )
        .batch(export_dir=export_dir)
        .form(
            submit_button_label="Export sweep",
            clear_on_submit=False,
            show_clear_button=True,
        )
    )
    export_form
    return (export_form,)


@app.cell
def _(error_msg, export_form, sweep_runs, true):
    mo.stop(export_form.value is None, mo.md("Submit the form to export your sweep."))
    mo.stop(
        error_msg is not None,
        mo.md(f"Fix the sweep definition before exporting. ```{error_msg}```"),
    )

    dir_entries = export_form.value.get("export_dir") or []

    mo.stop(not dir_entries, mo.md("Please select an export directory."))

    output_dir = Path(dir_entries[0].path)
    try:
        runs_path, run_ids_path = write_sweep_export(output_dir, sweep_runs)
    except Exception as exc:
        mo.stop(true, mo.md(rf"Export failed: `{exc}`"))

    first_id = sweep_runs[0].run_id if sweep_runs else "n/a"
    last_id = sweep_runs[-1].run_id if sweep_runs else "n/a"
    mo.vstack(
        [
            mo.md(rf"Exported `{len(sweep_runs)}` runs to `{runs_path}`."),
            mo.md(rf"Wrote queue file `{run_ids_path}`."),
            mo.md(rf"`run_id` range: `{first_id}` to `{last_id}`."),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
