# /// script
# dependencies = [
#     "altair==5.3.0",
#     "h5py==3.16.0",
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "pydantic==2.13.4",
#     "pyarrow",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="wide")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    from red_patterns import RunData, get_rbc_cmap, load_runs_jsonl, plot_psi
    from red_patterns.models import TaylorRun

    return (
        Path,
        RunData,
        TaylorRun,
        alt,
        get_rbc_cmap,
        load_runs_jsonl,
        mo,
        np,
        pd,
        plot_psi,
    )


@app.cell
def _(Path, mo):
    ui_sweep_dir = mo.ui.file_browser(
        initial_path=Path.cwd(),
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="directory",
        label="Choose Taylor sweep directory",
    )
    mo.vstack(
        [
            mo.md(
                "# Analyze Taylor $\\nu$–$\\mu$ density sweep\n\n"
                "Choose a sweep directory containing `runs.jsonl` and, when available, "
                "`results/<run_id>/run.h5`. Select an average density, then click a "
                "heatmap cell to inspect $\\psi(z,t)$ for that run."
            ),
            ui_sweep_dir,
        ],
        align="stretch",
    )
    return (ui_sweep_dir,)


@app.cell
def _(Path, TaylorRun, load_runs_jsonl, pd):
    def scan_sweep(sweep_dir: Path) -> pd.DataFrame:
        """Read Taylor metadata from runs.jsonl and locate expected result files."""
        runs_path = sweep_dir / "runs.jsonl"
        rows: list[dict[str, object]] = []

        for run in load_runs_jsonl(runs_path):
            if not isinstance(run, TaylorRun):
                continue

            phi_params = run.phi.params.model_dump(mode="json")
            run_h5 = sweep_dir / "results" / run.run_id / "run.h5"
            rows.append(
                {
                    "run_id": run.run_id,
                    "NU": float(run.NU),
                    "MU": float(run.MU),
                    "psi_avg": float(phi_params.pop("psi_avg")),
                    "phi_type": phi_params.pop("phi_type"),
                    "N": run.N,
                    "T": run.T,
                    "DT": run.DT,
                    "storeTime": run.storeTime,
                    "gradient": run.gradient.value,
                    "run_h5": str(run_h5),
                    "h5_exists": run_h5.is_file(),
                    **{f"phi_{key}": value for key, value in phi_params.items()},
                }
            )

        dataframe = pd.DataFrame(rows)
        if not dataframe.empty:
            dataframe = dataframe.sort_values(
                ["psi_avg", "MU", "NU", "run_id"], kind="stable"
            ).reset_index(drop=True)
        return dataframe

    return (scan_sweep,)


@app.cell
def _(Path, mo, scan_sweep, ui_sweep_dir):
    is_script_mode = mo.app_meta().mode == "script"
    selected_dir = ui_sweep_dir.path(0) if ui_sweep_dir.value else None
    default_dir = Path.cwd() / "data"
    sweep_dir = Path(selected_dir) if selected_dir else (default_dir if is_script_mode else None)

    if sweep_dir is None:
        sweep_df = None
        status = mo.md("Waiting for a sweep directory selection...")
    elif not (sweep_dir / "runs.jsonl").is_file():
        sweep_df = None
        status = mo.callout(
            f"`{sweep_dir}` does not contain `runs.jsonl`.", kind="warn"
        )
    else:
        sweep_df = scan_sweep(sweep_dir)
        status = mo.md(
            f"Loaded `{len(sweep_df)}` Taylor runs from `{sweep_dir / 'runs.jsonl'}`."
        )

    status
    return sweep_df, sweep_dir


@app.cell
def _(mo, sweep_df):
    mo.stop(sweep_df is None, mo.md("Select a directory with `runs.jsonl` to continue."))
    mo.stop(sweep_df.empty, mo.md("The JSONL file contains no Taylor runs."))

    available = int(sweep_df["h5_exists"].sum())
    mo.md(
        f"## Parsed runs\n\n"
        f"{len(sweep_df)} Taylor configurations across `{sweep_df['psi_avg'].nunique()}` "
        f"average densities; {available} expected `run.h5` files are present."
    )
    return


@app.cell
def _(mo, sweep_df):
    mo.stop(sweep_df is None or sweep_df.empty, mo.md("No dataframe to display yet."))
    dataframe_table = mo.ui.table(data=sweep_df, selection=None, pagination=True)
    dataframe_table
    return (dataframe_table,)


@app.cell
def _(mo, sweep_df):
    mo.stop(sweep_df is None or sweep_df.empty, mo.md("No densities are available yet."))

    density_values = sorted(float(value) for value in sweep_df["psi_avg"].unique())
    density_options = {f"{value:.17g}": value for value in density_values}
    ui_density = mo.ui.dropdown(
        options=density_options,
        value=f"{density_values[0]:.17g}",
        label=r"Average density $\langle\psi\rangle$",
    )
    ui_density
    return (ui_density,)


@app.cell
def _(mo, sweep_df, ui_density):
    mo.stop(sweep_df is None or sweep_df.empty, mo.md("No heatmap data yet."))
    density_df = sweep_df[sweep_df["psi_avg"] == float(ui_density.value)].copy()
    density_df["NU_label"] = density_df["NU"].map(lambda value: f"{value:.3e}")
    density_df["MU_label"] = density_df["MU"].map(lambda value: f"{value:.3e}")
    density_df["heatmap_value"] = 1.0
    return (density_df,)


@app.cell
def _(alt, density_df, mo, ui_density):
    click = alt.selection_point(fields=["run_id"], empty=False)
    heatmap = (
        alt.Chart(density_df)
        .mark_rect(stroke="black", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "NU_label:O",
                title="ν",
                sort=alt.SortField(field="NU", order="ascending"),
            ),
            y=alt.Y(
                "MU_label:O",
                title="μ",
                sort=alt.SortField(field="MU", order="ascending"),
            ),
            color=alt.Color(
                "heatmap_value:Q",
                title="Placeholder value",
                scale=alt.Scale(domain=[0.0, 1.0], scheme="blues"),
            ),
            opacity=alt.condition(click, alt.value(1.0), alt.value(0.45)),
            tooltip=[
                alt.Tooltip("run_id:N", title="run ID"),
                alt.Tooltip("NU:Q", title="ν", format=".3e"),
                alt.Tooltip("MU:Q", title="μ", format=".3e"),
                alt.Tooltip("psi_avg:Q", title="average density", format=".6g"),
                alt.Tooltip("phi_type:N", title="initial phi"),
                alt.Tooltip("h5_exists:N", title="run.h5 available"),
            ],
        )
        .add_params(click)
        .properties(
            width=500,
            height=430,
            title=f"Taylor runs at average density {float(ui_density.value):.6g}",
        )
    )
    ui_heatmap = mo.ui.altair_chart(heatmap)
    ui_heatmap
    return (ui_heatmap,)


@app.cell
def _(density_df, mo, ui_heatmap):
    def selected_run_id(selections):
        if isinstance(selections, dict):
            payload = next((value for value in selections.values() if value), None)
        else:
            payload = selections
        if isinstance(payload, list):
            payload = payload[0] if payload else None
        if hasattr(payload, "iloc") and hasattr(payload, "to_dict"):
            payload = payload.iloc[0].to_dict() if len(payload) else None
        if hasattr(payload, "to_dict") and not isinstance(payload, dict):
            payload = payload.to_dict()
        if not isinstance(payload, dict):
            return None
        run_id = payload.get("run_id")
        if isinstance(run_id, list):
            return run_id[0] if run_id else None
        return run_id

    run_id = selected_run_id(ui_heatmap.selections)
    mo.stop(run_id is None, mo.md("Click a heatmap cell to inspect that run."))
    selected_rows = density_df[density_df["run_id"] == run_id]
    mo.stop(selected_rows.empty, mo.md(f"No run found for selected ID `{run_id}`."))
    selected_row = selected_rows.iloc[0]
    return (selected_row,)


@app.cell
def _(mo, selected_row):
    selected_summary = mo.md(
        f"## Selected run\n\n"
        f"`{selected_row['run_id']}` — ν = `{float(selected_row['NU']):.3e}`, "
        f"μ = `{float(selected_row['MU']):.3e}`, "
        f"$\\langle\\psi\\rangle$ = `{float(selected_row['psi_avg']):.6g}`  \n"
        f"Expected result: `{selected_row['run_h5']}`"
    )
    selected_summary
    return


@app.cell
def _(RunData, get_rbc_cmap, mo, plot_psi, selected_row):
    run_h5 = selected_row["run_h5"]
    if not bool(selected_row["h5_exists"]):
        psi_panel = mo.callout(
            f"No `run.h5` exists yet for `{selected_row['run_id']}` at `{run_h5}`.",
            kind="warn",
        )
    else:
        selected_run = RunData.from_h5(run_h5, load_fields=False)
        psi_panel = mo.as_html(
            plot_psi(
                selected_run,
                vmin=0.0,
                vmax=100.0,
                cmap=get_rbc_cmap(),
                title=f"$\\psi(z,t)$ — {selected_row['run_id']}",
            )
        )
    psi_panel
    return


if __name__ == "__main__":
    app.run()
