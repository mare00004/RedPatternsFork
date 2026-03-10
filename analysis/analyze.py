# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "h5py==3.15.1",
#     "marimo>=0.19.4",
#     "matplotlib==3.10.8",
#     "numpy==2.4.1",
#     "pandas==3.0.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    import marimo as mo

    # Import the shared module we just created
    from red_patterns import RunData, plot_psi

    return Path, RunData, mo, pd, plot_psi


@app.cell
def _(Path, RunData, pd):
    def scan_runs(base_dir: Path) -> tuple[list[RunData], pd.DataFrame]:
        sims: list[RunData] = []
        rows = []

        for d in sorted(base_dir.iterdir()):
            if not d.is_dir() or "." not in d.name:
                continue

            h5 = d / "run.h5"
            if not h5.exists():
                continue

            try:
                # Load via the HDF5 schema
                run_data = RunData.from_h5(h5)
                sims.append(run_data)
                cfg = run_data.config

                # Parse the folder name
                cluster_id = job_id = None
                try:
                    a, b = d.name.split(".", 1)
                    cluster_id, job_id = int(a), int(b)
                except ValueError:
                    pass

                # Flatten parameters for pandas Table
                row = {
                    "run_h5": str(run_data.path),
                    "model": cfg.model.modelType,
                    "gradient": cfg.model.gradient,
                    "T": cfg.run.T,
                    "DT": cfg.run.DT,
                    "U": cfg.model.U,
                    "PSI": cfg.model.PSI,
                    "gamma": cfg.model.gamma,
                    "delta": cfg.model.delta,
                    "kappa": cfg.model.kappa,
                    "cluster_id": cluster_id,
                    "job_id": job_id,
                }

                if cfg.model.modelType == "TAYL":
                    row["NU"] = cfg.model.variant.NU
                    row["MU"] = cfg.model.variant.MU
                else:
                    row["NU"] = None
                    row["MU"] = None

                rows.append(row)

            except Exception as e:
                print(f"Failed loading {d}: {e}")

        df = pd.DataFrame(rows)
        if len(df):
            sort_cols = [
                c
                for c in ["cluster_id", "job_id", "model", "gradient"]
                if c in df.columns
            ]
            if sort_cols:
                df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

        return sims, df

    return (scan_runs,)


@app.cell
def _(mo, pd):
    get_df, set_df = mo.state(pd.DataFrame())
    return get_df, set_df


@app.cell
def _(mo):
    base_dir_ui = mo.ui.file_browser(
        selection_mode="directory",
        multiple=False,
        label="Choose data directory!",
    )
    base_dir_ui
    return (base_dir_ui,)


@app.cell
def _(base_dir_ui, mo, scan_runs, set_df):
    selected_dir = base_dir_ui.path(0)

    if selected_dir:
        sims, df_new = scan_runs(selected_dir)
        set_df(df_new)
        result = mo.md(f"Scanned `{selected_dir}`: {len(df_new)} runs")
    else:
        result = mo.md("Waiting for directory selection...")

    result
    return


@app.function
def format_mmss(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


@app.cell
def _(get_df, mo):
    df_display = get_df().reset_index(drop=True)

    if df_display.empty:
        table = mo.md("No data loaded.")
    else:
        table = mo.ui.table(
            data=df_display[
                [
                    "model",
                    "gradient",
                    "T",
                    "DT",
                    "U",
                    "PSI",
                    "gamma",
                    "delta",
                    "kappa",
                    "NU",
                    "MU",
                    "run_h5",
                    "cluster_id",
                    "job_id",
                ]
            ],
            format_mapping={
                "T": format_mmss,
                "DT": "{:.1e}",
                "U": "{:.3e}",
                "gamma": "{:.1e}",
                "delta": "{:.1e}",
                "kappa": "{:.1e}",
                "NU": "{:.3e}",
                "MU": "{:.3e}",
            },
            selection="multi",
            pagination=True,
            show_column_summaries=None,
        )
    table
    return (table,)


@app.cell
def _(RunData, mo, plot_psi, table):
    # Ensure a table exists and has values
    if isinstance(table, mo.ui.table):
        selection = table.value
        count = len(selection)

        if count == 0:
            _result = mo.md("**Select a run to plot!**")
        elif count == 1:
            run_data = RunData.from_h5(selection.iloc[0]["run_h5"])
            _result = plot_psi(run_data, vmin=0.0, vmax=0.5)
        elif count == 2:
            run_data_1 = RunData.from_h5(selection.iloc[0]["run_h5"])
            run_data_2 = RunData.from_h5(selection.iloc[1]["run_h5"])
            _result = mo.hstack(
                [
                    plot_psi(run_data_1, vmin=0.0, vmax=0.5),
                    plot_psi(run_data_2, vmin=0.0, vmax=0.5),
                ],
                align="center",
                gap=1,
            )
        else:
            _result = mo.md("**Too many runs selected!**")
    else:
        _result = None

    _result
    return


if __name__ == "__main__":
    app.run()
