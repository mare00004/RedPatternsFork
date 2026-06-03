# /// script
# dependencies = [
#     "h5py==3.16.0",
#     "altair==5.3.0",
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "pyarrow",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import altair as alt
    import pyarrow  # noqa: F401
    from red_patterns import RunData, plot_psi

    return Path, RunData, alt, mo, np, pd, plot_psi, plt


@app.cell
def _(Path, RunData, plot_psi, plt):
    def plot_psi_file(run_h5: Path, **kwargs) -> plt.Figure:
        run = RunData.from_h5(run_h5, load_fields=False)
        # Use a standard colormap; the repo's custom colormap can render poorly
        # depending on environment.
        kwargs.setdefault("cmap", "viridis")
        kwargs.setdefault("vmin", 0.0)
        kwargs.setdefault("vmax", 0.5)
        return plot_psi(run, **kwargs)

    return


@app.cell
def _(Path, mo):
    ui_sweep_dir = mo.ui.file_browser(
        initial_path=Path.cwd(),
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="directory",
        label="Choose sweep directory!",
    )
    ui_sweep_dir
    return (ui_sweep_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analyze $\nu, \mu$-sweep

    First select a directory that contains
    """)
    return


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
                    "gradient": cfg.model.gradientType,
                    "T": cfg.run.T,
                    "DT": cfg.run.DT,
                    "U": cfg.model.U,
                    "PSI": cfg.model.PSI,
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
def _(mo, pd, scan_runs, ui_sweep_dir):
    get_df, set_df = mo.state(pd.DataFrame())
    selected_dir = ui_sweep_dir.path(0)

    if selected_dir:
        sims, df_new = scan_runs(selected_dir)
        set_df(df_new)
        result = mo.md(f"Scanned `{selected_dir}`: {len(df_new)} runs")
    else:
        result = mo.md("Waiting for directory selection...")

    result
    return (get_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Explore your runs in a table:
    """)
    return


@app.cell
def _(get_df, mo):
    def format_mmss(seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}:{s:02d}"

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
            pagination=True,
            selection="single",
            show_column_summaries=None,
        )
    table
    return (table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RMSE vs. Reference Run

    Select a reference run in the table above. This computes
    \(\mathrm{RMSE}_{zt}\) between that reference \(\psi(t,z)\) and every Taylor run
    in the sweep, then plots \(\mathrm{RMSE}(\nu,\mu)\) as a surface.

    RMSE is normalized over all samples (both time and space):
    $$
    \mathrm{RMSE}_{zt} = \sqrt{\frac{1}{N_t N_z} \sum_{t,z} |\psi_1-\psi_2|^2}
    $$
    """)
    return


@app.cell
def _(RunData, get_df, mo, np, pd, table):
    # Reference selection comes from the table.
    mo.stop(not isinstance(table, mo.ui.table), mo.md("No table to select from."))
    selection = table.value
    mo.stop(
        selection is None or len(selection) != 1,
        mo.md("Select exactly 1 reference run."),
    )

    ref_h5 = selection.iloc[0]["run_h5"]
    ref_run = RunData.from_h5(ref_h5, load_fields=False)
    psi_ref = ref_run.load_psi()

    df = get_df().copy()
    tayl = df[
        (df["model"] == "TAYL") & df["NU"].notna() & df["MU"].notna()
    ].reset_index(drop=True)
    mo.stop(tayl.empty, mo.md("No Taylor runs found in this sweep."))

    results: list[dict[str, object]] = []
    nt_ref, nz_ref = int(psi_ref.shape[0]), int(psi_ref.shape[1])
    denom = float(nt_ref * nz_ref)

    for _, _row in tayl.iterrows():
        run_h5 = _row["run_h5"]
        nu = float(_row["NU"])
        mu = float(_row["MU"])

        try:
            run = RunData.from_h5(run_h5, load_fields=False)
            psi = run.load_psi()

            if psi.shape != psi_ref.shape:
                results.append(
                    {
                        "NU": nu,
                        "MU": mu,
                        "RMSE": np.nan,
                        "error": f"psi shape mismatch: {psi.shape} vs ref {psi_ref.shape}",
                        "run_h5": run_h5,
                    }
                )
                continue

            # Optional axis sanity checks. If they fail, still compute RMSE but mark it.
            axis_note = ""
            if run.time.shape != ref_run.time.shape or not np.allclose(
                run.time, ref_run.time
            ):
                axis_note += "time-axis differs; "
            if run.z.shape != ref_run.z.shape or not np.allclose(run.z, ref_run.z):
                axis_note += "z-axis differs; "

            d = psi.astype(np.float64) - psi_ref.astype(np.float64)
            rmse = float(np.sqrt(np.sum(d * d) / denom))
            results.append(
                {
                    "NU": nu,
                    "MU": mu,
                    "RMSE": rmse,
                    "error": axis_note.strip() if axis_note else None,
                    "run_h5": run_h5,
                }
            )
        except Exception as e:
            results.append(
                {"NU": nu, "MU": mu, "RMSE": np.nan, "error": str(e), "run_h5": run_h5}
            )

    df_rmse = pd.DataFrame(results)

    # Build a (MU, NU) grid for plotting.
    nu_grid = np.sort(df_rmse["NU"].unique())
    mu_grid = np.sort(df_rmse["MU"].unique())
    Z = np.full((mu_grid.size, nu_grid.size), np.nan, dtype=np.float64)
    nu_to_i = {float(v): i for i, v in enumerate(nu_grid)}
    mu_to_j = {float(v): j for j, v in enumerate(mu_grid)}
    for _, r in df_rmse.iterrows():
        i = nu_to_i[float(r["NU"])]
        j = mu_to_j[float(r["MU"])]
        Z[j, i] = float(r["RMSE"]) if pd.notna(r["RMSE"]) else np.nan
    return Z, df_rmse, mu_grid, nu_grid, psi_ref, ref_h5


@app.cell
def _(Z, df_rmse, mo, mu_grid, np, nu_grid, plt):
    # Surface plot: RMSE(NU, MU)
    NU, MU = np.meshgrid(nu_grid, mu_grid)
    _fig = plt.figure(constrained_layout=True)
    _ax = _fig.add_subplot(111, projection="3d")

    surf = _ax.plot_surface(NU, MU, Z, cmap="viridis", linewidth=0, antialiased=True)
    _ax.set_xlabel("NU")
    _ax.set_ylabel("MU")
    _ax.set_zlabel("RMSE")
    _ax.set_title(f"RMSE vs reference (n={len(df_rmse)})")
    _fig.colorbar(surf, ax=_ax, shrink=0.7, pad=0.1, label="RMSE")

    _ui_surface = mo.ui.matplotlib(_ax)
    _ui_surface
    return


@app.cell
def _(alt, df_rmse, mo, pd):
    # Interactive heatmap: click to select (NU, MU).
    mo.stop(df_rmse.empty, mo.md("No RMSE data to plot."))

    df_plot = df_rmse.copy()
    # Use stable string labels for discrete grid cells while preserving numeric values in tooltips.
    df_plot["NU_label"] = df_plot["NU"].map(
        lambda x: f"{float(x):.3e}" if pd.notna(x) else ""
    )
    df_plot["MU_label"] = df_plot["MU"].map(
        lambda x: f"{float(x):.3e}" if pd.notna(x) else ""
    )

    click = alt.selection_point(fields=["NU_label", "MU_label"], empty=False)

    base_chart = (
        alt.Chart(df_plot)
        .mark_rect(stroke="black", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "NU_label:O",
                title="NU",
                sort=alt.SortField(field="NU", order="ascending"),
            ),
            y=alt.Y(
                "MU_label:O",
                title="MU",
                sort=alt.SortField(field="MU", order="ascending"),
            ),
            color=alt.Color("RMSE:Q", scale=alt.Scale(scheme="viridis")),
            opacity=alt.condition(click, alt.value(1.0), alt.value(0.4)),
            tooltip=[
                alt.Tooltip("NU:Q", format=".3e"),
                alt.Tooltip("MU:Q", format=".3e"),
                alt.Tooltip("RMSE:Q", format=".6g"),
            ],
        )
        .add_params(click)
        .properties(width=420, height=420, title="RMSE(NU, MU) interactive heatmap")
    )

    ui_rmse_heatmap = mo.ui.altair_chart(base_chart)
    ui_rmse_heatmap
    return df_plot, ui_rmse_heatmap


@app.cell
def _(df_plot, mo, ui_rmse_heatmap):
    # `ui_rmse_heatmap.selections` is a selection store keyed by selection name.
    selections = ui_rmse_heatmap.selections
    mo.stop(not selections, mo.md("Click a (NU, MU) cell in the heatmap to select it."))

    # Altair auto-names params (e.g. "param_1"). Grab the first non-empty entry.
    sel = None
    if isinstance(selections, dict):
        for v in selections.values():
            if v:
                sel = v
                break
    else:
        sel = selections

    # Normalize to a single dict of fields.
    if isinstance(sel, list):
        sel = sel[0] if sel else None
    if hasattr(sel, "iloc") and hasattr(sel, "to_dict"):
        # pandas DataFrame
        sel = sel.iloc[0].to_dict() if len(sel) else None
    if hasattr(sel, "to_dict") and not isinstance(sel, dict):
        # pandas Series
        sel = sel.to_dict()

    mo.stop(not isinstance(sel, dict), mo.md(f"Unexpected selection payload: {sel!r}"))

    nu_label = sel.get("NU_label")
    mu_label = sel.get("MU_label")
    if hasattr(nu_label, "iloc"):
        nu_label = nu_label.iloc[0]
    if hasattr(mu_label, "iloc"):
        mu_label = mu_label.iloc[0]
    if isinstance(nu_label, list):
        nu_label = nu_label[0] if nu_label else None
    if isinstance(mu_label, list):
        mu_label = mu_label[0] if mu_label else None
    mo.stop(
        nu_label is None or mu_label is None,
        mo.md(f"Selection missing fields: {sel}"),
    )

    _sel_row = df_plot[
        (df_plot["NU_label"] == nu_label) & (df_plot["MU_label"] == mu_label)
    ]
    mo.stop(_sel_row.empty, mo.md(f"No row found for selection: {sel}"))
    _sel_row = _sel_row.iloc[0]

    selected_run_h5 = _sel_row["run_h5"]

    mo.md(
        f"Selected cell: NU={float(_sel_row['NU']):.3e}, MU={float(_sel_row['MU']):.3e}  \n"
        f"RMSE={_sel_row['RMSE']!s}  \n"
        f"run_h5=`{selected_run_h5}`"
    )
    return (selected_run_h5,)


@app.cell
def _(RunData, mo, plt, psi_ref, ref_h5, selected_run_h5):
    # Plot reference psi(t,z) next to the selected run.
    mo.stop(not selected_run_h5, mo.md("Click a (NU, MU) cell to pick a run."))

    _ref_run = RunData.from_h5(ref_h5, load_fields=False)
    _sel_run = RunData.from_h5(selected_run_h5, load_fields=False)
    psi_sel = _sel_run.load_psi()

    # Keep the comparison strict; shape mismatch makes side-by-side misleading.
    mo.stop(
        psi_sel.shape != psi_ref.shape,
        mo.md(
            f"psi shape mismatch: selected {psi_sel.shape} vs reference {psi_ref.shape}"
        ),
    )

    vmin = float(min(psi_ref.min(), psi_sel.min()))
    vmax = float(max(psi_ref.max(), psi_sel.max()))

    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    # Flip axes so x=t and y=z (psi is stored as psi[t, z]).
    im0 = _ax0.imshow(
        psi_ref.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    _ax0.set_title("Reference $\\psi(t,z)$")
    _ax0.set_xlabel("t index")
    _ax0.set_ylabel("z index")

    im1 = _ax1.imshow(
        psi_sel.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    _ax1.set_title("Selected $\\psi(t,z)$")
    _ax1.set_xlabel("t index")
    _ax1.set_ylabel("z index")

    _fig.colorbar(im1, ax=[_ax0, _ax1], shrink=0.9, pad=0.02, label="$\\psi$")
    _ui_psi = mo.ui.matplotlib(_ax0)
    _ui_psi
    return


if __name__ == "__main__":
    app.run()
