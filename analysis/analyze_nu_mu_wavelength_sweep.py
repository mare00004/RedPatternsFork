# /// script
# dependencies = [
#     "h5py==3.16.0",
#     "altair==5.3.0",
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "pyarrow",
#     "scipy==1.17.1",
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
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyarrow  # noqa: F401
    from red_patterns import RunData, find_peaks, get_rbc_cmap, plot_psi

    return (
        Path,
        RunData,
        alt,
        find_peaks,
        get_rbc_cmap,
        mo,
        np,
        pd,
        plot_psi,
        plt,
    )


@app.cell
def _(np):
    def parse_run_id(name: str) -> int | None:
        if not name.startswith("r"):
            return None
        suffix = name[1:]
        return int(suffix) if suffix.isdigit() else None

    def format_label(value: float) -> str:
        return f"{float(value):.3e}"

    def summarize_last_timestep(run: "RunData", find_peaks_fn):
        z_cm = 100.0 * np.asarray(run.z, dtype=np.float32)
        psi_last_pct = 100.0 * np.asarray(run.load_psi()[-1], dtype=np.float32)

        try:
            peak_z, peak_psi, wavelength_cm, wavelength_std_cm = find_peaks_fn(
                z_cm, psi_last_pct
            )
        except IndexError as exc:
            raise ValueError("No peaks detected in the final time step.") from exc

        if peak_z.size < 2:
            raise ValueError("Need at least two detected peaks to estimate wavelength.")
        if not np.isfinite(wavelength_cm):
            raise ValueError("Detected wavelength is not finite.")

        frequency = 1.0 / wavelength_cm
        frequency_std = wavelength_std_cm / wavelength_cm**2

        return {
            "z_cm": z_cm,
            "psi_last_pct": psi_last_pct,
            "peak_z": peak_z,
            "peak_psi": peak_psi,
            "n_peaks": int(peak_z.size),
            "wavelength_cm": float(wavelength_cm),
            "wavelength_std_cm": float(wavelength_std_cm),
            "frequency_cm_inv": float(frequency),
            "frequency_std_cm_inv": float(frequency_std),
        }

    return format_label, parse_run_id, summarize_last_timestep


@app.cell
def _(Path, mo):
    ui_sweep_dir = mo.ui.file_browser(
        initial_path=Path.cwd(),
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="directory",
        label="Choose NU/MU sweep directory",
    )
    mo.vstack(
        [
            mo.md(
                "# Analyze $\\nu,\\mu$ wavelength sweep\n\n"
                "Pick a sweep directory, then click a heatmap cell to inspect that run's\n"
                "final-time wavelength and the full $\\psi(z,t)$ field."
            ),
            ui_sweep_dir,
        ],
        align="stretch",
    )
    return (ui_sweep_dir,)


@app.cell
def _(RunData, find_peaks, parse_run_id, pd, summarize_last_timestep):
    def scan_runs(base_dir):
        rows: list[dict[str, object]] = []

        for run_dir in sorted(base_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            run_id = parse_run_id(run_dir.name)
            if run_id is None:
                continue

            run_h5 = run_dir / "run.h5"
            if not run_h5.exists():
                continue

            try:
                run = RunData.from_h5(run_h5, load_fields=False)
                cfg = run.config
                if cfg.model.modelType != "TAYL":
                    continue

                nu = float(cfg.model.variant.NU)
                mu = float(cfg.model.variant.MU)
                row = {
                    "run_h5": str(run.path),
                    "run_id": run_id,
                    "NU": nu,
                    "MU": mu,
                    "T": cfg.run.T,
                    "DT": cfg.run.DT,
                    "gradientType": cfg.model.gradientType,
                    "alpha": cfg.model.alpha,
                    "beta": cfg.model.beta,
                }

                try:
                    summary = summarize_last_timestep(run, find_peaks)
                    row.update(summary)
                    row["error"] = None
                except ValueError as exc:
                    row.update(
                        {
                            "n_peaks": None,
                            "wavelength_cm": None,
                            "wavelength_std_cm": None,
                            "frequency_cm_inv": None,
                            "frequency_std_cm_inv": None,
                            "error": str(exc),
                        }
                    )

                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "run_h5": str(run_h5),
                        "run_id": run_id,
                        "NU": None,
                        "MU": None,
                        "T": None,
                        "DT": None,
                        "gradientType": None,
                        "alpha": None,
                        "beta": None,
                        "n_peaks": None,
                        "wavelength_cm": None,
                        "wavelength_std_cm": None,
                        "frequency_cm_inv": None,
                        "frequency_std_cm_inv": None,
                        "error": str(exc),
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(
                ["MU", "NU", "run_id"], kind="stable", na_position="last"
            ).reset_index(drop=True)

        return df

    return (scan_runs,)


@app.cell
def _(Path, mo, scan_runs, ui_sweep_dir):
    is_script_mode = mo.app_meta().mode == "script"
    selected_dir = ui_sweep_dir.path(0) if ui_sweep_dir.value else None
    sweep_dir = selected_dir or (Path.cwd() if is_script_mode else None)

    if sweep_dir is None:
        status = mo.md("Waiting for directory selection...")
        sweep_df = None
    else:
        sweep_df = scan_runs(sweep_dir)
        status = mo.md(f"Scanned `{sweep_dir}` and found {len(sweep_df)} Taylor runs.")

    status
    return (sweep_df,)


@app.cell
def _(mo, sweep_df):
    mo.stop(sweep_df is None, mo.md("Select a sweep directory to load data."))
    mo.stop(sweep_df.empty, mo.md("No Taylor runs with `run.h5` were found."))

    valid_df = sweep_df[sweep_df["NU"].notna() & sweep_df["MU"].notna()].copy()
    mo.stop(valid_df.empty, mo.md("No runs with valid `NU` and `MU` values were found."))

    n_valid_wavelength = int(valid_df["wavelength_cm"].notna().sum())
    n_invalid_wavelength = int(valid_df["wavelength_cm"].isna().sum())

    summary = mo.md(
        f"""
        ## Wavelength Heatmap

        Loaded {len(valid_df)} runs.

        Valid wavelength estimates: {n_valid_wavelength}  
        Failed wavelength estimates: {n_invalid_wavelength}
        """
    )
    summary
    return (valid_df,)


@app.cell
def _(alt, format_label, mo, valid_df):
    df_plot = valid_df.copy()
    df_plot["NU_label"] = df_plot["NU"].map(format_label)
    df_plot["MU_label"] = df_plot["MU"].map(format_label)
    df_plot["error_text"] = df_plot["error"].fillna("")

    click = alt.selection_point(fields=["NU_label", "MU_label"], empty=False)

    heatmap = (
        alt.Chart(df_plot)
        .mark_rect(stroke="black", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "NU_label:O",
                title="ν",
                sort=alt.SortField(field="ν", order="ascending"),
            ),
            y=alt.Y(
                "MU_label:O",
                title="μ",
                sort=alt.SortField(field="μ", order="ascending"),
            ),
             color=alt.Color(
                 "wavelength_cm:Q",
                 title="λ [cm]",
                 scale=alt.Scale(type="log", scheme="viridis"),
             ),
            opacity=alt.condition(click, alt.value(1.0), alt.value(0.45)),
            tooltip=[
                alt.Tooltip("NU:Q", format=".3e"),
                alt.Tooltip("MU:Q", format=".3e"),
                alt.Tooltip("wavelength_cm:Q", title="lambda [cm]", format=".4f"),
                alt.Tooltip(
                    "wavelength_std_cm:Q", title="lambda std [cm]", format=".4f"
                ),
                alt.Tooltip("n_peaks:Q", title="peak count", format=".0f"),
                alt.Tooltip("run_h5:N", title="run.h5"),
                alt.Tooltip("error_text:N", title="error"),
            ],
        )
        .add_params(click)
        .properties(width=460, height=420, title=r"λ(ν, μ)")
    )

    ui_heatmap = mo.ui.altair_chart(heatmap)
    ui_heatmap
    return df_plot, ui_heatmap


@app.cell
def _(df_plot, mo, ui_heatmap):
    def _as_scalar(value):
        if hasattr(value, "iloc"):
            return value.iloc[0] if len(value) else None
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
            value = value.tolist()
            if isinstance(value, list):
                return value[0] if value else None
        return value

    selections = ui_heatmap.selections
    mo.stop(not selections, mo.md("Click a heatmap cell to inspect that run."))

    selected_payload = None
    if isinstance(selections, dict):
        for value in selections.values():
            if value:
                selected_payload = value
                break
    else:
        selected_payload = selections

    if isinstance(selected_payload, list):
        selected_payload = selected_payload[0] if selected_payload else None
    if hasattr(selected_payload, "iloc") and hasattr(selected_payload, "to_dict"):
        selected_payload = selected_payload.iloc[0].to_dict() if len(selected_payload) else None
    if hasattr(selected_payload, "to_dict") and not isinstance(selected_payload, dict):
        selected_payload = selected_payload.to_dict()

    mo.stop(
        not isinstance(selected_payload, dict),
        mo.md(f"Unexpected heatmap selection payload: {selected_payload!r}"),
    )

    nu_label = _as_scalar(selected_payload.get("NU_label"))
    mu_label = _as_scalar(selected_payload.get("MU_label"))
    mo.stop(
        nu_label is None or mu_label is None,
        mo.md(f"Selection missing `NU_label` or `MU_label`: {selected_payload!r}"),
    )

    selected_rows = df_plot[
        (df_plot["NU_label"] == nu_label) & (df_plot["MU_label"] == mu_label)
    ]
    mo.stop(selected_rows.empty, mo.md(f"No run found for selection {selected_payload!r}."))

    selected_row = selected_rows.iloc[0]
    selected_row_md = mo.md(
        f"""
        ## Selected Run

        Selected run: `NU={float(selected_row["NU"]):.3e}`, `MU={float(selected_row["MU"]):.3e}`  
        `run_h5={selected_row["run_h5"]}`
        """
    )
    selected_row_md
    return (selected_row,)


@app.cell
def _(RunData, find_peaks, mo, pd, selected_row, summarize_last_timestep):
    mo.stop(
        pd.isna(selected_row["wavelength_cm"]),
        mo.md(
            "The selected run does not have a valid wavelength estimate. "
            f"Error: `{selected_row['error']}`"
        ),
    )

    selected_run = RunData.from_h5(selected_row["run_h5"], load_fields=False)
    selected_summary = summarize_last_timestep(selected_run, find_peaks)
    return selected_run, selected_summary


@app.cell
def _(get_rbc_cmap, plot_psi, selected_run):
    psi_plot = plot_psi(
        selected_run,
        vmin=0.0,
        vmax=100.0,
        cmap=get_rbc_cmap(),
    )
    return (psi_plot,)


@app.cell
def _(mo, plt, selected_summary):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(selected_summary["z_cm"], selected_summary["psi_last_pct"], label=r"$\psi(z)$")
    ax.plot(
        selected_summary["peak_z"],
        selected_summary["peak_psi"],
        "x",
        color="red",
        label="Detected Peaks",
    )
    ax.set_xlabel(r"$z \; [cm]$")
    ax.set_ylabel(r"$\psi \; [\%]$")
    ax.set_title("Final time step peak detection")
    ax.legend()

    psi_profile_plot = mo.ui.matplotlib(ax)
    return (psi_profile_plot,)


@app.cell
def _(mo, selected_summary):
    wavelength_table = mo.md(
        f"""
        | Quantity | Value |
        |----------|-------|
        | **Number of peaks** | {selected_summary["n_peaks"]} |
        | **λ** (avg. peak spacing) | {selected_summary["wavelength_cm"]:.4f} ± {selected_summary["wavelength_std_cm"]:.4f} cm |
        | **ν** (spatial frequency) | {selected_summary["frequency_cm_inv"]:.4f} ± {selected_summary["frequency_std_cm_inv"]:.4f} cm⁻¹ |
        """
    )
    return (wavelength_table,)


@app.cell
def _(mo, psi_plot, psi_profile_plot, wavelength_table):
    mo.vstack(
        [
            mo.hstack([psi_plot, psi_profile_plot], align="center", justify="start"),
            wavelength_table,
        ],
        align="stretch",
    )
    return


if __name__ == "__main__":
    app.run()
