import marimo

__generated_with = "0.23.9"
app = marimo.App(width="wide")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import MaxNLocator
    from wigglystuff import PlaySlider

    from red_patterns import (
        RunData,
        SweepCatalog,
        find_peaks,
        get_rbc_cmap,
        plot_psi,
        selected_sweep_catalog,
        sweep_directory_picker,
    )
    from red_patterns.models import TaylorRun
    from red_patterns.phi import PhiResult, plot_phi

    return (
        Path,
        RunData,
        SweepCatalog,
        TaylorRun,
        alt,
        find_peaks,
        get_rbc_cmap,
        MaxNLocator,
        mo,
        np,
        pd,
        PhiResult,
        PlaySlider,
        plot_phi,
        plot_psi,
        selected_sweep_catalog,
        sweep_directory_picker,
        plt,
    )


@app.cell
def _(Path, mo, sweep_directory_picker):
    ui_sweep_dir = sweep_directory_picker(
        mo,
        initial_path=Path.cwd(),
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
def _(SweepCatalog, TaylorRun, np, pd):
    def scan_sweep(catalog: SweepCatalog) -> pd.DataFrame:
        """Read Taylor metadata from runs.jsonl and locate expected result files."""
        rows: list[dict[str, object]] = []

        for entry in catalog.entries:
            run = entry.run
            if not isinstance(run, TaylorRun):
                continue

            phi_params = run.phi.params.model_dump(mode="json")
            run_h5 = entry.run_h5
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
                    "h5_exists": entry.h5_exists,
                    **{f"phi_{key}": value for key, value in phi_params.items()},
                }
            )

        dataframe = pd.DataFrame(rows)
        if not dataframe.empty:
            dataframe = dataframe.sort_values(
                ["psi_avg", "MU", "NU", "run_id"], kind="stable"
            ).reset_index(drop=True)

            # NU = MU = 0 is the no-interaction Taylor reference. Pair it only
            # with runs that share every other simulation and phi setting.
            comparison_columns = [
                column
                for column in dataframe.columns
                if column
                not in {"run_id", "NU", "MU", "run_h5", "h5_exists"}
            ]
            baseline_mask = np.isclose(
                dataframe["NU"], 0.0, rtol=0.0, atol=1e-300
            ) & np.isclose(
                dataframe["MU"], 0.0, rtol=0.0, atol=1e-300
            )
            dataframe["is_no_interaction"] = baseline_mask
            dataframe["baseline_run_id"] = None
            dataframe["baseline_run_h5"] = None
            dataframe["comparison_status"] = "baseline not found"

            baselines = dataframe.loc[baseline_mask]
            for index, row in dataframe.loc[~baseline_mask].iterrows():
                matches = baselines
                for column in comparison_columns:
                    if pd.isna(row[column]):
                        matches = matches[matches[column].isna()]
                    else:
                        matches = matches[matches[column] == row[column]]
                if len(matches) == 1:
                    baseline = matches.iloc[0]
                    dataframe.at[index, "baseline_run_id"] = baseline["run_id"]
                    dataframe.at[index, "baseline_run_h5"] = baseline["run_h5"]
                    dataframe.at[index, "comparison_status"] = "pending FFT"
                elif len(matches) > 1:
                    dataframe.at[index, "comparison_status"] = "ambiguous baseline"
        return dataframe

    return (scan_sweep,)


@app.cell
def _(Path, RunData, np):
    def delta_dominant_wavelength_series(
        interaction_h5: str | Path, baseline_h5: str | Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return time, dominant mode, and wavelength for interaction minus baseline ψ."""
        interaction = RunData.from_h5(Path(interaction_h5), load_fields=False)
        baseline = RunData.from_h5(Path(baseline_h5), load_fields=False)
        interaction_psi = np.asarray(interaction.load_psi(), dtype=np.float64)
        baseline_psi = np.asarray(baseline.load_psi(), dtype=np.float64)
        time = np.asarray(interaction.time, dtype=np.float64)
        z = np.asarray(interaction.z, dtype=np.float64)

        if (
            interaction_psi.shape != baseline_psi.shape
            or not np.array_equal(time, np.asarray(baseline.time, dtype=np.float64))
            or not np.array_equal(z, np.asarray(baseline.z, dtype=np.float64))
        ):
            raise ValueError("interaction and no-interaction ψ grids differ")
        if interaction_psi.ndim != 2 or interaction_psi.shape[1] < 2:
            raise ValueError("ψ requires at least two z points for an FFT")
        if not np.all(np.isfinite(interaction_psi)) or not np.all(
            np.isfinite(baseline_psi)
        ):
            raise ValueError("ψ contains non-finite values")

        dz = float(z[1] - z[0])
        if not np.isfinite(dz) or dz <= 0.0:
            raise ValueError("z coordinates must be strictly increasing")
        difference = interaction_psi - baseline_psi
        coefficients = np.fft.rfft(
            difference - difference.mean(axis=1, keepdims=True), axis=1
        )
        amplitudes = np.abs(coefficients)
        if amplitudes.shape[1] < 2:
            raise ValueError("ψ has no non-DC FFT modes")

        mode_candidates = 1 + np.argmax(amplitudes[:, 1:], axis=1)
        has_nonzero_mode = np.any(amplitudes[:, 1:] > 0.0, axis=1)
        dominant_modes = np.where(has_nonzero_mode, mode_candidates, -1)
        spatial_frequencies = np.fft.rfftfreq(z.size, d=dz)
        wavelengths = np.full(time.shape, np.nan, dtype=np.float64)
        wavelengths[has_nonzero_mode] = 1.0 / spatial_frequencies[
            mode_candidates[has_nonzero_mode]
        ]
        return time, dominant_modes, wavelengths

    return (delta_dominant_wavelength_series,)


@app.cell
def _(mo, scan_sweep, selected_sweep_catalog, ui_sweep_dir):
    catalog, status = selected_sweep_catalog(mo, ui_sweep_dir)
    if catalog is None:
        sweep_df = None
    else:
        sweep_df = scan_sweep(catalog)
    sweep_dir = catalog.root if catalog is not None else None

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
def _(delta_dominant_wavelength_series, mo, np, sweep_df, ui_density):
    mo.stop(sweep_df is None or sweep_df.empty, mo.md("No heatmap data yet."))
    density_df = sweep_df[
        (sweep_df["psi_avg"] == float(ui_density.value))
        & ~sweep_df["is_no_interaction"]
    ].copy()
    density_df["delta_dominant_wavelength_cm"] = np.nan
    for _index, _row in density_df.iterrows():
        _baseline_h5 = _row["baseline_run_h5"]
        if (
            _row["comparison_status"] != "pending FFT"
            or not bool(_row["h5_exists"])
            or not _baseline_h5
        ):
            continue
        try:
            _, _, _wavelengths = delta_dominant_wavelength_series(
                _row["run_h5"], _baseline_h5
            )
            density_df.at[_index, "delta_dominant_wavelength_cm"] = (
                100.0 * _wavelengths[-1]
            )
            density_df.at[_index, "comparison_status"] = "ready"
        except (OSError, ValueError) as _error:
            density_df.at[_index, "comparison_status"] = str(_error)
    density_df["NU_label"] = density_df["NU"].map(lambda value: f"{value:.3e}")
    density_df["MU_label"] = density_df["MU"].map(lambda value: f"{value:.3e}")
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
                "delta_dominant_wavelength_cm:Q",
                title=r"Final Δψ dominant λ [cm]",
                scale=alt.Scale(scheme="viridis"),
            ),
            opacity=alt.condition(click, alt.value(1.0), alt.value(0.45)),
            tooltip=[
                alt.Tooltip("run_id:N", title="run ID"),
                alt.Tooltip("NU:Q", title="ν", format=".3e"),
                alt.Tooltip("MU:Q", title="μ", format=".3e"),
                alt.Tooltip("psi_avg:Q", title="average density", format=".6g"),
                alt.Tooltip("phi_type:N", title="initial phi"),
                alt.Tooltip("h5_exists:N", title="run.h5 available"),
                alt.Tooltip("baseline_run_id:N", title="no-interaction run"),
                alt.Tooltip(
                    "delta_dominant_wavelength_cm:Q",
                    title="final Δψ dominant λ [cm]",
                    format=".6g",
                ),
                alt.Tooltip("comparison_status:N", title="comparison status"),
            ],
        )
        .add_params(click)
        .properties(
            width=500,
            height=430,
            title=(
                f"Final Δψ dominant wavelength at average density "
                f"{float(ui_density.value):.6g}"
            ),
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
        f"Interaction result: `{selected_row['run_h5']}`  \n"
        f"No-interaction reference: `{selected_row['baseline_run_id']}`"
    )
    return (selected_summary,)


@app.cell
def _(Path, RunData, mo, selected_row):
    selected_run_h5 = Path(selected_row["run_h5"])
    baseline_run_h5 = selected_row["baseline_run_h5"]
    mo.stop(
        selected_row["comparison_status"] != "ready" or not baseline_run_h5,
        mo.callout(
            "This run has no usable no-interaction comparison: "
            f"{selected_row['comparison_status']}.",
            kind="warn",
        ),
    )
    mo.stop(
        not bool(selected_row["h5_exists"]),
        mo.callout(
            f"No `run.h5` exists yet for `{selected_row['run_id']}` at `{selected_run_h5}`.",
            kind="warn",
        ),
    )
    selected_run = RunData.from_h5(selected_run_h5, load_fields=False)
    mo.stop(
        selected_run.n_saved < 2,
        mo.callout("The selected run needs at least two saved timesteps.", kind="warn"),
    )
    selected_run_md = mo.md(
        f"**Interaction file:** `{selected_run_h5}`  \n"
        f"**No-interaction file:** `{baseline_run_h5}`"
    )
    return baseline_run_h5, selected_run, selected_run_h5, selected_run_md


@app.cell
def _(np, selected_run):
    inspect_psi = np.asarray(selected_run.load_psi(), dtype=np.float64)
    inspect_time = np.asarray(selected_run.time, dtype=np.float64)
    inspect_z = np.asarray(selected_run.z, dtype=np.float64)
    return inspect_psi, inspect_time, inspect_z


@app.cell
def _(baseline_run_h5, inspect_psi, inspect_time, inspect_z, mo, np, plt, RunData):
    baseline_run = RunData.from_h5(baseline_run_h5, load_fields=False)
    baseline_psi = np.asarray(baseline_run.load_psi(), dtype=np.float64)
    delta_psi = inspect_psi - baseline_psi
    delta_limit = max(float(np.max(np.abs(delta_psi))) * 100.0, 1e-12)
    _, delta_psi_axis = plt.subplots(constrained_layout=True)
    delta_image = delta_psi_axis.pcolormesh(
        inspect_time,
        100.0 * inspect_z,
        100.0 * delta_psi.T,
        shading="auto",
        cmap="RdBu_r",
        vmin=-delta_limit,
        vmax=delta_limit,
    )
    delta_psi_axis.set(
        xlabel=r"$t\;[s]$",
        ylabel=r"$z\;[cm]$",
        title=r"$\Delta\psi$: interaction − no interaction",
    )
    delta_psi_axis.figure.colorbar(
        delta_image,
        ax=delta_psi_axis,
        label=r"$\Delta\psi\;[\%]$",
    )
    delta_psi_panel = mo.vstack(
        [mo.md("### Delta Psi(z, t)"), mo.ui.matplotlib(delta_psi_axis)],
        align="stretch",
    )
    return (delta_psi_panel,)


@app.cell
def _(baseline_run_h5, delta_dominant_wavelength_series, selected_run_h5):
    (
        delta_time,
        delta_dominant_mode,
        delta_dominant_wavelength,
    ) = delta_dominant_wavelength_series(selected_run_h5, baseline_run_h5)
    return delta_dominant_mode, delta_dominant_wavelength, delta_time


@app.cell
def _(inspect_z, selected_run):
    # Keep the compact selected-run view deterministic: inspect the final saved
    # frame and use the full z domain for its FFT.
    fft_time_index = selected_run.n_saved - 1
    fft_z_start_index = 0
    fft_z_stop_index = inspect_z.shape[0] - 1
    return fft_time_index, fft_z_start_index, fft_z_stop_index


@app.cell
def _(
    delta_dominant_mode,
    delta_dominant_wavelength,
    delta_time,
    fft_time_index,
    mo,
    np,
    plt,
):
    _delta_wavelength_cm = 100.0 * delta_dominant_wavelength
    _, delta_wavelength_axis = plt.subplots(constrained_layout=True)
    delta_wavelength_axis.plot(
        delta_time,
        _delta_wavelength_cm,
        color="#7c3aed",
        linewidth=1.5,
        drawstyle="steps-mid",
    )
    if np.isfinite(_delta_wavelength_cm[fft_time_index]):
        delta_wavelength_axis.scatter(
            [delta_time[fft_time_index]],
            [_delta_wavelength_cm[fft_time_index]],
            color="#dc2626",
            zorder=3,
            label=f"mode {delta_dominant_mode[fft_time_index]}",
        )
        delta_wavelength_axis.legend()
    delta_wavelength_axis.set(
        xlabel=r"$t\;[s]$",
        ylabel=r"$\lambda_{\mathrm{dom}}(t)\;[\mathrm{cm}]$",
        title=r"Dominant wavelength of $\Delta\psi$",
    )
    delta_dominant_wavelength_panel = mo.vstack(
        [
            mo.md("### Delta Psi Dominant Wavelength"),
            mo.ui.matplotlib(delta_wavelength_axis),
        ],
        align="stretch",
    )
    return (delta_dominant_wavelength_panel,)


@app.cell
def _(fft_z_start_index, fft_z_stop_index, inspect_psi, inspect_z, mo, np):
    fft_z = np.asarray(
        inspect_z[slice(fft_z_start_index, fft_z_stop_index + 1)], dtype=np.float64
    )
    fft_n_points = int(fft_z.shape[0])
    mo.stop(
        fft_n_points < 2,
        mo.md("Select at least two z indices for the Fourier transform."),
    )
    psi_fft = np.asarray(
        inspect_psi[:, slice(fft_z_start_index, fft_z_stop_index + 1)],
        dtype=np.float64,
    )
    fft_coeffs = np.fft.rfft(psi_fft - psi_fft.mean(axis=1, keepdims=True), axis=1)
    fft_amplitudes = np.abs(fft_coeffs)
    fft_phases = np.angle(fft_coeffs)
    fft_spatial_freqs = np.fft.rfftfreq(fft_n_points, d=float(fft_z[1] - fft_z[0]))
    fft_wavelengths = np.full(fft_spatial_freqs.shape, np.inf, dtype=np.float64)
    fft_wavelengths[fft_spatial_freqs > 0.0] = 1.0 / fft_spatial_freqs[
        fft_spatial_freqs > 0.0
    ]
    fft_mode_numbers = np.arange(fft_coeffs.shape[1], dtype=int)
    fft_wavenumbers = 2.0 * np.pi * fft_spatial_freqs
    return (
        fft_amplitudes,
        fft_coeffs,
        fft_mode_numbers,
        fft_n_points,
        fft_phases,
        fft_spatial_freqs,
        fft_wavelengths,
        fft_wavenumbers,
        fft_z,
    )


@app.cell
def _(fft_coeffs, mo):
    max_mode = fft_coeffs.shape[1] - 1
    mo.stop(max_mode < 1, mo.md("The selected run has no non-zero Fourier modes."))
    fft_mode_selector = mo.ui.slider(
        start=1, stop=max_mode, step=1, value=1, label="Fourier mode n"
    )
    return (fft_mode_selector,)


@app.cell
def _(fft_mode_selector):
    fft_selected_mode = int(fft_mode_selector.value)
    return (fft_selected_mode,)


@app.cell
def _(fft_time_index, fft_time_slider, inspect_time, mo, selected_run):
    fft_time_panel = mo.vstack(
        [
            mo.md("### Time Step"),
            fft_time_slider,
            mo.md(
                f"Step `{fft_time_index}` of `{selected_run.n_saved - 1}`  \\n"
                f"Time `{inspect_time[fft_time_index]:.6g}` s"
            ),
        ],
        align="stretch",
    )
    return (fft_time_panel,)


@app.cell
def _(fft_n_points, fft_z, fft_z_index_range, fft_z_start_index, fft_z_stop_index, inspect_z, mo):
    fft_z_range_panel = mo.vstack(
        [
            mo.md("### FFT z-Index Range"),
            fft_z_index_range,
            mo.md(
                f"Indices `{fft_z_start_index}` to `{fft_z_stop_index}`  \\n"
                f"Physical range `{100 * fft_z[0]:.6g}` to `{100 * fft_z[-1]:.6g}` cm  \\n"
                f"Grid points used in FFT `{fft_n_points}` of `{inspect_z.shape[0]}`"
            ),
        ],
        align="stretch",
    )
    return (fft_z_range_panel,)


@app.cell
def _(fft_amplitudes, fft_mode_selector, fft_selected_mode, fft_spatial_freqs, fft_wavelengths, mo, np):
    wavelength_text = (
        r"$\infty$"
        if not np.isfinite(fft_wavelengths[fft_selected_mode])
        else f"{100 * fft_wavelengths[fft_selected_mode]:.6g} cm"
    )
    fft_mode_panel = mo.vstack(
        [
            mo.md("### Mode Selection"),
            fft_mode_selector,
            mo.md(
                f"Mode `{fft_selected_mode}`  \\n"
                f"Spatial frequency `{fft_spatial_freqs[fft_selected_mode]:.6g}` m$^{{-1}}$  \\n"
                f"Wavelength `{wavelength_text}`  \\n"
                f"Stored coefficient series shape `{fft_amplitudes[:, fft_selected_mode].shape}`"
            ),
        ],
        align="stretch",
    )
    return (fft_mode_panel,)


@app.cell
def _(PhiResult, fft_time_index, inspect_time, mo, plot_phi, selected_run):
    _phi_figure = plot_phi(
        PhiResult(rho=selected_run.rho, z=selected_run.z, phi_values=selected_run.phi_frame(fft_time_index))
    )
    _phi_figure.axes[0].set_title(rf"$\varphi(\rho, z)$ at $t={inspect_time[fft_time_index]:.3f}\,\mathrm{{s}}$")
    phi_panel = mo.vstack([mo.md("### Phi(z, rho)"), mo.as_html(_phi_figure)], align="stretch")
    return (phi_panel,)


@app.cell
def _(get_rbc_cmap, mo, plot_psi, selected_row, selected_run):
    _psi_figure = plot_psi(
        selected_run,
        vmin=0.0,
        vmax=100.0,
        cmap=get_rbc_cmap(),
        title=selected_row["run_id"],
    )
    psi_panel = mo.vstack([mo.md("### Psi(z, t)"), mo.as_html(_psi_figure)], align="stretch")
    return (psi_panel,)


@app.cell
def _(MaxNLocator, fft_amplitudes, fft_dominant_mode, fft_mode_numbers, fft_time_index, mo, plt):
    final_dominant_mode = int(fft_dominant_mode[fft_time_index])
    _, _fft_axis = plt.subplots(constrained_layout=True)
    _fft_axis.plot(fft_mode_numbers[1:], fft_amplitudes[fft_time_index, 1:], color="#2563eb")
    _fft_axis.scatter([final_dominant_mode], [fft_amplitudes[fft_time_index, final_dominant_mode]], color="#dc2626", label=f"dominant mode {final_dominant_mode}")
    _fft_axis.set(xlabel="Mode number n", ylabel=r"$A_n(t) = |\delta\hat{\psi}_n(t)|$", title="Final-time ψ FFT amplitude")
    _fft_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    _fft_axis.legend()
    fft_panel = mo.vstack([mo.md("### FFT Amplitude"), mo.ui.matplotlib(_fft_axis)], align="stretch")
    return (fft_panel,)


@app.cell
def _(fft_amplitudes, fft_selected_mode, fft_time_index, inspect_time, mo, np, plt):
    fft_mode_amplitude = np.asarray(fft_amplitudes[:, fft_selected_mode], dtype=np.float64)
    log_amplitude = np.log(np.clip(fft_mode_amplitude, np.finfo(np.float64).tiny, None))
    _, _growth_axis = plt.subplots(constrained_layout=True)
    _growth_axis.plot(inspect_time, log_amplitude, color="#059669")
    _growth_axis.scatter([inspect_time[fft_time_index]], [log_amplitude[fft_time_index]], color="#dc2626")
    _growth_axis.set(xlabel=r"$t\;[s]$", ylabel=r"$\ln A_n(t)$", title=f"Growth of mode {fft_selected_mode}")
    fft_growth_panel = mo.vstack([mo.md("### Growth Rate"), mo.ui.matplotlib(_growth_axis)], align="stretch")
    return fft_growth_panel, fft_mode_amplitude


@app.cell
def _(fft_amplitudes, fft_time_index, fft_wavelengths, inspect_time, mo, np, plt):
    fft_dominant_mode = 1 + np.argmax(fft_amplitudes[:, 1:], axis=1)
    dominant_wavelength = fft_wavelengths[fft_dominant_mode]
    _, _dominant_axis = plt.subplots(constrained_layout=True)
    _dominant_axis.plot(inspect_time, 100 * dominant_wavelength, color="#7c3aed", drawstyle="steps-mid")
    _dominant_axis.scatter([inspect_time[fft_time_index]], [100 * dominant_wavelength[fft_time_index]], color="#dc2626")
    _dominant_axis.set(xlabel=r"$t\;[s]$", ylabel=r"$\lambda_{\mathrm{dom}}(t)\;[\mathrm{cm}]$", title="Dominant wavelength")
    fft_dominant_panel = mo.vstack([mo.md("### Dominant Wavelength"), mo.ui.matplotlib(_dominant_axis)], align="stretch")
    return fft_dominant_mode, dominant_wavelength, fft_dominant_panel


@app.cell
def _(fft_amplitudes, fft_time_index, fft_wavelengths, inspect_time, mo, np, plt):
    log_amplitudes = np.log(np.clip(fft_amplitudes[:, 1:], np.finfo(np.float64).tiny, None))
    fft_log_growth_rates = np.gradient(log_amplitudes, inspect_time, axis=0)
    fft_fastest_mode = 1 + np.argmax(fft_log_growth_rates, axis=1)
    fastest_wavelength = fft_wavelengths[fft_fastest_mode]
    _, _fastest_axis = plt.subplots(constrained_layout=True)
    _fastest_axis.plot(inspect_time, 100 * fastest_wavelength, color="#ea580c", drawstyle="steps-mid")
    _fastest_axis.scatter([inspect_time[fft_time_index]], [100 * fastest_wavelength[fft_time_index]], color="#dc2626")
    _fastest_axis.set(xlabel=r"$t\;[s]$", ylabel=r"$\lambda_{\mathrm{fast}}(t)\;[\mathrm{cm}]$", title="Fastest-growing wavelength")
    fft_fastest_panel = mo.vstack([mo.md("### Fastest-Growing Wavelength"), mo.ui.matplotlib(_fastest_axis)], align="stretch")
    return fft_fastest_mode, fastest_wavelength, fft_fastest_panel, fft_log_growth_rates


@app.cell(hide_code=True)
def _(
    delta_dominant_wavelength_panel,
    delta_psi_panel,
    fft_panel,
    mo,
    phi_panel,
    psi_panel,
    selected_summary,
):
    mo.vstack(
        [
            selected_summary,
            mo.hstack([phi_panel, psi_panel], align="start", gap=1),
            mo.hstack([delta_psi_panel, fft_panel], align="start", gap=1),
            delta_dominant_wavelength_panel,
        ],
        align="stretch",
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
