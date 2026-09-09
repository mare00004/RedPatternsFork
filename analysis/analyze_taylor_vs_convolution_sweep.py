import marimo

__generated_with = "0.24.0"
app = marimo.App(width="wide")


@app.cell
def _():
    from io import BytesIO
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import TwoSlopeNorm

    from red_patterns import (
        RunData,
        SweepCatalog,
        get_rbc_cmap,
        selected_sweep_catalog,
        sweep_directory_picker,
    )
    from red_patterns.models import ConvRun, TaylorRun

    return (
        ConvRun,
        BytesIO,
        Path,
        RunData,
        SweepCatalog,
        TaylorRun,
        TwoSlopeNorm,
        alt,
        get_rbc_cmap,
        mo,
        np,
        pd,
        selected_sweep_catalog,
        sweep_directory_picker,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Compare Taylor $\nu$–$\mu$ sweep to convolution

    Choose a mixed sweep directory containing `runs.jsonl` and
    `results/<run_id>/run.h5`. The sweep must contain exactly one convolution
    run; every Taylor field is compared to it using the grid-normalized $L^2$
    difference (RMS over the saved $t$--$z$ grid):

    $$D_{\mathrm{grid}} = \sqrt{\frac{1}{N_t N_z}\sum_{i=1}^{N_t}\sum_{j=1}^{N_z}\left|\psi_{\mathrm{Taylor}}(t_i,z_j)-\psi_{\mathrm{conv}}(t_i,z_j)\right|^2}.$$

    Click a heatmap cell to inspect the full fields and their signed difference.
    """)
    return


@app.cell
def _(Path, mo, sweep_directory_picker):
    ui_sweep_dir = sweep_directory_picker(
        mo,
        initial_path=Path.cwd(),
        label="Choose mixed Taylor/convolution sweep directory",
    )
    ui_sweep_dir
    return (ui_sweep_dir,)


@app.cell
def _(RunData, np):
    def grid_normalized_l2_difference(reference_h5: str, comparison_h5: str):
        """Load two fields and return their strict-grid RMS L2 difference."""
        reference = RunData.from_h5(reference_h5, load_fields=False)
        comparison = RunData.from_h5(comparison_h5, load_fields=False)
        psi_reference = np.asarray(reference.load_psi(), dtype=np.float64)
        psi_comparison = np.asarray(comparison.load_psi(), dtype=np.float64)
        time_reference = np.asarray(reference.time, dtype=np.float64)
        time_comparison = np.asarray(comparison.time, dtype=np.float64)
        z_reference = np.asarray(reference.z, dtype=np.float64)
        z_comparison = np.asarray(comparison.z, dtype=np.float64)

        if psi_reference.ndim != 2 or psi_comparison.ndim != 2:
            raise ValueError("Both psi fields must be two-dimensional.")
        if time_reference.ndim != 1 or time_comparison.ndim != 1:
            raise ValueError("Both time coordinates must be one-dimensional.")
        if z_reference.ndim != 1 or z_comparison.ndim != 1:
            raise ValueError("Both z coordinates must be one-dimensional.")
        if psi_reference.shape != (time_reference.size, z_reference.size):
            raise ValueError("Convolution psi shape does not match its coordinates.")
        if psi_comparison.shape != (time_comparison.size, z_comparison.size):
            raise ValueError("Taylor psi shape does not match its coordinates.")
        if psi_reference.shape != psi_comparison.shape:
            raise ValueError(
                f"psi shape mismatch: {psi_comparison.shape} vs {psi_reference.shape}."
            )
        if not np.allclose(time_reference, time_comparison, rtol=1e-6, atol=1e-9):
            raise ValueError("Taylor and convolution time coordinates differ.")
        if not np.allclose(z_reference, z_comparison, rtol=1e-6, atol=1e-9):
            raise ValueError("Taylor and convolution z coordinates differ.")
        if not np.all(np.isfinite(psi_reference)) or not np.all(
            np.isfinite(psi_comparison)
        ):
            raise ValueError("psi contains NaN or infinite values.")

        grid_size = psi_reference.size
        if grid_size == 0:
            raise ValueError("psi grid is empty.")
        difference = float(
            np.linalg.norm(psi_comparison - psi_reference) / np.sqrt(grid_size)
        )
        return difference, psi_reference, psi_comparison, time_reference, z_reference

    return (grid_normalized_l2_difference,)


@app.cell
def _(ConvRun, SweepCatalog, TaylorRun, pd):
    def scan_sweep(catalog: SweepCatalog):
        """Return the convolution reference and Taylor metadata from one sweep."""
        runs = catalog.runs
        convolution_runs = [run for run in runs if isinstance(run, ConvRun)]
        if len(convolution_runs) != 1:
            raise ValueError(
                f"Expected exactly one convolution run, found {len(convolution_runs)}."
            )

        reference = convolution_runs[0]
        reference_h5 = next(
            entry.run_h5
            for entry in catalog.entries
            if entry.run_id == reference.run_id
        )
        rows = []
        for entry in catalog.entries:
            run = entry.run
            if not isinstance(run, TaylorRun):
                continue
            run_h5 = entry.run_h5
            rows.append(
                {
                    "run_id": run.run_id,
                    "NU": float(run.NU),
                    "MU": float(run.MU),
                    "run_h5": str(run_h5),
                    "h5_exists": entry.h5_exists,
                    "comparison_status": "pending",
                }
            )
        dataframe = pd.DataFrame(
            rows,
            columns=[
                "run_id",
                "NU",
                "MU",
                "run_h5",
                "h5_exists",
                "comparison_status",
            ],
        )
        if not dataframe.empty:
            dataframe = dataframe.sort_values(["MU", "NU", "run_id"], kind="stable")
            dataframe = dataframe.reset_index(drop=True)
        return reference.run_id, reference_h5, dataframe

    return (scan_sweep,)


@app.cell
def _(mo, np, pd, scan_sweep, selected_sweep_catalog, ui_sweep_dir):
    is_script_mode = mo.app_meta().mode == "script"
    selected_path = ui_sweep_dir.path(0) if ui_sweep_dir.value else None
    if is_script_mode:
        script_time = np.linspace(0.0, 60.0, 121)
        script_z = np.linspace(0.0, 0.06, 160)
        amplitude = 0.004 * (1.0 - np.exp(-script_time[:, None] / 15.0))
        reference_psi = 0.02 + amplitude * np.sin(
            2.0 * np.pi * script_z[None, :] / 0.012
        )
        synthetic_rows = []
        synthetic_taylor_fields = {}
        for _synthetic_run_id, nu, mu, shift in (
            ("synthetic-1", 1e-32, 1e-39, 0.00005),
            ("synthetic-2", 1e-31, 1e-38, 0.0002),
            ("synthetic-3", 1e-30, 1e-37, 0.0005),
            ("synthetic-4", 1e-29, 1e-36, 0.001),
        ):
            taylor_psi = 0.02 + amplitude * np.sin(
                2.0 * np.pi * (script_z[None, :] - shift) / 0.012
            )
            synthetic_taylor_fields[_synthetic_run_id] = taylor_psi
            synthetic_rows.append(
                {
                    "run_id": _synthetic_run_id,
                    "NU": nu,
                    "MU": mu,
                    "run_h5": None,
                    "h5_exists": True,
                    "comparison_status": "ready",
                    "grid_normalized_l2": float(
                        np.linalg.norm(taylor_psi - reference_psi)
                        / np.sqrt(reference_psi.size)
                    ),
                    "max_difference_pct": float(
                        np.max(np.abs(100.0 * (taylor_psi - reference_psi)))
                    ),
                }
            )
        comparison_df = pd.DataFrame(synthetic_rows)
        reference_id = "synthetic convolution"
        reference_h5 = None
        synthetic_reference = (reference_psi, script_time, script_z)
        status = mo.md("Script mode uses a small synthetic mixed sweep.")
    elif selected_path is None:
        comparison_df = None
        reference_id = None
        reference_h5 = None
        synthetic_reference = None
        synthetic_taylor_fields = None
        status = mo.md("Waiting for a sweep directory selection...")
    else:
        catalog, status = selected_sweep_catalog(mo, ui_sweep_dir)
        if catalog is None:
            comparison_df = None
            reference_id = None
            reference_h5 = None
            synthetic_reference = None
            synthetic_taylor_fields = None
        else:
            try:
                reference_id, reference_h5, comparison_df = scan_sweep(catalog)
            except ValueError as error:
                comparison_df = None
                reference_id = None
                reference_h5 = None
                synthetic_reference = None
                synthetic_taylor_fields = None
                status = mo.callout(str(error), kind="danger")
            else:
                synthetic_reference = None
                synthetic_taylor_fields = None
                status = mo.md(
                    f"Loaded convolution reference `{reference_id}` and "
                    f"{len(comparison_df)} Taylor runs from `{catalog.root}`."
                )
    status
    return (
        comparison_df,
        reference_h5,
        reference_id,
        synthetic_reference,
        synthetic_taylor_fields,
    )


@app.cell
def _(comparison_df, grid_normalized_l2_difference, mo, reference_h5):
    mo.stop(comparison_df is None, mo.md("Select a valid mixed sweep to continue."))
    assert comparison_df is not None
    mo.stop(comparison_df.empty, mo.md("The selected sweep contains no Taylor runs."))
    results = comparison_df.copy()
    if reference_h5 is not None and not reference_h5.is_file():
        results["comparison_status"] = "convolution run.h5 is missing"
        results["grid_normalized_l2"] = float("nan")
        results["max_difference_pct"] = float("nan")
    elif reference_h5 is not None:
        results["grid_normalized_l2"] = float("nan")
        results["max_difference_pct"] = float("nan")
        for index, row in results.iterrows():
            if not bool(row["h5_exists"]):
                results.at[index, "comparison_status"] = "Taylor run.h5 is missing"
                continue
            try:
                difference, _psi_reference, _psi_taylor, _, _ = grid_normalized_l2_difference(
                    str(reference_h5), row["run_h5"]
                )
                results.at[index, "grid_normalized_l2"] = difference
                results.at[index, "max_difference_pct"] = float(
                    np.max(np.abs(100.0 * (_psi_taylor - _psi_reference)))
                )
                results.at[index, "comparison_status"] = "ready"
            except (KeyError, OSError, TypeError, ValueError) as error:
                results.at[index, "comparison_status"] = str(error)
    results["NU_label"] = results["NU"].map(lambda value: f"{value:.3e}")
    results["MU_label"] = results["MU"].map(lambda value: f"{value:.3e}")
    positive = results.loc[results["grid_normalized_l2"] > 0.0, "grid_normalized_l2"]
    display_floor = float(positive.min() / 10.0) if not positive.empty else 1e-16
    results["grid_normalized_l2_for_color"] = results["grid_normalized_l2"].mask(
        results["grid_normalized_l2"].eq(0.0), display_floor
    )
    finite_difference_maxima = results.loc[
        np.isfinite(results["max_difference_pct"]), "max_difference_pct"
    ]
    difference_limit = (
        max(float(finite_difference_maxima.max()), 1e-12)
        if not finite_difference_maxima.empty
        else 1e-12
    )
    comparison_results = results
    return comparison_results, difference_limit, display_floor


@app.cell(hide_code=True)
def _(comparison_results, difference_limit, display_floor, mo, reference_id):
    ready = int((comparison_results["comparison_status"] == "ready").sum())
    mo.vstack(
        [
            mo.md(
                f"## Results\n\nReference convolution run: `{reference_id}`. "
                f"{ready}/{len(comparison_results)} Taylor runs are comparable."
            ),
            mo.callout(
                mo.md(
                    "Color encodes grid-normalized $L^2$ difference on a logarithmic scale. "
                    f"Exact zero is displayed at the positive floor `{display_floor:.3e}` "
                    "but remains zero in the tooltip. The signed-difference plots use "
                    f"the fixed sweep-wide range ±`{difference_limit:.3e}` percentage points."
                ),
                kind="info",
            ),
        ],
        align="stretch",
    )
    return


@app.cell
def _(alt, comparison_results, mo):
    click = alt.selection_point(fields=["run_id"], empty=False)
    heatmap = (
        alt.Chart(comparison_results)
        .mark_rect(stroke="black", strokeWidth=0.5)
        .encode(
            x=alt.X("NU_label:O", title="ν [J m³]", sort=alt.SortField(field="NU")),
            y=alt.Y("MU_label:O", title="μ [J m⁵]", sort=alt.SortField(field="MU")),
            color=alt.Color(
                "grid_normalized_l2_for_color:Q",
                title="L₂ difference",
                scale=alt.Scale(type="log", scheme="viridis"),
            ),
            opacity=alt.condition(click, alt.value(1.0), alt.value(0.45)),
            tooltip=[
                alt.Tooltip("run_id:N", title="Taylor run"),
                alt.Tooltip("NU:Q", title="ν [J m³]", format=".3e"),
                alt.Tooltip("MU:Q", title="μ [J m⁵]", format=".3e"),
                alt.Tooltip(
                    "grid_normalized_l2:Q",
                    title="L₂ difference",
                    format=".6g",
                ),
                alt.Tooltip("comparison_status:N", title="comparison status"),
                alt.Tooltip("run_h5:N", title="Taylor run.h5"),
            ],
        )
        .add_params(click)
        .properties(
            width=520,
            height=440,
            title="Taylor vs. Convolution L₂ difference",
        )
    )
    ui_heatmap = mo.ui.altair_chart(heatmap)
    ui_heatmap
    return (ui_heatmap,)


@app.cell
def _(comparison_results, mo, ui_heatmap):
    def selected_run_id(selections):
        payload = next(iter(selections.values()), None) if isinstance(selections, dict) else selections
        if isinstance(payload, list):
            payload = payload[0] if payload else None
        if hasattr(payload, "iloc") and len(payload):
            payload = payload.iloc[0].to_dict()
        if hasattr(payload, "to_dict") and not isinstance(payload, dict):
            payload = payload.to_dict()
        if not isinstance(payload, dict):
            return None
        run_id = payload.get("run_id")
        return run_id[0] if isinstance(run_id, list) and run_id else run_id

    selected_id = selected_run_id(ui_heatmap.selections)
    mo.stop(selected_id is None, mo.md("Click a heatmap cell to inspect that Taylor run."))
    selected_rows = comparison_results[comparison_results["run_id"] == selected_id]
    mo.stop(selected_rows.empty, mo.md(f"No run found for selected ID `{selected_id}`."))
    selected_row = selected_rows.iloc[0]
    mo.stop(
        selected_row["comparison_status"] != "ready",
        mo.callout(
            f"This run cannot be inspected: {selected_row['comparison_status']}", kind="warn"
        ),
    )
    return (selected_row,)


@app.cell
def _(
    reference_h5,
    grid_normalized_l2_difference,
    selected_row,
    synthetic_reference,
    synthetic_taylor_fields,
):
    if synthetic_reference is not None:
        psi_reference, time, z = synthetic_reference
        assert synthetic_taylor_fields is not None
        psi_taylor = synthetic_taylor_fields[selected_row["run_id"]]
    else:
        assert reference_h5 is not None
        _, psi_reference, psi_taylor, time, z = grid_normalized_l2_difference(
            str(reference_h5), selected_row["run_h5"]
        )
    return psi_reference, psi_taylor, time, z


@app.cell
def _(BytesIO, TwoSlopeNorm, difference_limit, get_rbc_cmap, mo, np, plt, psi_reference, psi_taylor, selected_row, time, z):
    difference_pct = 100.0 * (psi_taylor - psi_reference)
    extent = (float(time[0]), float(time[-1]), float(100.0 * z[0]), float(100.0 * z[-1]))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    source_images = []
    for axis, values, title in zip(
        axes[:2],
        (psi_reference, psi_taylor),
        (r"Convolution: $\psi(z,t)$", r"Taylor: $\psi(z,t)$"),
        strict=True,
    ):
        image = axis.imshow(
            (100.0 * values).T, origin="lower", aspect="auto", interpolation="nearest",
            extent=extent, vmin=0.0, vmax=100.0, cmap=get_rbc_cmap(),
        )
        source_images.append(image)
        axis.set(title=title, xlabel=r"$t\;[\mathrm{s}]$", ylabel=r"$z\;[\mathrm{cm}]$")
    difference_image = axes[2].imshow(
        difference_pct.T, origin="lower", aspect="auto", interpolation="nearest", extent=extent,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(
            vmin=-difference_limit, vcenter=0.0, vmax=difference_limit
        ),
    )
    axes[2].set(
        title=r"Difference: $\psi_\mathrm{Taylor}-\psi_\mathrm{conv}$",
        xlabel=r"$t\;[\mathrm{s}]$", ylabel=r"$z\;[\mathrm{cm}]$",
    )
    figure.colorbar(source_images[0], ax=axes[:2], shrink=0.9, pad=0.02, label=r"$\psi\;[\%]$")
    figure.colorbar(difference_image, ax=axes[2], shrink=0.9, pad=0.02, label=r"$\Delta\psi$ [percentage points]")
    figure_png = BytesIO()
    figure.savefig(figure_png, format="png", dpi=300, bbox_inches="tight")
    download_figure = mo.download(
        data=figure_png.getvalue(),
        filename=f"{selected_row['run_id']}_taylor_vs_convolution.png",
        mimetype="image/png",
        label="Download selected-run figure (PNG)",
    )
    mo.vstack(
        [
            mo.md(
                f"## Selected Taylor run `{selected_row['run_id']}`\n\n"
                f"$\\nu={selected_row['NU']:.3e}\\,\\mathrm{{J\\,m^3}}$, "
                f"$\\mu={selected_row['MU']:.3e}\\,\\mathrm{{J\\,m^5}}$, "
                f"grid-normalized $L^2$ difference = "
                f"`{selected_row['grid_normalized_l2']:.6g}`."
            ),
            mo.ui.matplotlib(axes[0]),
            download_figure,
        ],
        align="stretch",
    )
    return


if __name__ == "__main__":
    app.run()
