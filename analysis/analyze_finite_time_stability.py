import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


with app.setup:
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.fft import idct

    from red_patterns import get_rbc_cmap
    from red_patterns.dct_dispersion import read_run
    from red_patterns.finite_time_stability import (
        analyze_window, demo_sweep, phase_coordinates, prepare_pair, theory_overlays,
    )
    from red_patterns.stability_sweep import ridge_pairs
    from red_patterns.sweep_catalog import selected_sweep_catalog, sweep_directory_picker


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Paired ν, μ sweep: wavelengths and finite-time growth

    Select a sweep containing **runs.jsonl** and **results/**. One time window and
    one inclusive DCT-II mode range apply to every loaded pair (default modes 1–100).

    **Wavelength map:** fit $\ln|\mathrm{DCT}_m(\psi_\mathrm{perturbed})|$
    against time and show $\ell_m=2N\Delta z/m$ for the fastest resolved growing
    mode. This uses the perturbed field directly, so an evolving background can
    affect its modal fits.

    **Paired growth map:** subtract the unperturbed field at each saved time,
    then compute
    $D(t)=\sqrt{\sum_{m=m_\min}^{m_\max}|\mathrm{DCT}_m[
    \psi_\mathrm{perturbed}(t)-\psi_\mathrm{unperturbed}(t)]|^2}$ and
    $\lambda_\mathrm{eff}=[\ln D(t_\mathrm{end})-\ln D(t_\mathrm{start})]/
    (t_\mathrm{end}-t_\mathrm{start})$.
    Leave the start at 0 for the requested $\lambda_\mathrm{eff}(0,t_\mathrm{end})$.
    Positive values mean net growth; negative values mean net decay over this window.
    """)
    return


@app.cell
def _():
    directory = sweep_directory_picker(mo, initial_path=Path.cwd(), label="Paired Taylor sweep directory")
    refresh = mo.ui.run_button(label="Refresh saved results")
    demo = mo.ui.checkbox(value=mo.app_meta().mode == "script", label="Synthetic validation sweep")
    mo.vstack([directory, mo.hstack([refresh, demo])])
    return demo, directory, refresh


@app.cell
def _(demo, directory, refresh):
    refresh.value
    if demo.value:
        pairs, prepared = demo_sweep()
        _message = mo.md("**Synthetic data:** directly excited cosine modes with analytic growth rates.")
    else:
        _catalog, _message = selected_sweep_catalog(mo, directory)
        pairs = ridge_pairs(_catalog) if _catalog is not None else []
        prepared = {}
        for _pair in mo.status.progress_bar(pairs, title="Loading paired fields and caching DCTs"):
            if _pair["status"] == "ready":
                try:
                    prepared[_pair["run_id"]] = prepare_pair(
                        read_run(_pair["perturbed_path"]), read_run(_pair["equilibrium_path"])
                    )
                except (OSError, KeyError, ValueError) as _exc:
                    _pair["status"] = f"Cannot load: {_exc}"
    _availability = pd.DataFrame(pairs).drop(columns=["perturbed_path", "equilibrium_path"], errors="ignore")
    mo.vstack([_message, mo.md(f"**{len(prepared)} / {len(pairs)} pairs loaded.**"),
               mo.accordion({"Pair availability": mo.ui.table(_availability, selection=None)})])
    return pairs, prepared


@app.cell
def _(pairs, prepared):
    mo.stop(not prepared, mo.md("Select a sweep with complete Taylor LINEAR_FULL ridge pairs."))
    common_time = next(iter(prepared.values()))["time"]
    for _data in prepared.values():
        common_time = np.intersect1d(common_time, _data["time"])
    mo.stop(len(common_time) < 2, mo.callout(
        "Loaded pairs need at least two common saved times for a shared window. "
        "No time interpolation or endpoint substitution is performed.", kind="warn"))
    _max_mode = min(_data["raw"].shape[1] - 1 for _data in prepared.values())
    _ready = [_p for _p in pairs if _p["run_id"] in prepared]
    _low = min(max(1, min(_p["mode_min"] for _p in _ready)), _max_mode)
    _high = min(max(_p["mode_max"] for _p in _ready), _max_mode)
    _default_end = common_time[min(len(common_time) - 1, max(1, np.searchsorted(common_time, 5.0)))]
    window = mo.ui.range_slider(
        steps=common_time.tolist(), value=[float(common_time[0]), float(_default_end)],
        label="Shared linear regime [s]: start → end", show_value=True, full_width=True, debounce=True,
    )
    modes = mo.ui.range_slider(
        start=0, stop=_max_mode, step=1, value=[_low, max(_low, _high)],
        label="Inclusive DCT mode range (both maps and D(t))",
        show_value=True, full_width=True, debounce=True,
    )
    mo.vstack([window, modes, mo.md(
        "Slider endpoints are saved times shared by all loaded pairs. "
        "All saved frames inside the window enter each modal fit. "
        "DCTs are cached, so changing these sliders does not reread HDF5 files."
    )])
    return modes, window


@app.cell
def _(modes, pairs, prepared, window):
    mo.stop(window.value[0] >= window.value[1], mo.md("Choose distinct start and end times."))
    results = {
        _id: analyze_window(_data, *window.value, *map(int, modes.value))
        for _id, _data in prepared.items()
    }
    _rows = []
    for _pair in pairs:
        _row = {k: v for k, v in _pair.items() if not k.endswith("_path")}
        # Do not confuse the seeded band with the selected analysis band.
        _row["seeded_mode_min"] = _row.pop("mode_min")
        _row["seeded_mode_max"] = _row.pop("mode_max")
        _row.update(dict(fastest_wavelength_cm=np.nan, lambda_eff_s_inv=np.nan,
                         wavelength_status=_pair["status"], norm_status=_pair["status"]))
        if _pair["run_id"] in results:
            _row.update(results[_pair["run_id"]]["summary"])
            _row["notes"] = "; ".join(prepared[_pair["run_id"]]["notes"])
        _rows.append(_row)
    summary = pd.DataFrame(_rows)
    summary["nu_scaled"] = summary["NU"] / 1e-30
    summary["mu_scaled"] = summary["MU"] / 1e-37
    return results, summary


@app.cell
def _():
    coordinates = mo.ui.dropdown(
        options={
            "Normalized onset: |ν| / |ν_c(μ)| versus −μ": "normalized",
            "Squared onset: ν² versus −μ": "squared",
            "Original parameters: ν versus μ": "raw",
        },
        value="Normalized onset: |ν| / |ν_c(μ)| versus −μ",
        label="Phase-space coordinates",
    )
    mo.vstack([coordinates, mo.md(r"""
    For the saved Taylor model,
    $\sigma(k)=-\alpha g-\beta\bar\psi\nu k^2+\beta\bar\psi\mu k^4$.
    A fixed mode is neutral at
    $\nu=-\alpha g/(\beta\bar\psi k^2)+\mu k^2$.
    Maximizing over continuous $k>0$, for $\nu,\mu<0$, gives
    $\nu_c^2=C(-\mu)$ with $C=4\alpha g/(\beta\bar\psi)$ and $g=30/L$.

    Thus **ν² versus −μ has a straight predicted boundary** when C is fixed.
    **|ν|/|ν_c| has its predicted boundary at 1**: values above 1 predict
    instability. These are continuum predictions; the selected discrete DCT
    band and finite-time norm can have different apparent boundaries.
    """)])
    return (coordinates,)


@app.cell
def _(coordinates, summary):
    phase_data, axis_titles = phase_coordinates(summary, coordinates.value)
    return axis_titles, phase_data


@app.cell
def _():
    contour_levels = mo.ui.multiselect(
        options={f"{_value:g} cm": _value for _value in
                 [0.1, 0.14, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0, 10.0, 14.0]},
        value=["0.3 cm", "0.5 cm", "0.7 cm", "1 cm"],
        label="Predicted fastest-wavelength contours",
    )
    contour_levels
    return (contour_levels,)


@app.cell
def _(contour_levels, coordinates, modes, phase_data):
    overlay_data = theory_overlays(
        phase_data, coordinates.value, *map(int, modes.value), contour_levels.value,
    )
    return (overlay_data,)


@app.cell
def _(axis_titles, contour_levels, overlay_data, phase_data):
    _click = alt.selection_point(
        name="phase_pair", fields=["nu_scaled", "mu_scaled"], toggle=False, clear="dblclick",
    )
    _visible = phase_data[np.isfinite(phase_data[["phase_x", "phase_y"]]).all(axis=1)]
    _omitted = len(phase_data) - len(_visible)
    _onsets = overlay_data[overlay_data["kind"] == "onset"]
    _contours = overlay_data[overlay_data["kind"] == "wavelength"]
    _labels = _contours[
        _contours.groupby("curve_id").cumcount()
        == _contours.groupby("curve_id")["curve"].transform("size") // 2
    ]
    _shown_levels = set(_contours["wavelength_cm"].dropna())
    _hidden_levels = [f"{_level:g} cm" for _level in contour_levels.value if _level not in _shown_levels]
    _x_scale = alt.Scale(zero=False)
    _y_scale = alt.Scale(zero=False)
    if not _visible.empty:
        _x_scale = alt.Scale(domain=[float(_visible["x_low"].min()), float(_visible["x_high"].max())], nice=False)
        _y_scale = alt.Scale(domain=[float(_visible["y_low"].min()), float(_visible["y_high"].max())], nice=False)

    def phase_chart(metric, title, color_title, status):
        _valid = _visible[np.isfinite(_visible[metric])]
        # Vega's invalid-color handling can override a conditional fallback.
        # Give only the color scale a finite placeholder inside its valid range;
        # unresolved measurements retain NaN in tables/tooltips and draw grey.
        _plot_data = _visible.assign(
            measurement_available=np.isfinite(_visible[metric]),
            render_value=_visible[metric].fillna(float(_valid[metric].min()) if len(_valid) else 0.0),
        )
        _tooltips = [
            alt.Tooltip("run_id:N"), alt.Tooltip("reference_id:N"),
            alt.Tooltip("NU:Q", format=".6e"), alt.Tooltip("MU:Q", format=".6e"),
            alt.Tooltip("phase_x:Q", title=axis_titles[0], format=".6g"),
            alt.Tooltip("phase_y:Q", title=axis_titles[1], format=".6g"),
            alt.Tooltip("critical_ratio:Q", title="|ν| / |ν_c|", format=".6g"),
            alt.Tooltip("theory_onset_nu:Q", title="Selected-mode onset ν", format=".6e"),
            alt.Tooltip("theory_growing_modes:Q", title="Predicted growing modes"),
            alt.Tooltip("theory_fastest_mode:Q", title="Predicted fastest mode"),
            alt.Tooltip("theory_fastest_wavelength_cm:Q", title="Predicted fastest wavelength [cm]", format=".5g"),
            alt.Tooltip("theory_max_rate_s_inv:Q", title="Predicted maximum rate [s⁻¹]", format=".5g"),
            alt.Tooltip("seed:Q"), alt.Tooltip("epsilon:Q", format=".3e"),
            alt.Tooltip("start_s:Q"), alt.Tooltip("end_s:Q"),
            alt.Tooltip("mode_min:Q"), alt.Tooltip("mode_max:Q"),
            alt.Tooltip("fastest_mode:Q"), alt.Tooltip(metric + ":Q", format=".6g"),
            alt.Tooltip(status + ":N"),
        ]
        _scale = alt.Scale(scheme="viridis")
        if metric == "lambda_eff_s_inv":
            _limit = max(float(_valid[metric].abs().max()), 1e-12) if len(_valid) else 1.0
            _scale = alt.Scale(scheme="redblue", domain=[-_limit, _limit], domainMid=0, reverse=True)
        _cells = alt.Chart(_plot_data).mark_rect(
            stroke="#333", strokeWidth=0.5, invalid=None,
        ).encode(
            x=alt.X("x_low:Q", title=axis_titles[0], scale=_x_scale),
            x2="x_high:Q",
            y=alt.Y("y_low:Q", title=axis_titles[1], scale=_y_scale),
            y2="y_high:Q",
            color=alt.condition(
                alt.datum.measurement_available,
                alt.Color("render_value:Q", title=color_title, scale=_scale),
                alt.value("#bdbdbd"),
            ),
            opacity=alt.condition(_click, alt.value(1.0), alt.value(0.35)),
            tooltip=_tooltips,
        ).properties(name=metric)
        _chart = _cells
        if not _onsets.empty:
            _boundary = alt.Chart(_onsets).mark_line(
                color="black", strokeWidth=2, clip=True,
            ).encode(
                x="phase_x:Q", y="phase_y:Q", detail="curve_id:N", order="point_order:Q",
                strokeDash=alt.StrokeDash("curve:N", title="Predicted onset", scale=alt.Scale(
                    domain=["Continuum onset", "Selected-mode onset"], range=[[6, 4], [1, 0]],
                )),
                tooltip=[alt.Tooltip("curve:N"), alt.Tooltip("dct_length_m:Q", title="L [m]"),
                         alt.Tooltip("boundary_coefficient:Q", title="C", format=".5e")],
            )
            _chart = _chart + _boundary
        if not _contours.empty:
            _iso = alt.Chart(_contours).mark_line(
                color="#f97316", strokeWidth=2, strokeDash=[3, 3], clip=True,
            ).encode(x="phase_x:Q", y="phase_y:Q", detail="curve_id:N", order="point_order:Q",
                     tooltip=[alt.Tooltip("curve:N"), alt.Tooltip("dct_length_m:Q", title="L [m]")])
            _text = alt.Chart(_labels).mark_text(
                color="#f97316", fontWeight="bold", fontSize=11,
                align="right", baseline="bottom", dx=-4, dy=-4, clip=True,
            ).encode(x="phase_x:Q", y="phase_y:Q", text="curve:N")
            _chart = _chart + _iso + _text
        return _chart.properties(width=470, height=360, title=title)

    # Compose first: Altair renames unit views when concatenating layers.
    # Register the selection once against those final names, avoiding duplicate
    # definitions and stale pre-composition names.
    _heatmaps = alt.hconcat(
        phase_chart("fastest_wavelength_cm", "Fastest growing wavelength: perturbed ψ",
                    "Wavelength [cm]", "wavelength_status"),
        phase_chart("lambda_eff_s_inv", "Paired finite-time effective growth",
                    "λ_eff [s⁻¹]", "norm_status"),
    ).resolve_scale(color="independent")
    _views = [
        _panel.layer[0].name if isinstance(_panel, alt.LayerChart) else _panel.name
        for _panel in _heatmaps.hconcat
    ]
    _heatmaps.params = [
        alt.TopLevelSelectionParameter(**_click.param.to_dict(), views=_views),
    ]
    phase_heatmap = mo.ui.altair_chart(
        _heatmaps, chart_selection=False, legend_selection=False,
    )
    mo.vstack([phase_heatmap,
               mo.md("**Click a cell in either heatmap** to inspect that ν, μ pair below; "
                     "double-click to clear the selection. **Grey cells:** unavailable or unresolved estimates. "
                     "**Dashed black:** continuum onset. **Solid black:** onset among selected positive DCT modes. "
                     "**Orange dotted:** labelled continuum fastest-wavelength contours, shown where the "
                     "optimum is inside the selected band and at least one selected mode grows. "
                     "The two onset curves can overlap closely. Axes use numeric spacing; cell edges lie "
                     "halfway between samples in each row and do not imply interpolated measurements. "
                     f"**{_omitted} rows omitted** from this view because its coordinates are undefined "
                     "(nonnegative ν/μ in transformed views, or unavailable normalization metadata). "
                     "The rate color scale is centered on zero: red = growth, blue = decay. "
                     "Multiple seeds at the same coordinates overlap; the selector below lists the runs in the clicked cell."),
               mo.md("Contours with no growing segment inside this view: " + ", ".join(_hidden_levels))
               if _hidden_levels else mo.md("All requested wavelength contours have a growing segment in this view."),
               mo.ui.table(phase_data, selection=None, page_size=10),
               mo.download(phase_data.to_csv(index=False).encode(), filename="finite_time_stability.csv")])
    return (phase_heatmap,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The zero-rate boundary is a **finite-time, selected-band** diagnostic.
    A decaying norm can coexist with a growing mode when other seeded modes decay
    more strongly. Compare nearby endpoints and mode ranges before interpreting
    the instability boundary. Grey wavelength markers do not establish stability.

    Modal fits need at least five saved samples, a consistent coefficient sign,
    log relative fit error ≤ 0.1, and a slope resolved above twice its fit standard
    error. Mode 0 has no finite wavelength. Zero or numerically unresolved norm
    endpoints have no reported effective rate; no epsilon is added.
    """)
    return


@app.cell
def _(phase_heatmap, results, summary):
    _selected_rows = phase_heatmap.apply_selection(summary)
    _available = _selected_rows[_selected_rows["run_id"].isin(results)]
    mo.stop(_available.empty, mo.callout(
        "This cell has no loaded pair to inspect. See its availability status above, "
        "or click another heatmap cell.", kind="warn"))
    _options = {
        f"{_p['run_id']} · ν={_p['NU']:.6e}, μ={_p['MU']:.6e} · seed={_p['seed']}": _p["run_id"]
        for _p in _available.to_dict("records")
    }
    available_run_ids = _available["run_id"].tolist()
    inspect_pair = mo.ui.dropdown(
        options=_options, value=next(iter(_options)), label="Inspect a run (filtered by heatmap selection)",
    )
    inspect_pair
    return available_run_ids, inspect_pair


@app.cell
def _(available_run_ids, inspect_pair):
    # A heatmap click directly changes this plain reactive value. The dropdown
    # only disambiguates multiple seeds, and cannot retain an unrelated run.
    selected_run_id = (
        inspect_pair.value if inspect_pair.value in available_run_ids else available_run_ids[0]
    )
    return (selected_run_id,)


@app.cell
def _(prepared, selected_run_id, window):
    _data = prepared[selected_run_id]
    _time_indices = np.unique(np.linspace(
        0, len(_data["time"]) - 1, min(2400, len(_data["time"])), dtype=int,
    ))
    _z_indices = np.unique(np.linspace(
        0, len(_data["z"]) - 1, min(1200, len(_data["z"])), dtype=int,
    ))
    # The cache retains every spatial DCT coefficient. Inverting all modes
    # recovers the full perturbed field independently of the analysis band.
    _psi = idct(_data["raw"][_time_indices], type=2, norm="ortho", axis=1)[:, _z_indices]
    psi_figure, _axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    _image = _axis.pcolormesh(
        _data["time"][_time_indices], 100 * _data["z"][_z_indices], 100 * _psi.T,
        shading="nearest", cmap=get_rbc_cmap(), vmin=0, vmax=100, rasterized=True,
    )
    psi_figure.colorbar(_image, ax=_axis, label="ψ [%]")
    for _endpoint in window.value:
        _axis.axvline(_endpoint, color="firebrick", linestyle="--", linewidth=1.5)
    _axis.set(
        title=f"{selected_run_id}: perturbed ψ(z, t)",
        xlabel="Time [s]", ylabel="z [cm]",
        xlim=(_data["time"][0], _data["time"][-1]),
    )
    _output = mo.as_html(psi_figure)
    plt.close(psi_figure)
    mo.vstack([
        _output,
        mo.md("Full saved evolution of the selected perturbed run. Dashed red lines mark "
              "the shared linear window; the field includes all spatial modes."),
    ])
    return (psi_figure,)


@app.cell
def _(prepared, results, selected_run_id):
    selected = results[selected_run_id]
    _data = prepared[selected_run_id]
    _modes = selected["modes"]
    _distance = np.hypot.reduce(np.abs(_data["difference"][:, _modes]), axis=1)
    _curve = pd.DataFrame({"time_s": _data["time"], "D": _distance})
    _curve = _curve[_curve["D"] > 0]
    _bounds = pd.DataFrame({"time_s": selected["time"][[0, -1]]})
    _norm = alt.Chart(_curve).mark_line().encode(
        x=alt.X("time_s:Q", title="Saved time [s]"),
        y=alt.Y("D:Q", title="Selected-band D(t)", scale=alt.Scale(type="log")),
        tooltip=[alt.Tooltip("time_s:Q"), alt.Tooltip("D:Q", format=".6e")],
    )
    _rules = alt.Chart(_bounds).mark_rule(color="firebrick", strokeDash=[5, 3]).encode(x="time_s:Q")
    _s = selected["summary"]
    _measured = (
        f"mode {_s['fastest_mode']:g}, wavelength {_s['fastest_wavelength_cm']:.5g} cm"
        if np.isfinite(_s["fastest_mode"]) else "no resolved growing mode"
    )
    _predicted = (
        f"mode {_s['theory_fastest_mode']:g}, wavelength {_s['theory_fastest_wavelength_cm']:.5g} cm"
        if np.isfinite(_s["theory_fastest_mode"])
        else "no growing selected mode" if np.isfinite(_s["theory_max_rate_s_inv"])
        else "unavailable for this run"
    )
    _endpoints = pd.DataFrame({"time_s": selected["time"][[0, -1]], "D": selected["distance"][[0, -1]]})
    _endpoints = _endpoints[_endpoints["D"] > 0]
    _secant = alt.Chart(_endpoints).mark_line(color="firebrick", point=True).encode(x="time_s:Q", y="D:Q")
    mo.vstack([
        mo.md(f"**{selected_run_id}: {_s['norm_status']}**, λ_eff = "
              f"**{_s['lambda_eff_s_inv']:.6g} s⁻¹**. "
              f"D(start) = {_s['D_start']:.6e}, D(end) = {_s['D_end']:.6e}."),
        mo.md(f"**Measured fastest:** {_measured}.  \n"
              f"**Predicted fastest in the selected band:** {_predicted}.  \n"
              f"Predicted maximum modal rate: {_s['theory_max_rate_s_inv']:.6g} s⁻¹. "
              "Predictions evaluate the continuum dispersion at the selected DCT wavenumbers; "
              "they do not include finite-difference stencil corrections."),
        mo.ui.altair_chart((_norm + _rules + _secant).properties(
            width=950, height=280, title=f"{selected_run_id}: paired norm; red endpoint secant defines λ_eff"
        ), chart_selection=False),
        mo.ui.table(selected["table"], selection=None, page_size=10),
        mo.download(pd.DataFrame({"time_s": selected["time"], "D": selected["distance"],
                                  "numerical_floor": selected["norm_floor"]}).to_csv(index=False).encode(),
                    filename=f"{selected_run_id}_distance.csv"),
    ])
    return (selected,)


@app.cell
def _(selected):
    _fastest = selected["summary"]["fastest_mode"]
    inspect_mode = mo.ui.slider(
        steps=selected["modes"].tolist(),
        value=int(_fastest) if np.isfinite(_fastest) else int(selected["modes"][0]),
        label="Inspect perturbed-field DCT mode", show_value=True,
    )
    inspect_mode
    return (inspect_mode,)


@app.cell
def _(inspect_mode, selected, selected_run_id):
    _index = int(inspect_mode.value - selected["modes"][0])
    _amplitude = np.abs(selected["raw"][:, _index])
    _log = np.full_like(_amplitude, np.nan)
    np.log(_amplitude, out=_log, where=_amplitude > 0)
    _fit = selected["fits"][_index]
    _frame = pd.DataFrame({
        "time_s": selected["time"], "Measured ln|c_m|": _log,
        "Log-linear fit": _fit.log_fitted if _fit is not None else np.nan,
    }).melt(id_vars="time_s", var_name="series", value_name="log_amplitude")
    _chart = alt.Chart(_frame).mark_line().encode(
        x=alt.X("time_s:Q", title="Time [s]"), y=alt.Y("log_amplitude:Q", title="ln|c_m|"),
        color="series:N",
    ).properties(
        width=950, height=250,
        title=f"{selected_run_id} · mode {inspect_mode.value}: {selected['table'].iloc[_index]['status']}",
    )
    mo.ui.altair_chart(_chart, chart_selection=False)
    return


if __name__ == "__main__":
    app.run()
