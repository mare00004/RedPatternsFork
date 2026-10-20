import marimo

__generated_with = "0.23.16"
app = marimo.App(width="wide")


with app.setup:
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    from red_patterns.dct_dispersion import read_run
    from red_patterns.finite_time_stability import analyze_window, demo_sweep, phase_coordinates, prepare_pair
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
def _(axis_titles, coordinates, phase_data, summary):
    _click = alt.selection_point(
        name="phase_pair", fields=["nu_scaled", "mu_scaled"], toggle=False, clear="dblclick",
    )
    _visible = phase_data[np.isfinite(phase_data[["phase_x", "phase_y"]]).all(axis=1)]
    _omitted = len(phase_data) - len(_visible)
    _boundary_rows = []
    for _coefficient, _group in phase_data[
        (phase_data["boundary_coefficient"] > 0) & (phase_data["MU"] < 0)
    ].groupby("boundary_coefficient"):
        for _mu in np.linspace(_group["MU"].min(), _group["MU"].max(), 100):
            _critical = -np.sqrt(_coefficient * -_mu)
            if coordinates.value == "raw":
                _x, _y = _critical / 1e-30, _mu / 1e-37
            else:
                _x, _y = (_critical / 1e-30) ** 2, -_mu / 1e-37
            _boundary_rows.append(dict(
                phase_x=_x, phase_y=_y, coefficient=f"{_coefficient:.12e}",
            ))
    _boundary_frame = pd.DataFrame(_boundary_rows)

    def phase_chart(metric, title, color_title, status):
        _valid = _visible[np.isfinite(_visible[metric])]
        _tooltips = [
            alt.Tooltip("run_id:N"), alt.Tooltip("reference_id:N"),
            alt.Tooltip("NU:Q", format=".6e"), alt.Tooltip("MU:Q", format=".6e"),
            alt.Tooltip("phase_x:Q", title=axis_titles[0], format=".6g"),
            alt.Tooltip("phase_y:Q", title=axis_titles[1], format=".6g"),
            alt.Tooltip("critical_ratio:Q", title="|ν| / |ν_c|", format=".6g"),
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
        _cells = alt.Chart(_visible).mark_rect(
            stroke="#333", strokeWidth=0.5, invalid=None,
        ).encode(
            x=alt.X("x_low:Q", title=axis_titles[0], scale=alt.Scale(zero=False)),
            x2="x_high:Q",
            y=alt.Y("y_low:Q", title=axis_titles[1], scale=alt.Scale(zero=False)),
            y2="y_high:Q",
            color=alt.condition(
                alt.expr.isValid(alt.datum[metric]),
                alt.Color(metric + ":Q", title=color_title, scale=_scale),
                alt.value("#bdbdbd"),
            ),
            opacity=alt.condition(_click, alt.value(1.0), alt.value(0.35)),
            tooltip=_tooltips,
        ).properties(name=metric).add_params(_click)
        if coordinates.value == "normalized":
            _boundary = alt.Chart(pd.DataFrame({"onset": [1.0]})).mark_rule(
                color="black", strokeDash=[6, 4], strokeWidth=2,
            ).encode(x="onset:Q")
            _chart = _cells + _boundary
        elif not _boundary_frame.empty:
            _boundary = alt.Chart(_boundary_frame).mark_line(
                color="black", strokeDash=[6, 4], strokeWidth=2,
            ).encode(x="phase_x:Q", y="phase_y:Q", detail="coefficient:N",
                     order="phase_y:Q", tooltip=alt.Tooltip("coefficient:N", title="C"))
            _chart = _cells + _boundary
        else:
            _chart = _cells
        return _chart.properties(width=470, height=360, title=title)

    # Attach the shared selection to the unit marks above before layering the
    # boundary. Altair then scopes it to the actual compiled views; manually
    # specifying top-level view names breaks when a unit becomes a layer.
    _heatmaps = alt.hconcat(
        phase_chart("fastest_wavelength_cm", "Fastest growing wavelength: perturbed ψ",
                    "Wavelength [cm]", "wavelength_status"),
        phase_chart("lambda_eff_s_inv", "Paired finite-time effective growth",
                    "λ_eff [s⁻¹]", "norm_status"),
    ).resolve_scale(color="independent")
    phase_heatmap = mo.ui.altair_chart(
        _heatmaps, chart_selection=False, legend_selection=False,
    )
    mo.vstack([phase_heatmap,
               mo.md("**Click a cell in either heatmap** to inspect that ν, μ pair below; "
                     "double-click to clear the selection. **Grey cells:** unavailable or unresolved estimates. "
                     "**Dashed black:** continuum prediction. Axes use numeric spacing; cell edges lie "
                     "halfway between samples in each row and do not imply interpolated measurements. "
                     f"**{_omitted} rows omitted** from this view because its coordinates are undefined "
                     "(nonnegative ν/μ in transformed views, or unavailable normalization metadata). "
                     "The rate color scale is centered on zero: red = growth, blue = decay. "
                     "Multiple seeds at the same coordinates overlap; the selector below lists the runs in the clicked cell."),
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
    inspect_pair = mo.ui.dropdown(
        options=_options, value=next(iter(_options)), label="Inspect a run (filtered by heatmap selection)",
    )
    inspect_pair
    return (inspect_pair,)


@app.cell
def _(inspect_pair, prepared, results):
    selected = results[inspect_pair.value]
    _data = prepared[inspect_pair.value]
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
    _endpoints = pd.DataFrame({"time_s": selected["time"][[0, -1]], "D": selected["distance"][[0, -1]]})
    _endpoints = _endpoints[_endpoints["D"] > 0]
    _secant = alt.Chart(_endpoints).mark_line(color="firebrick", point=True).encode(x="time_s:Q", y="D:Q")
    mo.vstack([
        mo.md(f"**{inspect_pair.value}: {_s['norm_status']}**, λ_eff = "
              f"**{_s['lambda_eff_s_inv']:.6g} s⁻¹**. "
              f"D(start) = {_s['D_start']:.6e}, D(end) = {_s['D_end']:.6e}."),
        mo.ui.altair_chart((_norm + _rules + _secant).properties(
            width=950, height=280, title="Paired norm; red endpoint secant defines λ_eff"
        ), chart_selection=False),
        mo.ui.table(selected["table"], selection=None, page_size=10),
        mo.download(pd.DataFrame({"time_s": selected["time"], "D": selected["distance"],
                                  "numerical_floor": selected["norm_floor"]}).to_csv(index=False).encode(),
                    filename=f"{inspect_pair.value}_distance.csv"),
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
def _(inspect_mode, selected):
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
    ).properties(width=950, height=250, title=f"Mode {inspect_mode.value}: {selected['table'].iloc[_index]['status']}")
    mo.ui.altair_chart(_chart, chart_selection=False)
    return


if __name__ == "__main__":
    app.run()
