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
        analyze_adaptive, demo_sweep, phase_coordinates, prepare_pair, theory_overlays,
    )
    from red_patterns.stability_sweep import ridge_pairs
    from red_patterns.sweep_catalog import selected_sweep_catalog, sweep_directory_picker


@app.cell(hide_code=True)
def _():
    mo.vstack([
        mo.md(r"""
        # How do ridge perturbations grow or decay?

        This notebook compares Taylor simulations of a displaced one-cell ridge
        with matching unperturbed ridge simulations under a **LINEAR_FULL Percoll
        profile**. It measures how spatial cosine components of the concentration
        difference change over time, then compares those trends across ν and μ.

        1. **Load a sweep** containing `runs.jsonl` and `results/`.
        2. **Choose the cosine modes and fit settings.** Each mode gets a straight
           line fitted to its early log-amplitude trace.
        3. **Click a heatmap cell**, then inspect the rates for that run and the
           fitted time trace of an individual mode below.

        A positive slope means the measured amplitude grows during its fitted
        interval; a negative slope means it decays. These are empirical trends,
        not Jacobian eigenvalues. The continuum curves are comparison guides;
        they are not the discrete ridge's verified instability boundary.
        """),
        mo.accordion({
            "What is being fitted? Symbols and physical meaning": mo.md(r"""
            At each saved time we subtract the **matching reference at that time**,
            then transform the difference along z using an orthonormal DCT-II:

            \[
            \delta\psi(z,t)=\psi_{\rm perturbed}(z,t)-\psi_{\rm reference}(z,t),
            \qquad A_m(t)=\mathrm{DCT}_m[\delta\psi(z,t)].
            \]

            We fit a line by ordinary, unweighted least squares:

            \[
            y_m(t)=\ln|A_m(t)|\approx b_m+\gamma_m(t-t_0).
            \]

            | Symbol | Meaning |
            | --- | --- |
            | ψ(z,t) | Concentration marginal obtained by summing the density-resolved population over density. |
            | δψ | Perturbed concentration minus reference concentration; the signal analyzed here. |
            | z, t | Spatial coordinate and saved simulation time. Plots use cm for z and seconds for t. |
            | m | DCT-II cosine mode number. Higher m means shorter spatial wavelength. |
            | N, Δz, L | Number of z cells, cell spacing, and domain length L = NΔz. |
            | Aₘ(t) | Signed cosine coefficient; its amplitude is the absolute value, \|Aₘ(t)\|. This is not power. |
            | t₀ | First saved time in the pair, which starts every candidate fit window. |
            | bₘ | Fitted log amplitude at t₀. Its exponential is the fitted initial amplitude. |
            | γₘ | Fitted amplitude growth rate in s⁻¹, i.e. slope of the natural log amplitude. |
            | ℓₘ | Wavelength 2L/m for m > 0, converted to cm in tables and maps. |
            | ν, μ | Saved Taylor interaction parameters (`NU`, `MU`) varied across the sweep. |
            | ε, seed | Initial displacement amplitude (`epsilon`) and random seed specifying its modal mixture. |

            For example, γ = 0.01 s⁻¹ corresponds to amplitude multiplication by
            exp(0.01 × 100) ≈ 2.72 over 100 s **if that trend continues**.
            A negative rate has decay time 1/|γ|. There is no factor of two:
            fitting the log of power would give twice the amplitude rate.

            Mode 0 is the spatially constant component. It can be inspected but
            is excluded from the maps. The selected analysis band can differ from
            the modes originally seeded in the simulation. Dividing δψ by a
            constant ε changes the intercept, not the slope.
            """),
            "Why a fitted slope is not a stability eigenvalue": mo.md(r"""
            The full discrete ridge state includes both density and position.
            A spatial cosine in its marginal need not be an eigenvector of the
            full Jacobian. One DCT coefficient can mix several decaying modes,
            producing curved traces or transient amplitude growth even when the
            equilibrium is linearly stable. A good line fit is therefore a
            useful finite-time numerical measurement, rather than proof of an
            unstable eigenmode. Compare ε, grid spacing, timestep and fit settings
            when assessing its reproducibility.
            """),
        }),
        mo.md("## 1. Load paired simulations\n\nRefresh saved results to rescan a running sweep. "
              "Synthetic validation sweep uses generated exponential traces to demonstrate the workflow."),
    ])
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
               mo.accordion({"Loaded pairs and reasons for missing results": mo.ui.table(_availability, selection=None)})])
    return pairs, prepared


@app.cell
def _(pairs, prepared):
    mo.stop(not prepared, mo.md("Select a sweep with complete Taylor LINEAR_FULL ridge pairs."))
    _max_mode = min(_data["raw"].shape[1] - 1 for _data in prepared.values())
    _ready = [_p for _p in pairs if _p["run_id"] in prepared]
    _low = min(max(1, min(_p["mode_min"] for _p in _ready)), _max_mode)
    _high = min(max(_p["mode_max"] for _p in _ready), _max_mode)
    fit_policy = mo.ui.dropdown(
        options={"Empirical slopes": "empirical", "Strict": "strict"},
        value="Empirical slopes", label="Fit policy",
    )
    min_samples_ui = mo.ui.number(start=5, value=8, step=1, label="Minimum saved frames per fit")
    modes = mo.ui.range_slider(
        start=0, stop=_max_mode, step=1, value=[_low, max(_low, _high)],
        label="Cosine modes to analyze (inclusive; controls both maps)",
        show_value=True, full_width=True, debounce=True,
    )
    mo.vstack([mo.md("## 2. Choose how to fit\n\nStart with Empirical slopes for approximate trends. "
                         "Strict adds quality cutoffs. Increasing the residual tolerance permits noisier "
                         "or more curved traces and can lengthen the fitted interval."),
               mo.hstack([fit_policy, min_samples_ui]), modes, mo.md(
        "Each mode gets its own early window starting at its pair's first saved frame. "
        "The mode range sets which rates can enter the maps; it does not filter the full concentration heatmap."
    )])
    return fit_policy, min_samples_ui, modes


@app.cell
def _(fit_policy):
    _empirical = fit_policy.value == "empirical"
    tolerance_ui = mo.ui.number(
        start=0.001, value=0.25 if _empirical else 0.05, step=0.005,
        label=("Log residual standard deviation tolerance" if _empirical else
               "Maximum absolute log residual tolerance"),
    )
    mo.vstack([
        tolerance_ui,
        mo.md("A straight line is fitted to log amplitude. Starting at the first saved frame, "
              "the window grows until its first failed error check; the preceding passing window is kept. "
              "The Empirical error metric is normalized for the number of frames and the two fitted parameters."),
    ])
    return (tolerance_ui,)


@app.cell(hide_code=True)
def _(fit_policy, min_samples_ui, tolerance_ui):
    _empirical = fit_policy.value == "empirical"
    _metric = "log residual standard deviation" if _empirical else "maximum absolute log residual"
    _quality = (
        "Empirical mode includes all finite fitted slopes in the maps, even when "
        "their quality status flags a poor fit or an uncertain slope. R², residuals "
        "and slope standard errors remain visible; no R² or significance cutoff is applied. "
        "The default residual standard deviation tolerance is 0.25 in natural-log units."
        if _empirical else
        "Strict mode includes only rates with log relative fit error ≤ 0.1 (R² ≥ 0.99) "
        "and slope magnitude above twice its fit standard error."
    )
    mo.accordion({"How the fitting window and quality checks work": mo.md(rf"""
    **Current settings:** start at the first saved frame with
    **{min_samples_ui.value} samples**, then extend one frame at a time. Stop at the
    first prefix whose {_metric} exceeds **{tolerance_ui.value:g}**,
    reaches the numerical floor, or changes sign. Retain the preceding prefix.
    If the first candidate fails, there is no automatic fit; no later window is searched.
    If every frame passes, the data have not identified the end of the fitted trend.
    This is the **longest passing prefix before the first failure**, not a global
    search for the longest passing window: a longer window might pass again after
    a failure, but the algorithm does not look for it.

    {_quality}

    Every extension refits the intercept and slope by minimizing the sum of
    squared log residuals. Saved times are used directly, so irregular saved-time
    spacing is allowed; frames receive equal weight. No smoothing is applied.

    A **residual** is observed ln|Aₘ| minus the fitted line. Empirical mode measures
    its size using the **residual standard deviation**:

    \[
    r_i=\ln|A_m(t_i)|-[b_m+\gamma_m(t_i-t_0)],\qquad
    s_{{\rm residual}}=\sqrt{{\frac{{\sum_{{i=1}}^n r_i^2}}{{n-2}}}}.
    \]

    Here n is the number of saved frames in the candidate window. We divide by
    **n − 2** because the line fits two parameters: intercept and slope. This is
    a normalization, not a growing sum of errors. For independent noise with
    constant variance around a true line, the expected squared metric estimates
    that variance regardless of n. Ordinary RMS divides by n instead and is
    slightly biased downward for short fitted windows; it remains available in
    the table for comparison.

    The metric can still increase **or** decrease as frames are added: we refit
    the line, and the extended trace may have different noise or curvature. It
    has no enforced monotonic trend. More densely sampling a region can also
    change its weight because regression gives every frame equal weight; this
    metric does not guarantee invariance to save cadence or correlated errors.

    A residual of +0.25
    means an amplitude about exp(0.25) = 1.28 times the fitted amplitude; −0.25
    means about 0.78 times it. The tolerance is not a bound on every individual
    point. Strict uses the largest absolute residual instead, which can be more
    sensitive to the number of frames and isolated outliers.

    **Minimum saved frames** sets the shortest candidate interval, not a duration
    in seconds. Increasing the residual tolerance can retain longer, noisier or
    more curved intervals. Switching fit policy resets tolerance to its default.
    A fitted interval is a numerical choice, not a certified physical linear regime.

    **Numerical floor:** a conservative roundoff estimate based on float64
    precision and the background field magnitude. Coefficients at or below it
    are not logged or replaced with a small constant. This does not model all
    solver errors. Sign changes stop a fit because a single signed exponential
    cannot cross zero, even though we plot its absolute amplitude.

    **Included** means the rate enters the maximum-rate and wavelength summaries.
    Empirical mode can include a rate whose status warns of uncertainty or poor
    fit; Strict excludes it. Mode 0 is excluded under both policies. A negative
    maximum describes only the included modes.
    """)})
    return


@app.cell
def _(fit_policy, min_samples_ui, modes, pairs, prepared, tolerance_ui):
    mo.stop(min_samples_ui.value is None or tolerance_ui.value is None,
            mo.md("Enter the minimum sample count and log-residual tolerance."))
    results = {
        _id: analyze_adaptive(_data, *map(int, modes.value),
                              min_samples=int(min_samples_ui.value), log_tolerance=tolerance_ui.value,
                              empirical=fit_policy.value == "empirical")
        for _id, _data in prepared.items()
    }
    _rows = []
    for _pair in pairs:
        _row = {k: v for k, v in _pair.items() if not k.endswith("_path")}
        # Do not confuse the seeded band with the selected analysis band.
        _row["seeded_mode_min"] = _row.pop("mode_min")
        _row["seeded_mode_max"] = _row.pop("mode_max")
        _row.update(dict(fastest_wavelength_cm=np.nan, max_growth_s_inv=np.nan,
                         wavelength_status=_pair["status"], growth_status=_pair["status"]))
        if _pair["run_id"] in results:
            _row.update(results[_pair["run_id"]]["summary"])
            _row["notes"] = "; ".join(prepared[_pair["run_id"]]["notes"])
        _rows.append(_row)
    summary = pd.DataFrame(_rows)
    summary["nu_scaled"] = summary["NU"] / 1e-30
    summary["mu_scaled"] = summary["MU"] / 1e-37
    modal_summary = pd.concat([
        _result["table"].assign(run_id=_id) for _id, _result in results.items()
    ], ignore_index=True).merge(
        summary[["run_id", "reference_id", "NU", "MU", "epsilon", "seed"]], on="run_id", validate="many_to_one",
    )
    return modal_summary, results, summary


@app.cell
def _():
    coordinates = mo.ui.dropdown(
        options={
            "Normalized onset: |ν| / |ν_c(μ)| versus −μ": "normalized",
            "Squared onset: ν² versus −μ": "squared",
            "Original parameters: ν versus μ": "raw",
        },
        value="Normalized onset: |ν| / |ν_c(μ)| versus −μ",
        label="Heatmap axes",
    )
    mo.vstack([
        mo.md("## 3. Compare runs in parameter space\n\nEach colored cell summarizes one run. "
              "The left map shows the wavelength with the largest positive fitted slope; "
              "the right shows the largest fitted slope, including negative values."),
        coordinates,
        mo.md("The default x axis divides |ν| by the continuum reference threshold |ν_c|. "
              "The reference onset is at x = 1. Changing axes changes the display, not the fits."),
        mo.accordion({"Axes, black reference curves and continuum equations": mo.md(r"""
        **Original** displays ν against μ. **Squared** displays ν² against −μ;
        the continuous-wavenumber onset is straight when the other parameters are
        fixed. **Normalized** displays |ν|/|ν_c(μ)| against −μ; values above 1
        predict growth in the continuum model. The transformed views use ν < 0
        and μ < 0; normalization also requires valid model metadata.

        The continuum reference uses

        \[
        \sigma(k)=-\alpha g-\beta\bar\psi\nu k^2+\beta\bar\psi\mu k^4,
        \qquad k_m=m\pi/L,\qquad g=30/L.
        \]

        Here σ is the theoretical amplitude rate, k is angular wavenumber,
        α and β are the saved buoyancy and interaction coefficients, and ψ̄ is
        the initial reference mean concentration. The positive g is the magnitude
        of the Percoll gradient. These predictions omit discrete stencil effects
        and do not give the one-cell ridge's Jacobian spectrum.

        **Dashed black:** onset after optimizing over all continuous k > 0:

        \[
        \nu_c=-\sqrt{C(-\mu)},\qquad C=\frac{4\alpha g}{\beta\bar\psi}.
        \]

        **Solid black:** the same continuum formula sampled only at the selected
        positive DCT modes, with onset

        \[
        \nu_{\rm band}=\max_m\left(\mu k_m^2-\frac{C}{4k_m^2}\right).
        \]

        These curves can overlap. Neither is a boundary inferred from the fitted
        slopes. Display units such as ν/10⁻³⁰ only rescale the axes.
        """)}),
    ])
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
        label="Show continuum wavelength guides [cm]",
    )
    mo.vstack([
        contour_levels,
        mo.md("Choose wavelengths to draw orange dotted guides on both maps. "
              "For example, the 1 cm curve joins parameter values where the continuum "
              "model predicts a fastest-growing wavelength of 1 cm. Removing a choice hides its curve; "
              "this control does not change any fits or heatmap colors."),
        mo.accordion({"How wavelength guides are calculated, and why one may be missing": mo.md(r"""
        A guide is a contour of the **predicted** optimum wavelength, not a fit to
        the measured wavelength map. Compare it with the colors of the left map
        to see whether measured and predicted wavelength selection agree.

        For ν, μ < 0 the continuum optimum is

        \[
        k_* = \sqrt{\nu/(2\mu)},\qquad \ell_*=2\pi/k_*.
        \]

        Selecting a wavelength ℓ draws the parameter curve
        ν = 2μ(2π/ℓ)², converting the selected cm to meters first. Only segments
        inside the displayed parameter range are drawn, with k* inside the
        selected mode band and at least one selected mode predicted to grow.
        A guide can therefore disappear when you change the mode range or axes,
        or when the displayed region has no eligible growing segment. Missing
        guides are listed below the maps. Black onset curves remain independent
        of these wavelength choices.
        """)}),
    ])
    return (contour_levels,)


@app.cell
def _(contour_levels, coordinates, modes, phase_data):
    overlay_data = theory_overlays(
        phase_data, coordinates.value, *map(int, modes.value), contour_levels.value,
    )
    return (overlay_data,)


@app.cell
def _(axis_titles, contour_levels, modal_summary, overlay_data, phase_data):
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
            alt.Tooltip("max_growth_fit_end_s:Q", title="Max-rate mode fit end [s]"),
            alt.Tooltip("max_growth_fit_samples:Q", title="Max-rate mode fit samples"),
            alt.Tooltip("resolved_modes:Q", title="Modes used in maps"),
            alt.Tooltip("unresolved_modes:Q", title="Unavailable or excluded modes"),
            alt.Tooltip("mode_min:Q"), alt.Tooltip("mode_max:Q"),
            alt.Tooltip("fastest_mode:Q"), alt.Tooltip(metric + ":Q", format=".6g"),
            alt.Tooltip(status + ":N"),
        ]
        _scale = alt.Scale(scheme="viridis")
        if metric == "max_growth_s_inv":
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
                strokeDash=alt.StrokeDash("curve:N", title="Continuum reference onset", scale=alt.Scale(
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
        phase_chart("fastest_wavelength_cm", "Wavelength of largest positive fitted slope",
                    "Wavelength [cm]", "wavelength_status"),
        phase_chart("max_growth_s_inv", "Maximum fitted log-amplitude slope",
                    "max γₘ [s⁻¹]", "growth_status"),
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
               mo.md("**Click a cell in either map to inspect its run below.** Double-click to clear. "
                     "On the rate map, red means a positive slope and blue a negative slope. "
                     "Grey means no value is available: on the wavelength map this also includes "
                     "runs with no positive fitted slope."),
               mo.accordion({
                   "Map details and wavelength-guide visibility": mo.md(
                       "The rate map takes the largest slope among modes used in the maps; "
                       "it does not take the largest amplitude. The wavelength map uses that mode's "
                       "wavelength only if its slope is positive. Each mode may have a different fit interval.\n\n"
                       "Cell edges lie halfway between samples and do not imply interpolation or "
                       "averaging. Runs at identical ν, μ coordinates overlap. Inspection uses the "
                       "first loaded matching run in sweep order when several share a cell. "
                       "Before any selection, it shows the first loaded run.\n\n"
                       f"**{_omitted} rows omitted** because these coordinates are undefined "
                       "(sign restrictions in transformed views or missing normalization metadata).\n\n"
                       + ("Requested wavelength guides without a visible growing segment: "
                          + ", ".join(_hidden_levels) + "." if _hidden_levels else
                          "All requested wavelength guides have a visible growing segment. "
                          "If no wavelengths are selected, no orange guides are requested.")
                   ),
                   "All runs: summary table": mo.ui.table(phase_data, selection=None, page_size=10),
               }),
               mo.download(phase_data.to_csv(index=False).encode(), filename="adaptive_stability.csv"),
               mo.download(modal_summary.to_csv(index=False).encode(), filename="adaptive_sweep_modal_rates.csv",
                           label="Download all per-mode rates and early windows")])
    return (phase_heatmap,)


@app.cell
def _(phase_heatmap, results, summary):
    _selected_rows = phase_heatmap.apply_selection(summary)
    _available = _selected_rows[_selected_rows["run_id"].isin(results)]
    mo.stop(_available.empty, mo.callout(
        "This cell has no loaded pair to inspect. See its availability status above, "
        "or click another heatmap cell.", kind="warn"))
    # No extra selector: a single heatmap click controls all run diagnostics.
    _run = _available.iloc[0]
    selected_run_id = _run["run_id"]
    mo.vstack([
        mo.md(f"## 4. Inspect the selected run\n\n**{selected_run_id}** · "
              f"ν = {_run['NU']:.6e}, μ = {_run['MU']:.6e} · "
              f"ε = {_run['epsilon']:.3e}, seed = {_run['seed']}"),
        mo.md(f"{len(_available)} loaded runs match the current selection; showing the first in sweep order. "
              "Click a heatmap cell to narrow the selection.") if len(_available) > 1 else
        mo.md("The plots below update directly when you click another heatmap cell."),
    ])
    return (selected_run_id,)


@app.cell
def _(results, selected_run_id):
    selected = results[selected_run_id]
    return (selected,)


@app.cell
def _(selected, selected_run_id):
    _s = selected["summary"]
    _measured = (
        f"mode {_s['fastest_mode']:g}, wavelength {_s['fastest_wavelength_cm']:.5g} cm"
        if np.isfinite(_s["fastest_mode"]) else "no positive included slope"
    )
    _predicted = (
        f"mode {_s['theory_fastest_mode']:g}, wavelength {_s['theory_fastest_wavelength_cm']:.5g} cm"
        if np.isfinite(_s["theory_fastest_mode"])
        else "no growing selected mode" if np.isfinite(_s["theory_max_rate_s_inv"])
        else "unavailable for this run"
    )
    _table = selected["table"]
    _accepted = _table["included_in_summary"]
    _flagged = _accepted & (_table["status"] != "ok")
    _fig, _ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    _ax.errorbar(_table.loc[_accepted, "mode"], _table.loc[_accepted, "growth_s_inv"],
                 yerr=_table.loc[_accepted, "rate_std_s_inv"], fmt="o", label="Fitted slope ± 1 standard error (used in maps)")
    _ax.scatter(_table.loc[_flagged, "mode"], _table.loc[_flagged, "growth_s_inv"],
                facecolors="none", edgecolors="tab:orange", s=90, label="Quality warning (still used in maps)")
    _ax.plot(_table.loc[~_accepted, "mode"], _table.loc[~_accepted, "growth_s_inv"],
             "x", label="Fits omitted from maps")
    _ax.plot(_table["mode"], _table["theory_rate_s_inv"], "--", label="Continuum reference at each cosine wavelength")
    _ax.axhline(0, color="grey", linewidth=0.8)
    _ax.set(xlabel="Cosine mode number m (higher = shorter wavelength)", ylabel="Fitted log-amplitude slope γₘ [s⁻¹]",
             title=f"{selected_run_id}: growth and decay by cosine mode")
    _ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    _ax.grid(alpha=0.25)
    _ax.legend()
    _output = mo.as_html(_fig)
    plt.close(_fig)
    mo.vstack([
        mo.md(f"**{_s['growth_status']}**. Maximum slope used in maps: "
              f"**{_s['max_growth_s_inv']:.6g} s⁻¹**; "
              f"**{_s['resolved_modes']} included**, **{_s['unresolved_modes']} unavailable/excluded** positive modes. "
              f"Fit policy: **{_s['fit_policy']}**."),
        mo.md(f"**Largest positive measured slope:** {_measured}."),
        _output,
        mo.md("Each dot is one cosine mode's fitted rate; vertical bars show **± 1 standard error "
              "of that slope**. Orange rings mark quality warnings retained by Empirical slopes. "
              "Crosses mark finite rates omitted from the maps; modes without a finite rate have no point."),
        mo.accordion({
            "How to read error bars, quality flags and the dashed curve": mo.md(r"""
            **Standard error (SE)** estimates how much the fitted slope could vary
            under the regression's residual-noise model. For γ = 0.010 s⁻¹ and
            SE = 0.002 s⁻¹, the plotted bar runs from 0.008 to 0.012 s⁻¹. This is
            a one-standard-error bar, not a 95% confidence interval or a bound on
            the simulation's numerical error. The calculation assumes independent,
            equal-variance log residuals; correlated solver errors or systematic
            curvature can make that interpretation unreliable.

            **Quality warnings** use the strict diagnostic thresholds even in
            Empirical mode. `poor log-linear fit` means relative log fit error
            exceeds 0.1. `uncertain log slope` means |γ| ≤ 2 SE. Empirical mode
            retains those estimates; Strict omits them from the maps. `ok` means
            the fit passes these checks, not that the mode is an eigenmode.

            **The dashed curve** evaluates the continuum dispersion σ(kₘ) using
            the run's saved parameters. It is not fitted to the dots. The discrete
            ridge need not have these continuum rates. Dots can also use different
            time intervals, so compare their fit endpoints in the table.

            Mode 0, if selected, is always omitted from the maps; its coefficient
            is still available in the time-trace inspector below.
            """),
            "Continuum comparison for this run": mo.md(
                f"Fastest predicted growing mode in the selected band: {_predicted}.\n\n"
                f"Largest predicted rate in that band: {_s['theory_max_rate_s_inv']:.6g} s⁻¹. "
                "This comparison samples the continuum formula at DCT wavenumbers; "
                "it is not the actual discrete ridge Jacobian spectrum."
            ),
            "Per-mode fit results": mo.ui.table(_table, selection=None, page_size=10),
            "Fit table and CSV column definitions": mo.md(r"""
            Each modal row describes one DCT mode in one run.

            | Column | Meaning |
            | --- | --- |
            | `mode`, `wavelength_cm` | Cosine index m and corresponding wavelength 2L/m in cm; undefined for m = 0. |
            | `growth_s_inv` | Estimated γ, in s⁻¹. Positive means amplitude growth over this fit interval. |
            | `rate_std_s_inv` | Slope standard error, in s⁻¹; the half-length of the plotted error bar. |
            | `r_squared` | R² = 1 − SSE/SST in log space: 1 is an exact line; near 0 means little improvement over a constant log amplitude. |
            | `log_relative_rmse` | sqrt(SSE/SST), i.e. residual size relative to variation in the observed log trace. Strict requires ≤ 0.1, equivalent to R² ≥ 0.99. |
            | `residual_std_log` | sqrt(SSE/(n − 2)) in natural-log units, with n = fit_samples; controls the Empirical window and accounts for fitting intercept and slope. |
            | `rms_log_residual` | sqrt(SSE/n) in natural-log units; retained as a diagnostic, not the Empirical stopping metric. |
            | `max_abs_log_residual` | Largest absolute residual; controls the Strict window. |
            | `fit_samples` | Number of saved frames used, not the number of solver steps. |
            | `fit_start_s`, `fit_end_s` | First and last saved times used in this mode's fit. |
            | `fit_policy` | `empirical` or `strict`; determines window metric and inclusion rules. |
            | `status` | Quality diagnosis for the retained fit, or the reason no fit exists. |
            | `window_reason` | Why extension stopped, or a message that all frames passed. Separate from quality of the retained fit. |
            | `included_in_summary` | Whether this mode's slope enters the map summaries. False for mode 0, nonfinite rates, and strict-policy failures. |
            | `theory_rate_s_inv` | Continuum σ(kₘ) at this mode's wavelength; not the measured slope. |

            SSE is the sum of squared log residuals. SST is the sum of squared
            deviations of observed log amplitudes from their mean. Undefined or
            unavailable numerical results are stored as NaN.

            The run summary uses `max_growth_s_inv` and `max_growth_mode` for the
            largest included slope and its mode. `max_growth_fit_end_s` and
            `max_growth_fit_samples` describe that mode's fitting window.
            `fastest_growth_s_inv`, `fastest_mode` and `fastest_wavelength_cm`
            report the same winner only when its rate is positive; otherwise
            they are undefined. `growth_status` and `wavelength_status` explain
            availability. Historical names `resolved_modes` and `unresolved_modes`
            count positive modes included in, or omitted from, these summaries;
            in Empirical mode, “resolved” does not imply statistical significance.

            `run_id` identifies the perturbed simulation; `reference_id` identifies
            its matched unperturbed run. `NU`, `MU`, `epsilon`, and `seed` record
            the simulation settings. `seeded_mode_min/max` describe the initial
            perturbation band, while `mode_min/max` describe the selected analysis
            band. `saved_start_s`, `saved_end_s`, `saved_samples` describe the full
            saved evolution, not an individual fitting window.
            `min_fit_samples`, `log_tolerance`, `window_residual_metric` and
            `fit_policy` record how that analysis was performed.

            Columns beginning `theory_` are continuum predictions for the selected
            band: maximum rate, growing-mode count, fastest growing mode and its
            wavelength, and onset ν. `nu_critical` is the unrestricted continuum
            threshold; `critical_ratio` is |ν|/|ν_c|. `boundary_coefficient` is C.
            `nu_scaled`, `mu_scaled`, `phase_x`, `phase_y` are display coordinates;
            `x_low/high`, `y_low/high` are rectangle edges, not uncertainty bounds.
            `notes` and pair `status` record loading or matching diagnostics.
            """),
        }),
        mo.download(_table.to_csv(index=False).encode(), filename=f"{selected_run_id}_adaptive_modal_rates.csv",
                    label="Download selected pair's rates and early windows"),
    ])
    return


@app.cell
def _(selected):
    _fastest = selected["summary"]["max_growth_mode"]
    inspect_mode = mo.ui.slider(
        steps=selected["modes"].tolist(),
        value=int(_fastest) if np.isfinite(_fastest) else int(selected["modes"][0]),
        label="Cosine mode m to inspect in the time trace", show_value=True,
    )
    mo.vstack([mo.md("### Inspect one mode over time\n\nChoose a mode to see its log amplitude "
                         "and line fit. The initial choice is the mode with the largest slope used in the maps, "
                         "or the first selected mode if no rate is available."), inspect_mode])
    return (inspect_mode,)


@app.cell
def _(inspect_mode, prepared, selected, selected_run_id):
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
    _mode_row = selected["table"].set_index("mode").loc[inspect_mode.value]
    if _mode_row["fit_samples"]:
        for _endpoint in (_mode_row["fit_start_s"], _mode_row["fit_end_s"]):
            _axis.axvline(_endpoint, color="firebrick", linestyle="--", linewidth=1.5)
    _axis.set(
        title=f"{selected_run_id}: perturbed ψ(z, t)",
        xlabel="Time [s]", ylabel="z [cm]",
        xlim=(_data["time"][0], _data["time"][-1]),
    )
    _output = mo.as_html(psi_figure)
    plt.close(psi_figure)
    mo.vstack([
        mo.md("**Concentration context.** This is the total perturbed concentration ψ(z,t), "
              "including its background, rather than the subtracted signal used for fitting."),
        _output,
        mo.md("Dashed red lines mark "
              f"the adaptive window for inspected mode **{inspect_mode.value}** when available; "
              "the field includes all spatial modes."),
    ])
    return (psi_figure,)


@app.cell
def _(inspect_mode, selected, selected_run_id):
    _index = int(inspect_mode.value - selected["modes"][0])
    _coefficients = selected["coefficients"][:, _index]
    _amplitude = np.abs(_coefficients)
    _log = np.full_like(_amplitude, np.nan)
    np.log(_amplitude, out=_log, where=_amplitude > selected["floor"])
    _fit = selected["fits"][_index]
    _row = selected["table"].iloc[_index]
    _count = int(_row["fit_samples"])
    _time = selected["time"]
    _fitted = np.full_like(_amplitude, np.nan)
    _residual = np.full_like(_amplitude, np.nan)
    _used = np.zeros(len(_time), dtype=bool)
    _fig, _axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    _axes[0].plot(_time, _log, label=f"ln|A_{inspect_mode.value}(t)|")
    if selected["floor"] > 0:
        _axes[0].axhline(np.log(selected["floor"]), ls=":", color="grey", label="roundoff floor")
    if _fit is not None:
        _fitted[:_count] = _fit.log_fitted
        _used[:_count] = _fit.used_samples
        _residual[_used] = _log[_used] - _fitted[_used]
        for _ax in _axes:
            _ax.axvspan(_time[0], _time[_count - 1], alpha=0.12, color="green")
        _axes[0].plot(_time, _fitted, "--", label=f"line fitted in green interval, γ={_fit.rate:.6g} s⁻¹")
        _axes[1].plot(_time, _residual, ".-")
    _axes[0].set(ylabel="ln|Aₘ|", title=f"{selected_run_id} · mode {inspect_mode.value}: {_row['status']}")
    _axes[0].legend()
    _axes[1].set(xlabel="Time [s]", ylabel="Fit log residual")
    for _ax in _axes:
        _ax.grid(alpha=0.25)
    _output = mo.as_html(_fig)
    plt.close(_fig)
    _description = (
        f"Adaptive fit: **{_row['fit_start_s']:.6g}–{_row['fit_end_s']:.6g} s**, "
        f"**{_count} samples**; γ = **{_row['growth_s_inv']:.6g} s⁻¹**, "
        f"slope standard error = {_row['rate_std_s_inv']:.3g} s⁻¹, R² = {_row['r_squared']:.6g}. "
        if _fit is not None else "No acceptable initial log-linear prefix. "
    )
    _download = pd.DataFrame(dict(
        time_s=_time, signed_coefficient=_coefficients, numerical_floor=selected["floor"],
        log_amplitude=_log, used_in_fit=_used, log_fit=_fitted, log_residual=_residual,
    ))
    mo.vstack([
        mo.md(_description + _row["window_reason"]), _output,
        mo.md("The green interval contains the frames used to fit the dashed line. "
              "The lower panel shows observed log amplitude minus that line; residuals near zero "
              "mean agreement, while a curved pattern suggests the slope changes with time. "
              "The dotted horizontal line is the roundoff floor; no log is shown below it."),
        mo.accordion({"Time-trace download columns": mo.md(
            "`time_s`: saved time; `signed_coefficient`: Aₘ before taking its absolute value; "
            "`numerical_floor`: amplitude threshold for logging; `log_amplitude`: ln|Aₘ| above that floor; "
            "`used_in_fit`: whether a frame enters the retained regression; `log_fit`: fitted line "
            "on its interval; `log_residual`: observed log amplitude minus the line on used frames. "
            "Values outside the fitting interval are not extrapolated in the fit or residual columns."
        )}),
        mo.download(_download.to_csv(index=False).encode(),
                    filename=f"{selected_run_id}_mode_{inspect_mode.value}_adaptive_fit.csv",
                    label="Download inspected mode and fit samples"),
    ])
    return


if __name__ == "__main__":
    app.run()
