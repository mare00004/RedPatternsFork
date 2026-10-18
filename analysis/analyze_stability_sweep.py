import marimo

__generated_with = "0.23.16"
app = marimo.App(width="wide")


with app.setup:
    import io
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.fft import idct

    from red_patterns.dct_dispersion import read_run
    from red_patterns.stability_sweep import analyze_pair, ridge_pairs
    from red_patterns.sweep_catalog import selected_sweep_catalog, sweep_directory_picker

    def load_pair(pair, synthetic=False):
        if not synthetic:
            return read_run(pair["perturbed_path"]), read_run(pair["equilibrium_path"])
        # Analytic full-gradient dispersion: one growing and one decaying case per mu.
        n, length = 256, 0.07
        time = np.linspace(0, 300, 151)
        modes = np.arange(n)
        k = np.pi * modes / length
        rate = -2e-5 * 30 / length - 7.4e23 * 0.03 * pair["NU"] * k**2 + 7.4e23 * 0.03 * pair["MU"] * k**4
        coeff = np.zeros((len(time), n))
        coeff[:, 1:101] = 2e-7 * (-1.)**modes[1:101] * np.exp(time[:, None] * rate[None, 1:101])
        common = dict(time=time, z=np.arange(n) * length / n, rho=np.arange(n),
                      model=dict(modelType="TAYL", gradientType="LINEAR_FULL", alpha=2e-5, beta=7.4e23),
                      variant=dict(NU=pair["NU"], MU=pair["MU"]), run=dict(DT=0.005), kernel=None)
        equilibrium = dict(common, psi=np.full((len(time), n), 0.03), path="Synthetic reference")
        perturbed = dict(common, psi=equilibrium["psi"] + idct(coeff, type=2, norm="ortho", axis=1),
                         path="Synthetic perturbation")
        return perturbed, equilibrium

    def show_figure(fig):
        output = mo.as_html(fig)
        plt.close(fig)
        return output

    def edges(values):
        return np.r_[values[0] - (values[1] - values[0]) / 2,
                     (values[:-1] + values[1:]) / 2,
                     values[-1] + (values[-1] - values[-2]) / 2]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Taylor stability sweep

    Choose the sweep folder containing **runs.jsonl** and **results/**, then select
    a ν, μ pair. References are matched by parameters. Fit the **early linear regime**;
    compare nearby endpoints before interpreting onset. The reference is its first
    saved ψ profile, held fixed. Mode wavelengths are λ = 2NΔz/m.
    """)
    return


@app.cell
def _():
    directory = sweep_directory_picker(mo, initial_path=Path.cwd(), label="Sweep directory")
    refresh = mo.ui.run_button(label="Refresh results")
    demo = mo.ui.checkbox(value=mo.app_meta().mode == "script", label="Synthetic validation sweep")
    mo.vstack([directory, mo.hstack([refresh, demo])])
    return demo, directory, refresh


@app.cell
def _(demo, directory, refresh):
    refresh.value
    if demo.value:
        pairs = []
        for _mu in [-4.05e-37, -4.95e-37]:
            for _factor in [0.95, 1.05]:
                _nu = -_factor * np.sqrt(-4 * 2e-5 * (30 / 0.07) * _mu / (7.4e23 * 0.03))
                pairs.append(dict(run_id=f"demo-{len(pairs)+1}", reference_id="demo-reference",
                                  NU=_nu, MU=_mu, psi_avg=0.03, status="ready",
                                  mode_min=1, mode_max=100, epsilon=1e-8, seed=0))
        _message = mo.md("**Synthetic data:** analytic modes on either side of the continuum boundary.")
    else:
        _catalog, _message = selected_sweep_catalog(mo, directory)
        pairs = ridge_pairs(_catalog) if _catalog is not None else []
    mo.vstack([_message, mo.ui.table(pd.DataFrame(pairs).drop(
        columns=["perturbed_path", "equilibrium_path"], errors="ignore"), selection=None,
        label="Pair availability", page_size=8)])
    return (pairs,)


@app.cell
def _(pairs):
    mo.stop(not pairs, mo.md("Select a sweep with Taylor LINEAR_FULL ridge pairs."))
    pair_picker = mo.ui.dropdown(
        options={f"ν={p['NU']:.6e}, μ={p['MU']:.6e} · {p['run_id']} · {p['status']}": i
                 for i, p in enumerate(pairs)}, value=None, label="Parameter pair",
    )
    batch_settings = mo.ui.dictionary({
        "end": mo.ui.number(start=0, value=100.0, label="Sweep fit endpoint [s]"),
        "low": mo.ui.number(start=1, value=1, step=1, label="First mode"),
        "high": mo.ui.number(start=1, value=100, step=1, label="Last mode"),
    }).form(submit_button_label="Analyze all available pairs")
    mo.vstack([pair_picker, mo.accordion({"Whole-sweep growth map": mo.vstack([
        mo.md("Uses one common window from the first saved frame through this endpoint. "
              "Inspect individual pairs to check that this window remains linear."), batch_settings,
    ])})])
    return batch_settings, pair_picker


@app.cell
def _(batch_settings, demo, pairs):
    _settings = batch_settings.value
    if _settings is None and mo.app_meta().mode == "script":
        _settings = dict(end=100.0, low=1, high=100)
    mo.stop(_settings is None, mo.md("Submit the whole-sweep form to compute a growth map."))
    _rows = []
    for _pair in mo.status.progress_bar(pairs, title="Analyzing pairs"):
        _row = {k: v for k, v in _pair.items() if not k.endswith("_path")}
        if _pair["status"] == "ready":
            try:
                _p, _e = load_pair(_pair, demo.value)
                _result = analyze_pair(_p, _e, float(_settings["end"]), int(_settings["low"]), int(_settings["high"]))
                _row.update(_result["summary"])
                _row["notes"] = "; ".join(_result["notes"])
                del _p, _e, _result
            except (OSError, KeyError, ValueError) as _exc:
                _row["status"] = f"Cannot analyze: {_exc}"
        _rows.append(_row)
    sweep_summary = pd.DataFrame(_rows)
    return (sweep_summary,)


@app.cell
def _(sweep_summary):
    _fig, _ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    _good = sweep_summary["status"].isin(["resolved growth", "selected modes decay"])
    if _good.any():
        _values = sweep_summary.loc[_good, "max_growth_s_inv"]
        _limit = max(float(np.abs(_values).max()), 1e-12)
        _image = _ax.scatter(sweep_summary.loc[_good, "MU"] / 1e-37,
                            sweep_summary.loc[_good, "NU"] / 1e-30,
                            c=_values, cmap="RdBu_r", vmin=-_limit, vmax=_limit, s=70)
        _fig.colorbar(_image, ax=_ax, label="Max resolved growth rate [s⁻¹]")
    _ax.scatter(sweep_summary.loc[~_good, "MU"] / 1e-37,
                sweep_summary.loc[~_good, "NU"] / 1e-30,
                marker="x", color="grey", label="Unresolved / unavailable / check reference")
    if "nu_critical" in sweep_summary:
        # Separate curves for different saved density/geometry/model settings.
        for _, _group in sweep_summary.dropna(subset=["nu_critical"]).groupby(["alpha", "beta", "g", "psi_avg"]):
            _curve = _group[["MU", "nu_critical"]].drop_duplicates().sort_values("MU")
            _ax.plot(_curve["MU"] / 1e-37, _curve["nu_critical"] / 1e-30, "k--", alpha=0.7)
    _ax.set(xlabel="μ / 10⁻³⁷", ylabel="ν / 10⁻³⁰", title="Measured growth; dashed continuum boundary")
    _ax.legend(fontsize=8)
    mo.vstack([show_figure(_fig), mo.md(
        "Grey points are not classified as stable. Decay refers only to the selected modes. "
        "CSV includes the actual last saved time used for each fit."),
        mo.ui.table(sweep_summary, selection=None, page_size=10),
        mo.download(sweep_summary.to_csv(index=False).encode(), filename="stability_sweep.csv")])
    return


@app.cell
def _(demo, pair_picker, pairs):
    _index = pair_picker.value
    if _index is None and mo.app_meta().mode == "script":
        _index = 1
    mo.stop(_index is None, mo.md("Choose a parameter pair for detailed plots."))
    selected_pair = pairs[_index]
    mo.stop(selected_pair["status"] != "ready", mo.callout(selected_pair["status"], kind="warn"))
    _error = None
    try:
        perturbed, equilibrium = load_pair(selected_pair, demo.value)
    except (OSError, KeyError, ValueError) as _exc:
        _error = str(_exc)
    mo.stop(_error is not None, mo.callout(f"Cannot load pair: {_error}", kind="warn"))
    mo.stop(len(perturbed["time"]) < 5, mo.md("Need at least five saved perturbed frames."))
    endpoint = mo.ui.slider(steps=perturbed["time"][4:].tolist(),
                            value=float(perturbed["time"][max(4, np.searchsorted(perturbed["time"], 100, side="right") - 1)]),
                            label="Pair fit endpoint [s]", show_value=True, full_width=True, debounce=True)
    modes = mo.ui.range_slider(start=1, stop=len(perturbed["z"]) - 1,
                               value=[selected_pair["mode_min"], min(selected_pair["mode_max"], len(perturbed["z"]) - 1)],
                               label="Inclusive DCT modes", show_value=True, debounce=True)
    mo.vstack([mo.md(f"**Reference:** `{equilibrium['path']}`  \n**Perturbed:** `{perturbed['path']}`"), endpoint, modes])
    return endpoint, equilibrium, modes, perturbed, selected_pair


@app.cell
def _(endpoint, equilibrium, modes, perturbed):
    _error = None
    try:
        analysis = analyze_pair(perturbed, equilibrium, endpoint.value, *modes.value)
    except (KeyError, ValueError) as _exc:
        _error = str(_exc)
    mo.stop(_error is not None, mo.callout(f"Cannot fit pair: {_error}", kind="warn"))
    _s = analysis["summary"]
    mo.vstack([mo.md(
        f"**{_s['status']}** · resolved modes: {_s['resolved_modes']}/{_s['selected_modes']} · "
        f"reference drift: {_s['equilibrium_drift']:.3e}  \n"
        f"ν = {_s['NU']:.6e}, μ = {_s['MU']:.6e}; continuum ν boundary = {_s['nu_critical']:.6e}.  \n"
        f"Fastest growing mode: {_s['fastest_mode']:g}; wavelength: {_s['fastest_wavelength_cm']:.5g} cm."),
        *[mo.callout(note, kind="warn") for note in analysis["notes"]]])
    return (analysis,)


@app.cell
def _(analysis, endpoint, equilibrium, perturbed):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    _vmin = 100 * min(np.min(equilibrium["psi"]), np.min(perturbed["psi"]))
    _vmax = max(100 * max(np.max(equilibrium["psi"]), np.max(perturbed["psi"])), _vmin + 1e-9)
    for _ax, _run, _title in zip(_axes[:2], [equilibrium, perturbed], ["Unperturbed ψ [%]", "Perturbed ψ [%]"]):
        _take = np.unique(np.linspace(0, len(_run["time"]) - 1, min(1000, len(_run["time"])), dtype=int))
        _times = _run["time"][_take]
        _te = edges(_times) if len(_times) > 1 else np.array([_times[0] - 0.5, _times[0] + 0.5])
        _im = _ax.pcolormesh(_te, 100 * edges(_run["z"]), 100 * _run["psi"][_take].T,
                             shading="flat", cmap="viridis", vmin=_vmin, vmax=_vmax, rasterized=True)
        _ax.axvline(endpoint.value, color="red", linestyle="--")
        _ax.set(title=_title, xlabel="Time [s]", ylabel="z [cm]")
    _fig.colorbar(_im, ax=list(_axes[:2]))
    _take = np.unique(np.linspace(0, len(analysis["time"]) - 1, min(1000, len(analysis["time"])), dtype=int))
    _delta = perturbed["psi"][_take] - equilibrium["psi"][0]
    _lim = max(float(np.abs(_delta).max()), 1e-16)
    _im = _axes[2].pcolormesh(edges(analysis["time"][_take]), 100 * edges(perturbed["z"]), _delta.T,
                              shading="flat", cmap="RdBu_r", vmin=-_lim, vmax=_lim, rasterized=True)
    _axes[2].set(title="δψ in fit window [fraction]", xlabel="Time [s]", ylabel="z [cm]")
    _fig.colorbar(_im, ax=_axes[2])
    show_figure(_fig)
    return


@app.cell
def _(analysis):
    _table = analysis["table"]
    _ok = _table["status"] == "ok"
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for _ax, _x, _scale, _label in zip(_axes, ["mode", "wavelength_m"], [1, 100], ["DCT mode", "Wavelength [cm]"]):
        _ax.plot(_scale * _table[_x], _table["theory_rate_s^-1"], "k--", label="Taylor continuum")
        _ax.errorbar(_scale * _table.loc[_ok, _x], _table.loc[_ok, "growth_rate_s^-1"],
                     yerr=_table.loc[_ok, "rate_std_s^-1"], fmt=".", label="Resolved ± fit std. error")
        _ax.scatter(_scale * _table.loc[~_ok, _x], _table.loc[~_ok, "growth_rate_s^-1"],
                    marker="x", color="orange", label="Poor fit")
        _ax.axhline(0, color="grey", linewidth=1)
        _ax.set(xlabel=_label, ylabel="Growth rate [s⁻¹]", yscale="symlog")
        _ax.set_yscale("symlog", linthresh=1e-4)
        _ax.legend(fontsize=8)
    mo.vstack([show_figure(_fig), mo.md(
        r"Theory: $\sigma=-\alpha g-\beta\bar\psi\nu k^2+\beta\bar\psi\mu k^4$, "
        r"$g=30/(N\Delta z)$. Saved parameters and the initial reference mean define the curve. "
        "Axes use a symmetric log scale to show rates close to zero. Finite-grid rates can differ."),
        mo.accordion({"Mode rates and fit diagnostics": mo.ui.table(_table, selection=None),
                      "Resolved wavelength cutoffs": mo.ui.table(analysis["crossings"], selection=None)})])
    return


@app.cell
def _(analysis):
    _fastest = analysis["fastest"]
    inspect_mode = mo.ui.slider(steps=analysis["table"]["mode"].tolist(),
                                value=int(_fastest["mode"]) if _fastest else int(analysis["table"]["mode"].iloc[0]),
                                label="Inspect mode", show_value=True, debounce=True)
    inspect_mode
    return (inspect_mode,)


@app.cell
def _(analysis, inspect_mode, selected_pair):
    _m = inspect_mode.value
    _fit = analysis["fits"][_m - int(analysis["table"]["mode"].iloc[0])]
    _t = analysis["time"]
    _amp = np.abs(analysis["coefficients"][:, _m])
    _log = np.full_like(_amp, np.nan)
    np.log(_amp, out=_log, where=_amp > 0)
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 3.5), constrained_layout=True)
    _axes[0].plot(_t, _log, label="ln|coefficient|")
    _axes[0].plot(_t[_fit.used_samples], _fit.log_fitted[_fit.used_samples], "--", label="Fit")
    _axes[0].axhline(np.log(analysis["floor"]), color="grey", linestyle=":", label="Precision floor")
    _axes[0].set(title=f"Mode {_m}: {_fit.status}", xlabel="Time [s]", ylabel="Log amplitude")
    _axes[0].legend()
    _axes[1].plot(_t[_fit.used_samples], (_log - _fit.log_fitted)[_fit.used_samples])
    _axes[1].set(title=f"Log residuals; relative error {_fit.relative_rmse:.3g}", xlabel="Time [s]")
    _buffer = io.BytesIO()
    _metadata = dict(pair=selected_pair, **analysis["summary"], notes=analysis["notes"],
                     fit_start_s=float(_t[0]), mode_min=int(analysis["table"]["mode"].min()),
                     mode_max=int(analysis["table"]["mode"].max()),
                     reference="fixed initial equilibrium psi", fit="ln|c| = b + gamma*(t-t0)")
    np.savez_compressed(_buffer, time=_t, modes=analysis["table"]["mode"].to_numpy(),
                        coefficients=analysis["coefficients"][:, analysis["table"]["mode"].to_numpy()],
                        fitted_coefficients=np.column_stack([f.fitted for f in analysis["fits"]]),
                        used_samples=np.column_stack([f.used_samples for f in analysis["fits"]]),
                        metadata_json=json.dumps(_metadata))
    mo.vstack([show_figure(_fig), mo.md(
        "Fits use original saved times and at least five samples above the precision floor. "
        "Resolved fits require a consistent sign, log relative error ≤ 0.1, and |rate| > 2 standard errors. "
        "Unresolved modes do not establish stability. Inspect residuals and vary the endpoint."),
        mo.hstack([mo.download(analysis["table"].to_csv(index=False).encode(), filename=f"{selected_pair['run_id']}_rates.csv"),
                   mo.download(_buffer.getvalue(), filename=f"{selected_pair['run_id']}_fits.npz")])])
    return


if __name__ == "__main__":
    app.run()
