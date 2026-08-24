# /// script
# dependencies = [
#     "h5py==3.16.0",
#     "marimo>=0.23.6",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "pydantic==2.13.4",
#     "scipy==1.17.1",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="wide")


with app.setup:
    import json
    import sys
    import warnings
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.fft import dct
    from scipy.optimize import OptimizeWarning, curve_fit

    NOTEBOOK_FILE = (
        Path(__file__).resolve()
        if "__file__" in globals()
        else (Path.cwd() / "analysis" / "analyze_single_mode_dct_modes.py").resolve()
    )
    ANALYSIS_DIR = NOTEBOOK_FILE.parent
    REPO_ROOT = ANALYSIS_DIR.parent
    if str(ANALYSIS_DIR) not in sys.path:
        sys.path.insert(0, str(ANALYSIS_DIR))

    from red_patterns import RunData, get_rbc_cmap, load_runs_jsonl, plot_psi
    from red_patterns.models import TaylorRun
    from red_patterns.types import PhiType


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Single-mode DCT-II growth analysis

    Select a Taylor sweep containing either a `smooth_homogeneous` reference
    with `single_mode_smooth_homogeneous` runs, or a `linear_full_ridge`
    reference with `single_mode_linear_full_ridge` runs. The notebook subtracts
    the matching reference from the selected injected mode $n$, computes signed
    spatial DCT-II coefficients $A_{m,n}(t)$, and fits every coefficient to
    $C + A_0 e^{\gamma t}$.
    """)
    return


@app.cell
def _(mo):
    analysis_mode_selector = mo.ui.dropdown(
        options={
            "One-sided positive amplitude": "one_sided",
            "Paired positive/negative amplitudes": "paired",
        },
        value=None,
        label="Amplitude analysis mode",
    )
    analysis_mode_selector
    return (analysis_mode_selector,)


@app.cell
def _(analysis_mode_selector, mo):
    mo.stop(
        analysis_mode_selector.value is None,
        mo.md("Choose an amplitude analysis mode to continue."),
    )
    analysis_mode = str(analysis_mode_selector.value)
    return (analysis_mode,)


@app.cell
def _(Path):
    ui_sweep_dir = mo.ui.file_browser(
        initial_path=Path.cwd(),
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="directory",
        label="Choose Taylor sweep directory",
    )
    ui_sweep_dir
    return (ui_sweep_dir,)


@app.cell
def _(Path, TaylorRun, load_runs_jsonl, np, pd):
    def scan_sweep(sweep_dir: Path) -> pd.DataFrame:
        """Return compatible base and single-mode Taylor runs from a sweep."""
        compatible_phi_types = {
            PhiType.SMOOTH_HOMOGENEOUS.value,
            PhiType.SINGLE_MODE_SMOOTH_HOMOGENEOUS.value,
            PhiType.LINEAR_FULL_RIDGE.value,
            PhiType.SINGLE_MODE_LINEAR_FULL_RIDGE.value,
        }
        rows: list[dict[str, object]] = []
        for run in load_runs_jsonl(sweep_dir / "runs.jsonl"):
            if not isinstance(run, TaylorRun):
                continue

            phi_params = run.phi.params.model_dump(mode="json")
            phi_type = str(phi_params.pop("phi_type"))
            if phi_type not in compatible_phi_types:
                continue

            amplitude = phi_params.pop("amplitude", None)
            mode_number = phi_params.pop("mode_number", None)
            shared_phi = json.dumps(phi_params, sort_keys=True, separators=(",", ":"))
            result_path = sweep_dir / "results" / run.run_id / "run.h5"
            rows.append(
                {
                    "run_id": run.run_id,
                    "NU": float(run.NU),
                    "MU": float(run.MU),
                    "phi_type": phi_type,
                    "mode_number": None if mode_number is None else int(mode_number),
                    "amplitude": None if amplitude is None else float(amplitude),
                    "shared_phi": shared_phi,
                    "N": int(run.N),
                    "T": float(run.T),
                    "DT": float(run.DT),
                    "storeTime": float(run.storeTime),
                    "gradient": run.gradient.value,
                    "run_h5": result_path,
                    "h5_exists": result_path.is_file(),
                }
            )
        return pd.DataFrame(rows)

    def validate_sweeps(
        sweep_df: pd.DataFrame, analysis_mode: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        candidate_columns = [
            "NU", "MU", "N", "T", "DT", "storeTime", "gradient", "shared_phi",
            "base_phi_type", "mode_phi_type", "base_id", "mode_numbers", "mode_ids",
            "negative_mode_ids", "amplitudes",
        ]
        diagnostic_columns = ["NU", "MU", "status", "details"]
        if sweep_df.empty:
            return pd.DataFrame(columns=candidate_columns), pd.DataFrame(columns=diagnostic_columns)

        setup_columns = ["NU", "MU", "N", "T", "DT", "storeTime", "gradient", "shared_phi"]
        phi_families = (
            (
                PhiType.SMOOTH_HOMOGENEOUS.value,
                PhiType.SINGLE_MODE_SMOOTH_HOMOGENEOUS.value,
            ),
            (
                PhiType.LINEAR_FULL_RIDGE.value,
                PhiType.SINGLE_MODE_LINEAR_FULL_RIDGE.value,
            ),
        )
        candidates: list[dict[str, object]] = []
        diagnostics: list[dict[str, object]] = []
        for setup, group in sweep_df.groupby(setup_columns, sort=True, dropna=False):
            setup_values = dict(zip(setup_columns, setup, strict=True))
            prefix = {"NU": float(setup_values["NU"]), "MU": float(setup_values["MU"])}
            for base_phi_type, mode_phi_type in phi_families:
                family_group = group[group["phi_type"].isin((base_phi_type, mode_phi_type))]
                base = family_group[family_group["phi_type"] == base_phi_type]
                modes = family_group[family_group["phi_type"] == mode_phi_type]
                family_prefix = f"{base_phi_type} / {mode_phi_type}"

                if modes.empty:
                    continue
                missing = family_group.loc[~family_group["h5_exists"], "run_id"].tolist()
                if missing:
                    diagnostics.append({**prefix, "status": "incomplete", "details": f"{family_prefix}: missing run.h5 for " + ", ".join(missing)})
                    continue

                if modes["mode_number"].isna().any() or modes["amplitude"].isna().any():
                    diagnostics.append({**prefix, "status": "invalid", "details": f"{family_prefix}: every single-mode run needs a mode number and amplitude."})
                    continue

                if analysis_mode == "one_sided":
                    if len(base) != 1:
                        diagnostics.append({**prefix, "status": "invalid", "details": f"{family_prefix}: expected one matching reference; found {len(base)}."})
                        continue
                    if (modes["amplitude"] <= 0.0).any():
                        diagnostics.append({**prefix, "status": "invalid", "details": "One-sided analysis requires exactly positive single-mode amplitudes."})
                        continue
                    if modes["mode_number"].duplicated().any():
                        diagnostics.append({**prefix, "status": "invalid", "details": "Mode numbers must be unique."})
                        continue
                    positive_modes = modes.sort_values("mode_number", kind="stable")
                    base_id: str | None = str(base.iloc[0]["run_id"])
                    negative_ids = tuple(None for _ in range(len(positive_modes)))
                elif analysis_mode == "paired":
                    if (modes["amplitude"] == 0.0).any():
                        diagnostics.append({**prefix, "status": "invalid", "details": "Paired analysis does not allow zero amplitudes."})
                        continue
                    positive_rows: list[pd.Series] = []
                    negative_rows: list[pd.Series] = []
                    pairing_error: str | None = None
                    for mode_number, mode_group in modes.groupby("mode_number", sort=True):
                        positive = mode_group[mode_group["amplitude"] > 0.0]
                        negative = mode_group[mode_group["amplitude"] < 0.0]
                        if len(positive) != 1 or len(negative) != 1:
                            pairing_error = (
                                f"Mode n={int(mode_number)} needs exactly one positive and one negative run; "
                                f"found {len(positive)} positive and {len(negative)} negative."
                            )
                            break
                        positive_amplitude = float(positive.iloc[0]["amplitude"])
                        negative_amplitude = float(negative.iloc[0]["amplitude"])
                        if not np.isclose(positive_amplitude, -negative_amplitude, rtol=1e-12, atol=0.0):
                            pairing_error = (
                                f"Mode n={int(mode_number)} amplitudes must have equal magnitude; "
                                f"found {positive_amplitude:.6g} and {negative_amplitude:.6g}."
                            )
                            break
                        positive_rows.append(positive.iloc[0])
                        negative_rows.append(negative.iloc[0])
                    if pairing_error is not None:
                        diagnostics.append({**prefix, "status": "invalid", "details": f"{family_prefix}: {pairing_error}"})
                        continue
                    positive_modes = pd.DataFrame(positive_rows).sort_values("mode_number", kind="stable")
                    negative_modes = pd.DataFrame(negative_rows).sort_values("mode_number", kind="stable")
                    base_id = None
                    negative_ids = tuple(str(value) for value in negative_modes["run_id"])
                else:
                    raise ValueError(f"Unknown analysis mode: {analysis_mode}.")

                candidates.append(
                    {
                        **setup_values,
                        "base_phi_type": base_phi_type,
                        "mode_phi_type": mode_phi_type,
                        "base_id": base_id,
                        "mode_numbers": tuple(int(value) for value in positive_modes["mode_number"]),
                        "mode_ids": tuple(str(value) for value in positive_modes["run_id"]),
                        "negative_mode_ids": negative_ids,
                        "amplitudes": tuple(float(value) for value in positive_modes["amplitude"]),
                    }
                )
                details = (
                    f"{family_prefix}: reference plus {len(positive_modes)} positive single-mode runs."
                    if analysis_mode == "one_sided"
                    else f"{family_prefix}: {len(positive_modes)} matched positive/negative single-mode pairs."
                )
                diagnostics.append({**prefix, "status": "ready", "details": details})

        return pd.DataFrame(candidates, columns=candidate_columns), pd.DataFrame(diagnostics, columns=diagnostic_columns)

    return scan_sweep, validate_sweeps


@app.cell
def _(Path, analysis_mode, mo, pd, scan_sweep, ui_sweep_dir, validate_sweeps):
    selected_path = ui_sweep_dir.path(0) if ui_sweep_dir.value else None
    sweep_dir = Path(selected_path) if selected_path else REPO_ROOT / "data"
    if not (sweep_dir / "runs.jsonl").is_file():
        sweep_df = pd.DataFrame()
        candidate_df = pd.DataFrame()
        diagnostics_df = pd.DataFrame()
        scan_status = mo.callout(f"`{sweep_dir}` does not contain `runs.jsonl`. Choose a sweep directory.", kind="warn")
    else:
        try:
            sweep_df = scan_sweep(sweep_dir)
            candidate_df, diagnostics_df = validate_sweeps(sweep_df, analysis_mode)
            families = {
                f"{row.base_phi_type} / {row.mode_phi_type}"
                for _, row in candidate_df.iterrows()
            }
            family_text = ", ".join(sorted(families)) if families else "no compatible phi family"
            scan_status = mo.md(
                f"Found {len(sweep_df)} relevant Taylor runs and {len(candidate_df)} compatible "
                f"setups for {analysis_mode} analysis in {sweep_dir}. Detected: {family_text}."
            )
        except ValueError as exc:
            sweep_df = pd.DataFrame()
            candidate_df = pd.DataFrame()
            diagnostics_df = pd.DataFrame()
            scan_status = mo.callout(f"Could not read `{sweep_dir / 'runs.jsonl'}`: {exc}", kind="warn")
    scan_status
    return candidate_df, diagnostics_df, sweep_dir


@app.cell
def _(diagnostics_df, mo):
    mo.stop(diagnostics_df.empty, mo.md("No compatible base/single-mode Taylor candidates found."))
    mo.ui.table(data=diagnostics_df, selection=None, pagination=True)
    return


@app.cell
def _(candidate_df, mo):
    mo.stop(candidate_df.empty, mo.md("No complete compatible single-mode setups are available yet."))
    options = {
        (
            f"{row.base_phi_type} / {row.mode_phi_type}; "
            f"ν={row.NU:.6e}, μ={row.MU:.6e} (N={row.N}, T={row['T']:g})"
        ): index
        for index, row in candidate_df.iterrows()
    }
    pair_selector = mo.ui.dropdown(options=options, value=next(iter(options)), label=r"Select $(\nu, \mu)$ setup")
    pair_selector
    return (pair_selector,)


@app.cell
def _(candidate_df, pair_selector):
    selected_setup = candidate_df.loc[int(pair_selector.value)]
    return (selected_setup,)


@app.cell
def _(mo, selected_setup):
    mode_selector = mo.ui.slider(
        start=min(selected_setup["mode_numbers"]),
        stop=max(selected_setup["mode_numbers"]),
        step=1,
        value=min(selected_setup["mode_numbers"]),
        label="Injected single-cosine mode n",
        full_width=True,
        show_value=True,
    )
    mode_selector
    return (mode_selector,)


@app.cell
def _(RunData, analysis_mode, mode_selector, np, selected_setup, sweep_dir):
    selected_n = int(mode_selector.value)
    mode_index = selected_setup["mode_numbers"].index(selected_n)
    selected_run_id = selected_setup["mode_ids"][mode_index]
    selected_amplitude = selected_setup["amplitudes"][mode_index]
    mode_run = RunData.from_h5(sweep_dir / "results" / selected_run_id / "run.h5", load_fields=False)
    time = np.asarray(mode_run.time, dtype=np.float64)
    z = np.asarray(mode_run.z, dtype=np.float64)
    mode_psi = np.asarray(mode_run.load_psi(), dtype=np.float64)
    base_run = None
    negative_run = None
    selected_negative_run_id = None
    if analysis_mode == "one_sided":
        base_id = str(selected_setup["base_id"])
        base_run = RunData.from_h5(sweep_dir / "results" / base_id / "run.h5", load_fields=False)
        base_time = np.asarray(base_run.time, dtype=np.float64)
        base_z = np.asarray(base_run.z, dtype=np.float64)
        base_psi = np.asarray(base_run.load_psi(), dtype=np.float64)
        if not np.array_equal(base_time, time) or not np.array_equal(base_z, z):
            raise ValueError(f"{selected_run_id} has a different saved-time or z grid than reference {base_id}.")
        if mode_psi.shape != base_psi.shape:
            raise ValueError(f"{selected_run_id} psi shape {mode_psi.shape} differs from reference shape {base_psi.shape}.")
        delta_psi = mode_psi - base_psi
    elif analysis_mode == "paired":
        selected_negative_run_id = str(selected_setup["negative_mode_ids"][mode_index])
        negative_run = RunData.from_h5(
            sweep_dir / "results" / selected_negative_run_id / "run.h5",
            load_fields=False,
        )
        negative_time = np.asarray(negative_run.time, dtype=np.float64)
        negative_z = np.asarray(negative_run.z, dtype=np.float64)
        negative_psi = np.asarray(negative_run.load_psi(), dtype=np.float64)
        if not np.array_equal(negative_time, time) or not np.array_equal(negative_z, z):
            raise ValueError(f"{selected_negative_run_id} has a different saved-time or z grid than positive run {selected_run_id}.")
        if mode_psi.shape != negative_psi.shape:
            raise ValueError(f"{selected_negative_run_id} psi shape {negative_psi.shape} differs from positive shape {mode_psi.shape}.")
        delta_psi = (mode_psi - negative_psi) / (2.0 * selected_amplitude)
    else:
        raise ValueError(f"Unknown analysis mode: {analysis_mode}.")
    return base_run, delta_psi, mode_run, negative_run, selected_amplitude, selected_n, selected_negative_run_id, selected_run_id, time, z


@app.cell(hide_code=True)
def _(analysis_mode, delta_psi, mo, selected_amplitude, selected_n, selected_negative_run_id, selected_run_id, selected_setup, z):
    response_text = (
        f"Reference: `{selected_setup['base_id']}`"
        if analysis_mode == "one_sided"
        else f"Negative-amplitude partner: `{selected_negative_run_id}`"
    )
    mo.md(
        f"## Selected run\n\n"
        f"$\\nu={selected_setup['NU']:.6e}$, $\\mu={selected_setup['MU']:.6e}$  \n"
        f"Analysis mode: `{analysis_mode}`  \n"
        f"Initial-phi family: `{selected_setup['base_phi_type']}` $\\to$ `{selected_setup['mode_phi_type']}`  \n"
        f"{response_text}  \n"
        f"Positive single-mode run: `{selected_run_id}`; injected $n={selected_n}$; amplitude: `{selected_amplitude:.6g}`  \n"
        f"Saved frames: `{delta_psi.shape[0]}`; spatial points: `{z.size}`."
    )
    return


@app.cell
def _(analysis_mode, base_run, get_rbc_cmap, mo, plot_psi, selected_setup):
    mo.stop(
        analysis_mode != "one_sided",
        mo.md("No smooth reference is used in paired-amplitude analysis."),
    )
    assert base_run is not None
    base_plot = plot_psi(
        base_run,
        vmin=0.0,
        vmax=100.0,
        cmap=get_rbc_cmap(),
        title=(
            rf"{selected_setup['base_phi_type']} reference $\psi(z,t)$ "
            rf"($\nu={selected_setup['NU']:.3e}$, $\mu={selected_setup['MU']:.3e}$)"
        ),
    )
    base_plot
    return


@app.cell
def _(get_rbc_cmap, mode_run, plot_psi, selected_n, selected_setup):
    mode_plot = plot_psi(
        mode_run,
        vmin=0.0,
        vmax=100.0,
        cmap=get_rbc_cmap(),
        title=(
            rf"{selected_setup['mode_phi_type']} $\psi(z,t)$ for injected $n={selected_n} "
            rf"($\nu={selected_setup['NU']:.3e}$, $\mu={selected_setup['MU']:.3e}$)"
        ),
    )
    mode_plot
    return


@app.cell
def _(dct, delta_psi):
    # Shape: (time, DCT mode). The DCT-II acts along the spatial z axis.
    dct_amplitudes = dct(delta_psi, type=2, norm="ortho", axis=1)
    return (dct_amplitudes,)


@app.cell
def _(OptimizeWarning, curve_fit, dct_amplitudes, np, pd, time, warnings):
    def exponential_with_offset(t, offset, initial_amplitude, gamma):
        # Clipping avoids numerical overflow while leaving physically relevant fits unchanged.
        return offset + initial_amplitude * np.exp(np.clip(gamma * t, -700.0, 700.0))

    def fit_mode(values: np.ndarray) -> dict[str, object]:
        if not np.all(np.isfinite(values)):
            return {"success": False, "message": "Non-finite coefficient values."}
        offset_guess = float(values[-1])
        amplitude_guess = float(values[0] - offset_guess)
        if np.isclose(amplitude_guess, 0.0, atol=np.finfo(float).eps):
            return {"success": False, "message": "Coefficient is indistinguishable from a constant."}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                parameters, covariance = curve_fit(
                    exponential_with_offset,
                    time,
                    values,
                    p0=(offset_guess, amplitude_guess, 0.0),
                    maxfev=20_000,
                )
        except (RuntimeError, ValueError, FloatingPointError, OptimizeWarning) as exc:
            return {"success": False, "message": str(exc)}

        fitted = exponential_with_offset(time, *parameters)
        residual_ss = float(np.sum((values - fitted) ** 2))
        total_ss = float(np.sum((values - np.mean(values)) ** 2))
        r_squared = np.nan if np.isclose(total_ss, 0.0) else 1.0 - residual_ss / total_ss
        gamma_std = float(np.sqrt(covariance[2, 2])) if np.isfinite(covariance[2, 2]) and covariance[2, 2] >= 0 else np.nan
        return {
            "success": True,
            "message": "",
            "offset": float(parameters[0]),
            "initial_amplitude": float(parameters[1]),
            "gamma": float(parameters[2]),
            "gamma_std": gamma_std,
            "r_squared": r_squared,
            "fitted": fitted,
        }

    fit_results = [fit_mode(dct_amplitudes[:, mode]) for mode in range(dct_amplitudes.shape[1])]
    fit_table = pd.DataFrame(
        [
            {
                "m": mode,
                "success": result["success"],
                "gamma [s^-1]": result.get("gamma", np.nan),
                "gamma std [s^-1]": result.get("gamma_std", np.nan),
                "A0": result.get("initial_amplitude", np.nan),
                "C": result.get("offset", np.nan),
                "R²": result.get("r_squared", np.nan),
                "message": result["message"],
            }
            for mode, result in enumerate(fit_results)
        ]
    )
    return exponential_with_offset, fit_results, fit_table


@app.cell
def _(fit_table, mo):
    mo.ui.table(data=fit_table, selection=None, pagination=True)
    return


@app.cell
def _(np):
    def relative_log_amplitude(values: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        """Return log |A(t) / A(0)|, or an explanation when it is undefined."""
        if not np.all(np.isfinite(values)):
            return None, "Non-finite DCT coefficients."
        if values[0] == 0.0:
            return None, r"$A_{m,n}(0)=0$."
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.log(np.abs(values / values[0]))
        if not np.all(np.isfinite(result)):
            return None, r"$R_{m,n}(t)$ is non-finite."
        return result, None

    def fit_linear_plateau(
        time: np.ndarray, values: np.ndarray, min_samples: int = 10
    ) -> dict[str, object]:
        """Fit R(t) = a min(t, tau) + b by scanning valid saved-time breakpoints."""
        if time.ndim != 1 or values.ndim != 1 or time.shape != values.shape:
            return {"success": False, "message": "Time and R arrays must be one-dimensional and equal length."}
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(values)):
            return {"success": False, "message": "Time or R values are non-finite."}
        if time.size < 2 * min_samples + 1:
            return {"success": False, "message": f"Need at least {2 * min_samples + 1} saved frames."}

        best: dict[str, object] | None = None
        # The transition sample is included in the linear portion; the following
        # min_samples saved frames establish the plateau.
        for tau_index in range(min_samples - 1, time.size - min_samples):
            tau = float(time[tau_index])
            design = np.column_stack((np.minimum(time, tau), np.ones_like(time)))
            coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
            fitted = design @ coefficients
            residual_ss = float(np.sum((values - fitted) ** 2))
            if best is None or residual_ss < best["residual_ss"]:
                best = {
                    "tau": tau,
                    "tau_index": tau_index,
                    "a": float(coefficients[0]),
                    "b": float(coefficients[1]),
                    "fitted": fitted,
                    "residual_ss": residual_ss,
                }

        assert best is not None
        total_ss = float(np.sum((values - np.mean(values)) ** 2))
        best["r_squared"] = np.nan if np.isclose(total_ss, 0.0) else 1.0 - best["residual_ss"] / total_ss
        best["success"] = True
        best["message"] = ""
        return best

    return fit_linear_plateau, relative_log_amplitude


@app.cell
def _(dct_amplitudes, mo, selected_n):
    inspect_mode = min(selected_n, dct_amplitudes.shape[1] - 1)
    dct_mode_selector = mo.ui.slider(
        start=0,
        stop=dct_amplitudes.shape[1] - 1,
        step=1,
        value=inspect_mode,
        label="Inspect DCT-II mode m",
        full_width=True,
        show_value=True,
    )
    dct_mode_selector
    return (dct_mode_selector,)


@app.cell
def _(dct_amplitudes, np, plt, selected_n, time):
    _figure, _axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for _mode in range(dct_amplitudes.shape[1]):
        _color = "#dc2626" if _mode == selected_n else "#64748b"
        _alpha = 0.9 if _mode == selected_n else 0.12
        _linewidth = 1.8 if _mode == selected_n else 0.6
        _axis.plot(time, dct_amplitudes[:, _mode], color=_color, alpha=_alpha, linewidth=_linewidth)
    _axis.set_xlabel(r"$t\;[\mathrm{s}]$")
    _axis.set_ylabel(r"$A_{m,n}(t)$")
    _axis.set_title(rf"All signed DCT-II coefficients for injected $n={selected_n}$ (highlight: $m=n$)")
    _axis.grid(True, alpha=0.3)
    _figure
    return


@app.cell
def _(dct_amplitudes, dct_mode_selector, plt, selected_n, time):
    _mode = int(dct_mode_selector.value)
    _figure, _axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    _axis.plot(time, dct_amplitudes[:, _mode], color="#2563eb", linewidth=1.5, label=rf"$A_{{{_mode},{selected_n}}}(t)$")
    _axis.set_xlabel(r"$t\;[\mathrm{s}]$")
    _axis.set_ylabel(rf"$A_{{{_mode},{selected_n}}}(t)$")
    _axis.set_title(rf"DCT-II mode $m={_mode}$ for injected $n={selected_n}$")
    _axis.grid(True, alpha=0.3)
    _axis.legend()
    _figure
    return


@app.cell
def _(analysis_mode, dct_amplitudes, dct_mode_selector, fit_linear_plateau, mo, plt, relative_log_amplitude, selected_n, time):
    mo.stop(
        analysis_mode != "one_sided",
        mo.md(r"$R_{m,n}(t)$ is shown only for one-sided analysis."),
    )
    _mode = int(dct_mode_selector.value)
    _amplitudes = dct_amplitudes[:, _mode]
    _relative_log_amplitude, _error = relative_log_amplitude(_amplitudes)

    _figure, _axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if _relative_log_amplitude is None:
        _axis.text(0.02, 0.98, f"Fit unavailable: {_error}", transform=_axis.transAxes, va="top", color="#b91c1c")
    else:
        _fit = fit_linear_plateau(time, _relative_log_amplitude)
        _axis.plot(time, _relative_log_amplitude, color="#7c3aed", linewidth=1.5, label=rf"$R_{{{_mode},{selected_n}}}(t)$")
        if _fit["success"]:
            _axis.plot(time, _fit["fitted"], color="#dc2626", linestyle="--", linewidth=2.0, label=rf"fit: $a={_fit['a']:.4e}\,\mathrm{{s}}^{{-1}}$")
            _axis.axvline(_fit["tau"], color="#dc2626", linestyle=":", linewidth=1.5, label=rf"$\tau={_fit['tau']:.4g}\,\mathrm{{s}}$")
        else:
            _axis.text(0.02, 0.98, f"Fit unavailable: {_fit['message']}", transform=_axis.transAxes, va="top", color="#b91c1c")
    _axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    _axis.set_xlabel(r"$t\;[\mathrm{s}]$")
    _axis.set_ylabel(rf"$R_{{{_mode},{selected_n}}}(t)$")
    _axis.set_title(rf"$R_{{{_mode},{selected_n}}}(t)=\ln\left|A_{{{_mode},{selected_n}}}(t) / A_{{{_mode},{selected_n}}}(0)\right|$")
    _axis.grid(True, alpha=0.3)
    _axis.legend()
    _figure
    return


@app.cell
def _(analysis_mode, dct_amplitudes, dct_mode_selector, mo, np, plt, selected_n, time):
    mo.stop(
        analysis_mode != "paired",
        mo.md(r"$P_{m,n}(t)$ is shown only for paired-amplitude analysis."),
    )
    _selected_mode = int(dct_mode_selector.value)
    _denominator = dct_amplitudes[0, selected_n]
    if _denominator == 0.0:
        _relative_amplitudes = np.full_like(dct_amplitudes, np.nan)
        _message = (
            rf"$P_{{m,{selected_n}}}(t)$ is undefined because "
            rf"$A_{{{selected_n},{selected_n}}}(0)=0$."
        )
    else:
        _relative_amplitudes = dct_amplitudes / _denominator
        _message = None

    _figure, _axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if _message is not None:
        _axis.text(0.02, 0.98, _message, transform=_axis.transAxes, va="top", color="#b91c1c")
    for _mode in range(_relative_amplitudes.shape[1]):
        _is_selected = _mode == _selected_mode
        _is_denominator = _mode == selected_n
        _color = "#2563eb" if _is_selected else "#dc2626" if _is_denominator else "#64748b"
        _alpha = 1.0 if _is_selected else 0.85 if _is_denominator else 0.14
        _linewidth = 2.2 if _is_selected else 1.6 if _is_denominator else 0.6
        _label = (
            rf"selected $P_{{{_selected_mode},{selected_n}}}(t)$"
            if _is_selected
            else rf"$P_{{{selected_n},{selected_n}}}(t)$"
            if _is_denominator
            else None
        )
        _axis.plot(
            time,
            _relative_amplitudes[:, _mode],
            color=_color,
            alpha=_alpha,
            linewidth=_linewidth,
            label=_label,
        )
    _axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    _axis.set_xlabel(r"$t\;[\mathrm{s}]$")
    _axis.set_ylabel(rf"$P_{{m,{selected_n}}}(t)$")
    _axis.set_title(rf"$P_{{m,{selected_n}}}(t)=A_{{m,{selected_n}}}(t) / A_{{{selected_n},{selected_n}}}(0)$; selected $m={_selected_mode}$")
    _axis.grid(True, alpha=0.3)
    _axis.legend()
    _figure
    return


@app.cell
def _(analysis_mode, dct_amplitudes, dct_mode_selector, mo, np, plt, selected_n, time):
    mo.stop(
        analysis_mode != "paired",
        mo.md(r"Short-time $P_{m,n}(t)$ is shown only for paired-amplitude analysis."),
    )
    _selected_mode = int(dct_mode_selector.value)
    _denominator = dct_amplitudes[0, selected_n]
    _short_time = (time >= 0.0) & (time <= 40.0)
    _figure, _axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if _denominator == 0.0:
        _axis.text(
            0.02,
            0.98,
            rf"Undefined because $A_{{{selected_n},{selected_n}}}(0)=0$.",
            transform=_axis.transAxes,
            va="top",
            color="#b91c1c",
        )
    elif np.count_nonzero(_short_time) < 2:
        _axis.text(
            0.02,
            0.98,
            "Need at least two saved times in [0, 40 s] to estimate the slope.",
            transform=_axis.transAxes,
            va="top",
            color="#b91c1c",
        )
    else:
        _relative_amplitude = dct_amplitudes[:, _selected_mode] / _denominator
        _short_times = time[_short_time]
        _short_relative_amplitude = _relative_amplitude[_short_time]
        _edge_order = 2 if _short_times.size >= 3 else 1
        _slope = float(np.gradient(_short_relative_amplitude, _short_times, edge_order=_edge_order)[0])
        _intercept = 1.0 if _selected_mode == selected_n else 0.0
        _linear_approximation = _intercept + _slope * _short_times
        _axis.plot(
            _short_times,
            _short_relative_amplitude,
            color="#2563eb",
            linewidth=1.8,
            label=rf"$P_{{{_selected_mode},{selected_n}}}(t)$",
        )
        _axis.plot(
            _short_times,
            _linear_approximation,
            color="#dc2626",
            linestyle="--",
            linewidth=2.0,
            label=rf"$L_{{{_selected_mode},{selected_n}}}={_slope:.4e}\,\mathrm{{s}}^{{-1}}$",
        )
    _axis.set_xlabel(r"$t\;[\mathrm{s}]$")
    _axis.set_ylabel(rf"$P_{{{_selected_mode},{selected_n}}}(t)$")
    _axis.set_title(rf"Short-time $P_{{{_selected_mode},{selected_n}}}(t)$ and its prescribed linear approximation")
    _axis.grid(True, alpha=0.3)
    _axis.legend()
    _figure
    return


@app.cell
def _(RunData, analysis_mode, dct, mo, np, selected_setup, sweep_dir):
    mo.stop(
        analysis_mode != "paired",
        mo.md(r"$L_{m,n}$ is shown only for paired-amplitude analysis."),
    )
    _mode_numbers = np.asarray(selected_setup["mode_numbers"], dtype=int)
    _slopes = np.full((int(selected_setup["N"]), _mode_numbers.size), np.nan)
    _failures: list[str] = []
    for _column, (_n, _positive_id, _negative_id, _amplitude) in enumerate(
        zip(
            selected_setup["mode_numbers"],
            selected_setup["mode_ids"],
            selected_setup["negative_mode_ids"],
            selected_setup["amplitudes"],
            strict=True,
        )
    ):
        try:
            _positive_run = RunData.from_h5(
                sweep_dir / "results" / _positive_id / "run.h5",
                load_fields=False,
            )
            _negative_run = RunData.from_h5(
                sweep_dir / "results" / str(_negative_id) / "run.h5",
                load_fields=False,
            )
            _time = np.asarray(_positive_run.time, dtype=np.float64)
            _positive_psi = np.asarray(_positive_run.load_psi(), dtype=np.float64)
            _negative_time = np.asarray(_negative_run.time, dtype=np.float64)
            _negative_z = np.asarray(_negative_run.z, dtype=np.float64)
            _negative_psi = np.asarray(_negative_run.load_psi(), dtype=np.float64)
            if (
                not np.array_equal(_negative_time, _time)
                or not np.array_equal(_negative_z, np.asarray(_positive_run.z, dtype=np.float64))
                or _negative_psi.shape != _positive_psi.shape
            ):
                raise ValueError("Positive and negative grids or psi shapes differ.")
            if _time.size < 2:
                raise ValueError("Need at least two saved times.")

            _coefficients = dct(
                (_positive_psi - _negative_psi) / (2.0 * float(_amplitude)),
                type=2,
                norm="ortho",
                axis=1,
            )
            _denominator = _coefficients[0, _n]
            if _denominator == 0.0:
                raise ValueError(r"$A_{n,n}(0)=0$.")
            _relative_amplitudes = _coefficients / _denominator
            _edge_order = 2 if _time.size >= 3 else 1
            _slopes[:, _column] = np.gradient(
                _relative_amplitudes,
                _time,
                axis=0,
                edge_order=_edge_order,
            )[0]
        except Exception as _exc:
            _failures.append(f"n={_n}: {_exc}")

    slope_heatmap_failures = _failures
    slope_heatmap_modes = _mode_numbers
    slope_heatmap_values = _slopes
    return slope_heatmap_failures, slope_heatmap_modes, slope_heatmap_values


@app.cell(hide_code=True)
def _(mo, slope_heatmap_failures):
    if slope_heatmap_failures:
        _status = mo.callout(
            "Could not calculate some heatmap columns: " + "; ".join(slope_heatmap_failures),
            kind="warn",
        )
    else:
        _status = mo.md("")
    _status
    return


@app.cell
def _(np, plt, slope_heatmap_modes, slope_heatmap_values):
    _figure, _axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    _image = _axis.imshow(
        slope_heatmap_values,
        origin="lower",
        aspect="auto",
        interpolation="none",
        cmap="coolwarm",
    )
    _axis.set_xlabel("Injected mode n")
    _axis.set_ylabel("DCT-II mode m")
    _axis.set_title(r"Initial relative-mode slope $L_{m,n}=\partial_t P_{m,n}(0)$")
    _axis.set_xticks(np.arange(slope_heatmap_modes.size))
    _axis.set_xticklabels(slope_heatmap_modes)
    _axis.set_ylim(-0.5, slope_heatmap_values.shape[0] - 0.5)
    _figure.colorbar(_image, ax=_axis, label=r"$L_{m,n}\;[\mathrm{s}^{-1}]$")
    _figure
    return


@app.cell
def _(analysis_mode, mo, np, slope_heatmap_modes, slope_heatmap_values):
    mo.stop(
        analysis_mode != "paired",
        mo.md(r"$L_{n\pm\Delta,n}$ is shown only for paired-amplitude analysis."),
    )
    _max_delta = min(
        int(np.max(slope_heatmap_modes)),
        slope_heatmap_values.shape[0] - 1 - int(np.min(slope_heatmap_modes)),
    )
    delta_selector = mo.ui.slider(
        start=0,
        stop=_max_delta,
        step=1,
        value=0,
        label=r"Mode offset $\Delta$",
        full_width=True,
        show_value=True,
    )
    delta_selector
    return (delta_selector,)


@app.cell
def _(analysis_mode, delta_selector, mo, np, plt, slope_heatmap_modes, slope_heatmap_values):
    mo.stop(
        analysis_mode != "paired",
        mo.md(r"$L_{n\pm\Delta,n}$ is shown only for paired-amplitude analysis."),
    )
    _delta = int(delta_selector.value)
    _n = np.asarray(slope_heatmap_modes, dtype=int)
    _column_indices = np.arange(_n.size)
    _plus_m = _n + _delta
    _minus_m = _n - _delta
    _plus_valid = _plus_m < slope_heatmap_values.shape[0]
    _minus_valid = _minus_m >= 0

    _figure, _axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    _axis.plot(
        _n[_plus_valid],
        slope_heatmap_values[_plus_m[_plus_valid], _column_indices[_plus_valid]],
        color="#2563eb",
        marker="o",
        linewidth=1.8,
        label=rf"$L_{{n+{_delta},n}}$",
    )
    if _delta > 0:
        _axis.plot(
            _n[_minus_valid],
            slope_heatmap_values[_minus_m[_minus_valid], _column_indices[_minus_valid]],
            color="#dc2626",
            marker="s",
            linewidth=1.8,
            label=rf"$L_{{n-{_delta},n}}$",
        )
    _axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    _axis.set_xlabel("Injected mode n")
    _axis.set_ylabel(r"$L_{m,n}\;[\mathrm{s}^{-1}]$")
    _axis.set_title(rf"Initial mode-coupling slopes for $\Delta={_delta}$")
    _axis.grid(True, alpha=0.3)
    _axis.legend()
    _figure
    return


@app.cell
def _(RunData, analysis_mode, candidate_df, dct, fit_linear_plateau, np, pd, relative_log_amplitude, sweep_dir):
    diagonal_rows: list[dict[str, object]] = []
    for _, _setup in candidate_df.iterrows():
        _pair_label = f"ν={_setup['NU']:.3e}, μ={_setup['MU']:.3e}"
        for _n, _positive_id, _negative_id, _amplitude in zip(
            _setup["mode_numbers"],
            _setup["mode_ids"],
            _setup["negative_mode_ids"],
            _setup["amplitudes"],
            strict=True,
        ):
            _row = {
                "NU": _setup["NU"],
                "MU": _setup["MU"],
                "pair": _pair_label,
                "n": _n,
                "positive_run_id": _positive_id,
                "negative_run_id": _negative_id,
                "amplitude": _amplitude,
            }
            try:
                _positive_run = RunData.from_h5(
                    sweep_dir / "results" / _positive_id / "run.h5",
                    load_fields=False,
                )
                _time = np.asarray(_positive_run.time, dtype=np.float64)
                _z = np.asarray(_positive_run.z, dtype=np.float64)
                _positive_psi = np.asarray(_positive_run.load_psi(), dtype=np.float64)
                if analysis_mode == "one_sided":
                    _base_run = RunData.from_h5(
                        sweep_dir / "results" / _setup["base_id"] / "run.h5",
                        load_fields=False,
                    )
                    _base_time = np.asarray(_base_run.time, dtype=np.float64)
                    _base_z = np.asarray(_base_run.z, dtype=np.float64)
                    _base_psi = np.asarray(_base_run.load_psi(), dtype=np.float64)
                    if not np.array_equal(_base_time, _time) or not np.array_equal(_base_z, _z) or _base_psi.shape != _positive_psi.shape:
                        raise ValueError("Saved-time grid, z grid, or psi shape differs from the reference.")
                    _delta_psi = _positive_psi - _base_psi
                elif analysis_mode == "paired":
                    _negative_run = RunData.from_h5(
                        sweep_dir / "results" / str(_negative_id) / "run.h5",
                        load_fields=False,
                    )
                    _negative_time = np.asarray(_negative_run.time, dtype=np.float64)
                    _negative_z = np.asarray(_negative_run.z, dtype=np.float64)
                    _negative_psi = np.asarray(_negative_run.load_psi(), dtype=np.float64)
                    if not np.array_equal(_negative_time, _time) or not np.array_equal(_negative_z, _z) or _negative_psi.shape != _positive_psi.shape:
                        raise ValueError("Saved-time grid, z grid, or psi shape differs between the positive and negative runs.")
                    _delta_psi = (_positive_psi - _negative_psi) / (2.0 * float(_amplitude))
                else:
                    raise ValueError(f"Unknown analysis mode: {analysis_mode}.")

                _coefficients = dct(_delta_psi, type=2, norm="ortho", axis=1)
                if _n >= _coefficients.shape[1]:
                    raise ValueError(f"Injected mode n={_n} is outside the DCT coefficient range.")
                _relative_log, _error = relative_log_amplitude(_coefficients[:, _n])
                if _relative_log is None:
                    raise ValueError(_error)
                _fit = fit_linear_plateau(_time, _relative_log)
                if not _fit["success"]:
                    raise ValueError(_fit["message"])
                diagonal_rows.append({**_row, "success": True, "a_n [s^-1]": _fit["a"], "b_n": _fit["b"], "tau [s]": _fit["tau"], "R²": _fit["r_squared"], "message": ""})
            except Exception as _exc:
                diagonal_rows.append({**_row, "success": False, "a_n [s^-1]": np.nan, "b_n": np.nan, "tau [s]": np.nan, "R²": np.nan, "message": str(_exc)})

    diagonal_fit_df = pd.DataFrame(diagonal_rows)
    return (diagonal_fit_df,)


@app.cell
def _(diagonal_fit_df, mo):
    mo.ui.table(data=diagonal_fit_df, selection=None, pagination=True)
    return


@app.cell
def _(diagonal_fit_df, json, np, plt, selected_setup, z):
    _figure, _axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    _valid = diagonal_fit_df[diagonal_fit_df["success"]]
    for _pair, _pair_df in _valid.groupby("pair", sort=True):
        _pair_df = _pair_df.sort_values("n", kind="stable")
        _axis.plot(_pair_df["n"], _pair_df["a_n [s^-1]"], marker="o", linewidth=1.6, label=_pair)

    _selected_pair = f"ν={selected_setup['NU']:.3e}, μ={selected_setup['MU']:.3e}"
    _selected_fits = _valid[_valid["pair"] == _selected_pair].sort_values("n", kind="stable")
    _phi_params = json.loads(selected_setup["shared_phi"])
    _psi_zero = float(_phi_params["psi_avg"])
    _length = float(z[-1] - z[0])
    _n = _selected_fits["n"].to_numpy(dtype=np.float64)
    _measured_a = _selected_fits["a_n [s^-1]"].to_numpy(dtype=np.float64)
    if _length <= 0.0:
        _axis.text(0.02, 0.98, "Theory overlay unavailable: z span is zero.", transform=_axis.transAxes, va="top", color="#b91c1c")
    else:
        _k = _n * np.pi / _length
        _theory_basis = (_psi_zero / 90e-18) * (
            -float(selected_setup["NU"]) * _k**2 + float(selected_setup["MU"]) * _k**4
        )
        _fit_mask = np.isfinite(_theory_basis) & np.isfinite(_measured_a) & (_theory_basis != 0.0)
        if not np.any(_fit_mask):
            _axis.text(0.02, 0.98, "Theory overlay unavailable: no nonzero valid modes for Γ fit.", transform=_axis.transAxes, va="top", color="#b91c1c")
        else:
            _gamma_hat = float(np.dot(_theory_basis[_fit_mask], _measured_a[_fit_mask]) / np.dot(_theory_basis[_fit_mask], _theory_basis[_fit_mask]))
            _axis.plot(
                _n,
                _gamma_hat * _theory_basis,
                color="#111827",
                linestyle="--",
                marker="x",
                linewidth=2.0,
                label=rf"theory ({_selected_pair}): $\hat{{\Gamma}}={_gamma_hat:.4e}$",
            )
    _axis.set_xlabel("Injected mode n")
    _axis.set_ylabel(r"$a_n\;[\mathrm{s}^{-1}]$")
    _axis.set_title(r"Early-time diagonal growth rate from $R_{n,n}(t)=a_nt+b_n$ for $t<\tau$; dashed line fits $\Gamma$ for the selected pair")
    _axis.grid(True, alpha=0.3)
    if not _valid.empty:
        _axis.legend()
    _figure
    return


@app.cell
def _(analysis_mode, mode_run, mo, negative_run, np, selected_amplitude):
    mo.stop(
        analysis_mode != "paired",
        mo.md("This initial-response diagnostic requires paired positive/negative runs."),
    )
    assert negative_run is not None
    _positive_phi = np.asarray(mode_run.phi_frame(0), dtype=np.float64)
    _negative_phi = np.asarray(negative_run.phi_frame(0), dtype=np.float64)
    _positive_rho = np.asarray(mode_run.rho, dtype=np.float64)
    _negative_rho = np.asarray(negative_run.rho, dtype=np.float64)
    if not np.array_equal(_positive_rho, _negative_rho):
        raise ValueError("Positive and negative runs have different rho grids.")
    if _positive_phi.shape != _negative_phi.shape:
        raise ValueError("Positive and negative initial phi frames have different shapes.")

    delta_phi_initial = (_positive_phi - _negative_phi) / (2.0 * selected_amplitude)
    delta_psi_phi_integral = np.trapezoid(delta_phi_initial, x=_positive_rho, axis=0)
    # This matches the CUDA definition of psi, which is a discrete radial sum.
    delta_psi_phi_sum = np.sum(delta_phi_initial, axis=0)
    delta_psi_psi_initial = (
        np.asarray(mode_run.psi_frame(0), dtype=np.float64)
        - np.asarray(negative_run.psi_frame(0), dtype=np.float64)
    ) / (2.0 * selected_amplitude)
    return (
        delta_phi_initial,
        delta_psi_phi_integral,
        delta_psi_phi_sum,
        delta_psi_psi_initial,
    )


@app.cell(hide_code=True)
def _(mo, selected_n):
    mo.md(
        r"""
        ## Initial paired response for injected \(n="""
        + str(selected_n)
        + r"""\)

        \[
        \delta\varphi_n(\rho,z,0)
        =
        \frac{\varphi_{+,n}(\rho,z,0)-\varphi_{-,n}(\rho,z,0)}
        {2\varepsilon_n}.
        \]

        The next plots compare its numerical \(\rho\)-integral with the
        centered difference calculated directly from the stored \(\psi\) fields.
        """
    )
    return


@app.cell
def _(delta_phi_initial, mode_run, plt):
    _figure, _axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    _image = _axis.pcolormesh(
        mode_run.z,
        mode_run.rho,
        delta_phi_initial,
        shading="auto",
        cmap="RdBu_r",
    )
    _axis.set_xlabel(r"$z\;[\mathrm{m}]$")
    _axis.set_ylabel(r"$\rho$")
    _axis.set_title(r"$\delta\varphi_n(\rho,z,0)$")
    _figure.colorbar(_image, ax=_axis, label=r"$\delta\varphi_n$")
    _figure
    return


@app.cell
def _(delta_psi_phi_integral, delta_psi_phi_sum, delta_psi_psi_initial, mode_run, plt):
    _figure, _axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    _axis.plot(
        mode_run.z,
        delta_psi_phi_integral,
        color="#7c3aed",
        linewidth=1.8,
        label=r"$\delta\psi_n^\varphi(z,0)=\int\delta\varphi_n\,d\rho$",
    )
    _axis.plot(
        mode_run.z,
        delta_psi_psi_initial,
        color="#2563eb",
        linestyle="--",
        linewidth=1.8,
        label=r"$\delta\psi_n^\psi(z,0)$ from stored $\psi$",
    )
    _axis.plot(
        mode_run.z,
        delta_psi_phi_sum,
        color="#dc2626",
        linestyle=":",
        linewidth=1.8,
        label=r"$\sum_\rho\delta\varphi_n$ (solver-consistent)",
    )
    _axis.set_xlabel(r"$z\;[\mathrm{m}]$")
    _axis.set_ylabel(r"Centered initial response")
    _axis.set_title(r"Initial longitudinal response from $\varphi$ and $\psi$")
    _axis.grid(True, alpha=0.3)
    _axis.legend()
    _figure
    return


if __name__ == "__main__":
    app.run()
