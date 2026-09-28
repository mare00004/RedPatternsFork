import marimo

__generated_with = "0.23.9"
app = marimo.App(width="wide")


with app.setup:
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.fft import dct

    NOTEBOOK_FILE = (
        Path(__file__).resolve()
        if "__file__" in globals()
        else (Path.cwd() / "analysis" / "analyze_perturbation_dct_modes.py").resolve()
    )
    REPO_ROOT = NOTEBOOK_FILE.parent.parent

    from red_patterns import (
        RunData,
        SweepCatalog,
        get_rbc_cmap,
        plot_psi,
        selected_sweep_catalog,
        sweep_directory_picker,
    )
    from red_patterns.models import TaylorRun
    from red_patterns.types import PhiType


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Seed-ensemble DCT-II mode analysis

    Choose a Taylor sweep directory containing `runs.jsonl` and
    `results/<run_id>/run.h5`. For each selected $(\nu, \mu)$ pair, the notebook
    subtracts the matching unperturbed base run from every seeded perturbed run,
    computes a spatial orthonormal DCT-II, and averages squared mode magnitudes
    over seeds. Both the smooth-homogeneous and linear-gradient-diagonal phi
    families are supported.
    """)
    return


@app.cell
def _(Path, sweep_directory_picker):
    ui_sweep_dir = sweep_directory_picker(
        mo,
        initial_path=Path.cwd(),
        label="Choose Taylor sweep directory",
    )
    ui_sweep_dir
    return (ui_sweep_dir,)


@app.cell
def _(SweepCatalog, TaylorRun, pd):
    def scan_sweep(catalog: SweepCatalog) -> pd.DataFrame:
        """Load candidate Taylor base and perturbed runs plus their result paths."""
        rows: list[dict[str, object]] = []
        for entry in catalog.entries:
            run = entry.run
            if not isinstance(run, TaylorRun):
                continue

            phi_params = run.phi.params.model_dump(mode="json")
            phi_type = str(phi_params.pop("phi_type"))
            family_by_type = {
                PhiType.SMOOTH_HOMOGENEOUS.value: (
                    PhiType.SMOOTH_HOMOGENEOUS.value,
                    "amplitude",
                ),
                PhiType.PERTURBED_SMOOTH_HOMOGENEOUS.value: (
                    PhiType.SMOOTH_HOMOGENEOUS.value,
                    "amplitude",
                ),
                PhiType.LINEAR_FULL_RIDGE.value: (
                    PhiType.LINEAR_FULL_RIDGE.value,
                    "epsilon",
                ),
                PhiType.PERTURBED_LINEAR_FULL_RIDGE.value: (
                    PhiType.LINEAR_FULL_RIDGE.value,
                    "epsilon",
                ),
            }
            if phi_type not in family_by_type:
                continue

            family, perturbation_name = family_by_type[phi_type]
            seed = phi_params.pop("seed", None)
            perturbation = phi_params.pop(perturbation_name, None)
            # These identify the random displacement, not the shared base setup.
            # Remove them so a perturbed linear ridge can match its base ridge.
            phi_params.pop("mode_min", None)
            phi_params.pop("mode_max", None)
            shared_phi = json.dumps(phi_params, sort_keys=True, separators=(",", ":"))
            result_path = entry.run_h5
            rows.append(
                {
                    "run_id": run.run_id,
                    "NU": float(run.NU),
                    "MU": float(run.MU),
                    "phi_type": phi_type,
                    "family": family,
                    "seed": None if seed is None else int(seed),
                    "perturbation_name": perturbation_name,
                    "perturbation": None if perturbation is None else float(perturbation),
                    "shared_phi": shared_phi,
                    "N": int(run.N),
                    "T": float(run.T),
                    "DT": float(run.DT),
                    "storeTime": float(run.storeTime),
                    "gradient": run.gradient.value,
                    "run_h5": result_path,
                    "h5_exists": entry.h5_exists,
                }
            )

        return pd.DataFrame(rows)

    def validate_ensembles(sweep_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return valid pair candidates and diagnostics for every discovered pair."""
        columns = [
            "NU", "MU", "family", "base_id", "seed_ids", "seeds",
            "perturbation_name", "perturbation",
        ]
        diagnostic_columns = ["NU", "MU", "status", "details"]
        if sweep_df.empty:
            return pd.DataFrame(columns=columns), pd.DataFrame(columns=diagnostic_columns)

        candidates: list[dict[str, object]] = []
        diagnostics: list[dict[str, object]] = []
        setup_columns = ["shared_phi", "N", "T", "DT", "storeTime", "gradient"]
        families = (
            (
                PhiType.SMOOTH_HOMOGENEOUS.value,
                PhiType.PERTURBED_SMOOTH_HOMOGENEOUS.value,
            ),
            (
                PhiType.LINEAR_FULL_RIDGE.value,
                PhiType.PERTURBED_LINEAR_FULL_RIDGE.value,
            ),
        )
        for (nu, mu), pair_df in sweep_df.groupby(["NU", "MU"], sort=True):
            for base_type, perturbed_type in families:
                family_df = pair_df[pair_df["phi_type"].isin((base_type, perturbed_type))]
                base_df = family_df[family_df["phi_type"] == base_type]
                seed_df = family_df[family_df["phi_type"] == perturbed_type]
                family_label = str(family_df["family"].iloc[0]) if not family_df.empty else base_type
                if base_df.empty and seed_df.empty:
                    continue
                if len(base_df) != 1:
                    diagnostics.append(
                        {
                            "NU": nu,
                            "MU": mu,
                            "status": "invalid",
                            "details": f"{family_label}: expected exactly one base run; found {len(base_df)}.",
                        }
                    )
                    continue
                if seed_df.empty:
                    diagnostics.append(
                        {"NU": nu, "MU": mu, "status": "invalid", "details": f"{family_label}: no perturbed seed runs found."}
                    )
                    continue

                base_row = base_df.iloc[0]
                mismatched = seed_df[
                    (seed_df[setup_columns] != base_row[setup_columns]).any(axis=1)
                ]
                if not mismatched.empty:
                    diagnostics.append(
                        {
                            "NU": nu,
                            "MU": mu,
                            "status": "invalid",
                            "details": f"{family_label}: seed setup differs from base: " + ", ".join(mismatched["run_id"]),
                        }
                    )
                    continue
                if seed_df["perturbation"].nunique(dropna=False) != 1:
                    diagnostics.append(
                        {
                            "NU": nu,
                            "MU": mu,
                            "status": "invalid",
                            "details": f"{family_label}: perturbation magnitude differs across seeds.",
                        }
                    )
                    continue
                if seed_df["seed"].isna().any() or seed_df["seed"].duplicated().any():
                    diagnostics.append(
                        {
                            "NU": nu,
                            "MU": mu,
                            "status": "invalid",
                            "details": f"{family_label}: seeds must be present and unique.",
                        }
                    )
                    continue

                missing = family_df[~family_df["h5_exists"]]["run_id"].tolist()
                if missing:
                    diagnostics.append(
                        {
                            "NU": nu,
                            "MU": mu,
                            "status": "incomplete",
                            "details": f"{family_label}: missing run.h5 for " + ", ".join(missing),
                        }
                    )
                    continue

                sorted_seeds = seed_df.sort_values("seed", kind="stable")
                candidates.append(
                    {
                        "NU": float(nu),
                        "MU": float(mu),
                        "family": family_label,
                        "base_id": str(base_row["run_id"]),
                        "seed_ids": tuple(sorted_seeds["run_id"].tolist()),
                        "seeds": tuple(int(seed) for seed in sorted_seeds["seed"].tolist()),
                        "perturbation_name": str(sorted_seeds["perturbation_name"].iloc[0]),
                        "perturbation": float(sorted_seeds["perturbation"].iloc[0]),
                    }
                )
                diagnostics.append({"NU": nu, "MU": mu, "status": "ready", "details": f"{family_label}: compatible ensemble."})

        return pd.DataFrame(candidates, columns=columns), pd.DataFrame(diagnostics, columns=diagnostic_columns)

    return scan_sweep, validate_ensembles


@app.cell
def _(mo, pd, scan_sweep, selected_sweep_catalog, ui_sweep_dir, validate_ensembles):
    catalog, scan_status = selected_sweep_catalog(mo, ui_sweep_dir)
    if catalog is None:
        sweep_df = pd.DataFrame()
        ensemble_df = pd.DataFrame()
        diagnostics_df = pd.DataFrame()
    else:
        try:
            sweep_df = scan_sweep(catalog)
        except ValueError as exc:
            sweep_df = pd.DataFrame()
            ensemble_df = pd.DataFrame()
            diagnostics_df = pd.DataFrame()
            scan_status = mo.callout(
                f"Could not process `{catalog.root}` with the current sweep schema: {exc}",
                kind="warn",
            )
        else:
            ensemble_df, diagnostics_df = validate_ensembles(sweep_df)
            scan_status = mo.md(
                f"Found `{len(sweep_df)}` smooth/perturbed Taylor runs and "
                f"`{len(ensemble_df)}` compatible ensembles in `{catalog.root}`."
            )

    sweep_dir = catalog.root if catalog is not None else None
    scan_status
    return diagnostics_df, ensemble_df, sweep_dir


@app.cell
def _(diagnostics_df, mo):
    mo.stop(diagnostics_df.empty, mo.md("No supported smooth or linear-gradient-diagonal candidate runs found."))
    mo.ui.table(data=diagnostics_df, selection=None, pagination=True)
    return


@app.cell
def _(ensemble_df, mo):
    mo.stop(ensemble_df.empty, mo.md("No complete, compatible ensembles are available yet."))
    options = {
        f"{row.family}: ν={row.NU:.6e}, μ={row.MU:.6e}": index
        for index, row in ensemble_df.iterrows()
    }
    pair_selector = mo.ui.dropdown(
        options=options,
        value=next(iter(options)),
        label=r"Select $(\nu, \mu)$ ensemble",
    )
    pair_selector
    return (pair_selector,)


@app.cell
def _(ensemble_df, pair_selector):
    selected_ensemble = ensemble_df.loc[int(pair_selector.value)]
    return (selected_ensemble,)


@app.cell
def _(RunData, selected_ensemble, sweep_dir):
    def load_ensemble() -> tuple[RunData, np.ndarray, np.ndarray, np.ndarray]:
        base_path = sweep_dir / "results" / selected_ensemble["base_id"] / "run.h5"
        base_run = RunData.from_h5(base_path, load_fields=False)
        time = np.asarray(base_run.time, dtype=np.float64)
        z = np.asarray(base_run.z, dtype=np.float64)
        base_psi = np.asarray(base_run.load_psi(), dtype=np.float64)

        delta_psi_by_seed: list[np.ndarray] = []
        for run_id in selected_ensemble["seed_ids"]:
            seed_path = sweep_dir / "results" / run_id / "run.h5"
            seed_run = RunData.from_h5(seed_path, load_fields=False)
            seed_time = np.asarray(seed_run.time, dtype=np.float64)
            seed_z = np.asarray(seed_run.z, dtype=np.float64)
            seed_psi = np.asarray(seed_run.load_psi(), dtype=np.float64)
            if not np.array_equal(seed_time, time) or not np.array_equal(seed_z, z):
                raise ValueError(
                    f"{run_id} has a different saved-time or z grid than base run "
                    f"{selected_ensemble['base_id']}."
                )
            if seed_psi.shape != base_psi.shape:
                raise ValueError(
                    f"{run_id} psi shape {seed_psi.shape} differs from base shape {base_psi.shape}."
                )
            delta_psi_by_seed.append(seed_psi - base_psi)

        return base_run, np.stack(delta_psi_by_seed, axis=0), time, z

    base_run, delta_psi_by_seed, time, z = load_ensemble()
    return base_run, delta_psi_by_seed, time, z


@app.cell
def _(base_run, get_rbc_cmap, plot_psi, selected_ensemble):
    base_psi_plot = plot_psi(
        base_run,
        vmin=0.0,
        vmax=100.0,
        cmap=get_rbc_cmap(),
        title=(
            r"Base $\\psi(z,t)$ "
            f"($\\nu={selected_ensemble['NU']:.3e}$, $\\mu={selected_ensemble['MU']:.3e}$)"
        ),
    )
    base_psi_plot
    return (base_psi_plot,)


@app.cell
def _(dct, delta_psi_by_seed, np):
    # Shape: (seed, time, mode). The DCT-II acts along the spatial z axis.
    dct_coefficients = dct(delta_psi_by_seed, type=2, norm="ortho", axis=2)
    amplitudes = np.abs(dct_coefficients)
    mean_powers = np.mean(amplitudes**2, axis=0)
    return amplitudes, mean_powers


@app.cell
def _(np):
    def fit_log_power(
        time: np.ndarray, power: np.ndarray, min_samples: int = 10
    ) -> dict[str, object]:
        """Fit early log power to ``a min(t, tau) + b``.

        For a modal amplitude proportional to ``exp(gamma * t)``, its power
        grows as ``exp(2 * gamma * t)``. Thus the early-time slope is ``2γ``.
        """
        if time.ndim != 1 or power.ndim != 1 or time.shape != power.shape:
            return {"success": False, "message": "Time and power arrays must be one-dimensional and equal length."}
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(power)):
            return {"success": False, "message": "Time or power values are non-finite."}
        if power[0] <= 0.0:
            return {"success": False, "message": "Initial modal power must be positive."}
        if time.size < 2 * min_samples + 1:
            return {"success": False, "message": f"Need at least {2 * min_samples + 1} saved frames."}

        with np.errstate(divide="ignore", invalid="ignore"):
            log_relative_power = np.log(power / power[0])
        if not np.all(np.isfinite(log_relative_power)):
            return {"success": False, "message": "Log-relative power is non-finite."}

        best: dict[str, object] | None = None
        for tau_index in range(min_samples - 1, time.size - min_samples):
            tau = float(time[tau_index])
            design = np.column_stack((np.minimum(time, tau), np.ones_like(time)))
            coefficients, _, _, _ = np.linalg.lstsq(design, log_relative_power, rcond=None)
            fitted = design @ coefficients
            residual_ss = float(np.sum((log_relative_power - fitted) ** 2))
            if best is None or residual_ss < best["residual_ss"]:
                best = {
                    "tau": tau,
                    "slope": float(coefficients[0]),
                    "fitted": fitted,
                    "residual_ss": residual_ss,
                }

        assert best is not None
        total_ss = float(np.sum((log_relative_power - np.mean(log_relative_power)) ** 2))
        best["r_squared"] = np.nan if np.isclose(total_ss, 0.0) else 1.0 - best["residual_ss"] / total_ss
        best["gamma"] = 0.5 * best["slope"]
        best["success"] = True
        best["message"] = ""
        return best

    return (fit_log_power,)


@app.cell
def _(fit_log_power, mean_powers, np, pd, time):
    growth_fits = [
        fit_log_power(time, mean_powers[:, mode])
        for mode in range(mean_powers.shape[1])
    ]
    growth_fit_table = pd.DataFrame(
        [
            {
                "m": mode,
                "success": fit["success"],
                "gamma [s^-1]": fit.get("gamma", np.nan),
                "log-power slope [s^-1]": fit.get("slope", np.nan),
                "tau [s]": fit.get("tau", np.nan),
                "R²": fit.get("r_squared", np.nan),
                "message": fit["message"],
            }
            for mode, fit in enumerate(growth_fits)
        ]
    )
    return growth_fit_table, growth_fits


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Early-time growth-rate estimates

    Each mode's seed-averaged power is fit as
    \(\ln[P_m(t)/P_m(0)] = a_m\min(t,\tau_m)+b_m\). Since
    \(P_m\propto |A_m|^2\), the reported modal growth rate is
    \(\gamma_m=a_m/2\). Treat this as an ensemble screening diagnostic: modes
    seeded at very low power or affected by mode coupling can yield unreliable
    fits.
    """)
    return


@app.cell
def _(growth_fit_table, mo):
    mo.ui.table(data=growth_fit_table, selection=None, pagination=True)
    return


@app.cell
def _(growth_fit_table, plt):
    _figure, _axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    _valid = growth_fit_table[growth_fit_table["success"]]
    _axis.plot(
        _valid["m"],
        _valid["gamma [s^-1]"],
        marker="o",
        linewidth=1.6,
        label="ensemble-power estimate",
    )
    _axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    _axis.set(
        xlabel="DCT-II mode m",
        ylabel=r"$\gamma_m\;[\mathrm{s}^{-1}]$",
        title="Early-time growth-rate dispersion from seeded multi-mode ensemble",
    )
    _axis.grid(True, alpha=0.3)
    if not _valid.empty:
        _axis.legend()
    _figure
    return


@app.cell
def _(mean_powers, mo):
    mode_selector = mo.ui.slider(
        start=0,
        stop=mean_powers.shape[1] - 1,
        step=1,
        value=0,
        label="DCT-II mode m",
        full_width=True,
        show_value=True,
    )
    mode_selector
    return (mode_selector,)


@app.cell(hide_code=True)
def _(amplitudes, mo, selected_ensemble, z):
    seed_labels = ", ".join(
        f"{run_id} (seed {seed})"
        for run_id, seed in zip(selected_ensemble["seed_ids"], selected_ensemble["seeds"], strict=True)
    )
    mo.md(
        f"## Selected ensemble\n\n"
        f"$\\nu={selected_ensemble['NU']:.6e}$, $\\mu={selected_ensemble['MU']:.6e}$  \n"
        f"Base: `{selected_ensemble['base_id']}`  \n"
        f"Seeds ({amplitudes.shape[0]}): {seed_labels}  \n"
        f"Perturbation {selected_ensemble['perturbation_name']}: `{selected_ensemble['perturbation']:.6g}`; "
        f"spatial points: `{z.size}`."
    )
    return


@app.cell
def _(mean_powers, mode_selector, plt, time):
    mode = int(mode_selector.value)
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.plot(time, mean_powers[:, mode], linewidth=1.8)
    axis.set_xlabel(r"$t\;[\mathrm{s}]$")
    axis.set_ylabel(rf"$P_{{{mode}}}(t)$")
    axis.set_title(rf"Seed-averaged DCT-II power, mode $m={mode}$")
    axis.grid(True, alpha=0.3)
    figure
    return


@app.cell
def _(growth_fits, mean_powers, mode_selector, np, plt, time):
    _mode = int(mode_selector.value)
    _initial_power = mean_powers[0, _mode]
    with np.errstate(divide="ignore", invalid="ignore"):
        _log_relative_power = np.log(
            mean_powers[:, _mode] / _initial_power
        )

    _figure, _axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    _axis.plot(time, _log_relative_power, linewidth=1.8)
    _fit = growth_fits[_mode]
    if _fit["success"]:
        _axis.plot(
            time,
            _fit["fitted"],
            color="#dc2626",
            linestyle="--",
            linewidth=2.0,
            label=rf"early-time fit: $\gamma={_fit['gamma']:.4e}\,\mathrm{{s}}^{{-1}}$",
        )
    _axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    _axis.set_xlabel(r"$t\;[\mathrm{s}]$")
    _axis.set_ylabel(rf"$\ln\!\\left(P_{{{_mode}}}(t) / P_{{{_mode}}}(0)\\right)$")
    _axis.set_title(rf"Log relative seed-averaged power, mode $m={_mode}$")
    _axis.grid(True, alpha=0.3)
    if _fit["success"]:
        _axis.legend()
    _figure
    return


if __name__ == "__main__":
    app.run()
