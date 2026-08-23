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
    # Seed-ensemble DCT-II mode analysis

    Choose a Taylor sweep directory containing `runs.jsonl` and
    `results/<run_id>/run.h5`. For each selected $(\nu, \mu)$ pair, the notebook
    subtracts its smooth-homogeneous base run from every seeded perturbed run,
    computes a spatial orthonormal DCT-II, and averages squared mode magnitudes
    over seeds.
    """)
    return


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
def _(Path, TaylorRun, load_runs_jsonl, pd):
    def scan_sweep(sweep_dir: Path) -> pd.DataFrame:
        """Load candidate Taylor base and perturbed runs plus their result paths."""
        rows: list[dict[str, object]] = []
        for run in load_runs_jsonl(sweep_dir / "runs.jsonl"):
            if not isinstance(run, TaylorRun):
                continue

            phi_params = run.phi.params.model_dump(mode="json")
            phi_type = str(phi_params.pop("phi_type"))
            if phi_type not in {
                PhiType.SMOOTH_HOMOGENEOUS.value,
                PhiType.PERTURBED_SMOOTH_HOMOGENEOUS.value,
            }:
                continue

            seed = phi_params.pop("seed", None)
            amplitude = phi_params.pop("amplitude", None)
            shared_phi = json.dumps(phi_params, sort_keys=True, separators=(",", ":"))
            result_path = sweep_dir / "results" / run.run_id / "run.h5"
            rows.append(
                {
                    "run_id": run.run_id,
                    "NU": float(run.NU),
                    "MU": float(run.MU),
                    "phi_type": phi_type,
                    "seed": None if seed is None else int(seed),
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

    def validate_ensembles(sweep_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return valid pair candidates and diagnostics for every discovered pair."""
        columns = ["NU", "MU", "base_id", "seed_ids", "seeds", "amplitude"]
        diagnostic_columns = ["NU", "MU", "status", "details"]
        if sweep_df.empty:
            return pd.DataFrame(columns=columns), pd.DataFrame(columns=diagnostic_columns)

        candidates: list[dict[str, object]] = []
        diagnostics: list[dict[str, object]] = []
        setup_columns = ["shared_phi", "N", "T", "DT", "storeTime", "gradient"]
        for (nu, mu), pair_df in sweep_df.groupby(["NU", "MU"], sort=True):
            base_df = pair_df[pair_df["phi_type"] == PhiType.SMOOTH_HOMOGENEOUS.value]
            seed_df = pair_df[
                pair_df["phi_type"] == PhiType.PERTURBED_SMOOTH_HOMOGENEOUS.value
            ]
            if len(base_df) != 1:
                diagnostics.append(
                    {
                        "NU": nu,
                        "MU": mu,
                        "status": "invalid",
                        "details": f"Expected exactly one smooth base run; found {len(base_df)}.",
                    }
                )
                continue
            if seed_df.empty:
                diagnostics.append(
                    {"NU": nu, "MU": mu, "status": "invalid", "details": "No perturbed seed runs found."}
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
                        "details": "Seed setup differs from base: " + ", ".join(mismatched["run_id"]),
                    }
                )
                continue
            if seed_df["amplitude"].nunique(dropna=False) != 1:
                diagnostics.append(
                    {
                        "NU": nu,
                        "MU": mu,
                        "status": "invalid",
                        "details": "Perturbation amplitudes differ across seeds.",
                    }
                )
                continue
            if seed_df["seed"].isna().any() or seed_df["seed"].duplicated().any():
                diagnostics.append(
                    {
                        "NU": nu,
                        "MU": mu,
                        "status": "invalid",
                        "details": "Seeds must be present and unique.",
                    }
                )
                continue

            missing = pair_df[~pair_df["h5_exists"]]["run_id"].tolist()
            if missing:
                diagnostics.append(
                    {
                        "NU": nu,
                        "MU": mu,
                        "status": "incomplete",
                        "details": "Missing run.h5 for: " + ", ".join(missing),
                    }
                )
                continue

            sorted_seeds = seed_df.sort_values("seed", kind="stable")
            candidates.append(
                {
                    "NU": float(nu),
                    "MU": float(mu),
                    "base_id": str(base_row["run_id"]),
                    "seed_ids": tuple(sorted_seeds["run_id"].tolist()),
                    "seeds": tuple(int(seed) for seed in sorted_seeds["seed"].tolist()),
                    "amplitude": float(sorted_seeds["amplitude"].iloc[0]),
                }
            )
            diagnostics.append({"NU": nu, "MU": mu, "status": "ready", "details": "Compatible ensemble."})

        return pd.DataFrame(candidates, columns=columns), pd.DataFrame(diagnostics, columns=diagnostic_columns)

    return scan_sweep, validate_ensembles


@app.cell
def _(Path, mo, scan_sweep, ui_sweep_dir, validate_ensembles):
    selected_path = ui_sweep_dir.path(0) if ui_sweep_dir.value else None
    default_dir = REPO_ROOT / "data"
    sweep_dir = Path(selected_path) if selected_path else default_dir

    if not (sweep_dir / "runs.jsonl").is_file():
        sweep_df = pd.DataFrame()
        ensemble_df = pd.DataFrame()
        diagnostics_df = pd.DataFrame()
        scan_status = mo.callout(
            f"`{sweep_dir}` does not contain `runs.jsonl`. Choose a sweep directory.",
            kind="warn",
        )
    else:
        try:
            sweep_df = scan_sweep(sweep_dir)
        except ValueError as exc:
            sweep_df = pd.DataFrame()
            ensemble_df = pd.DataFrame()
            diagnostics_df = pd.DataFrame()
            scan_status = mo.callout(
                f"Could not read `{sweep_dir / 'runs.jsonl'}` with the current sweep schema: {exc}",
                kind="warn",
            )
        else:
            ensemble_df, diagnostics_df = validate_ensembles(sweep_df)
            scan_status = mo.md(
                f"Found `{len(sweep_df)}` smooth/perturbed Taylor runs and "
                f"`{len(ensemble_df)}` compatible ensembles in `{sweep_dir}`."
            )

    scan_status
    return diagnostics_df, ensemble_df, sweep_dir


@app.cell
def _(diagnostics_df, mo):
    mo.stop(diagnostics_df.empty, mo.md("No smooth-homogeneous candidate runs found."))
    mo.ui.table(data=diagnostics_df, selection=None, pagination=True)
    return


@app.cell
def _(ensemble_df, mo):
    mo.stop(ensemble_df.empty, mo.md("No complete, compatible ensembles are available yet."))
    options = {
        f"ν={row.NU:.6e}, μ={row.MU:.6e}": index
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
        f"Perturbation amplitude: `{selected_ensemble['amplitude']:.6g}`; "
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
def _(mean_powers, mode_selector, np, plt, time):
    _mode = int(mode_selector.value)
    _initial_power = mean_powers[0, _mode]
    with np.errstate(divide="ignore", invalid="ignore"):
        _log_relative_power = np.log(
            mean_powers[:, _mode] / _initial_power
        )

    _figure, _axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    _axis.plot(time, _log_relative_power, linewidth=1.8)
    _axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    _axis.set_xlabel(r"$t\;[\mathrm{s}]$")
    _axis.set_ylabel(rf"$\ln\!\\left(P_{{{_mode}}}(t) / P_{{{_mode}}}(0)\\right)$")
    _axis.set_title(rf"Log relative seed-averaged power, mode $m={_mode}$")
    _axis.grid(True, alpha=0.3)
    _figure
    return


if __name__ == "__main__":
    app.run()
