"""Shared-window DCT diagnostics for paired Taylor sweeps."""

import numpy as np
import pandas as pd
from scipy.fft import dct, idct

from .dct_dispersion import fit_log_mode, validate_pair


def prepare_pair(perturbed, reference):
    """Validate a pair and cache spatial transforms, preserving saved precision.

    Subtract fields at identical saved times, never a fixed initial profile.
    Time interpolation is deliberately excluded from this measurement.
    """
    drift, notes = validate_pair(perturbed, reference)
    if not np.array_equal(perturbed["time"], reference["time"]):
        raise ValueError("Paired runs must have identical saved-time grids; no interpolation is used.")
    # validate_pair also describes its own fixed-reference analysis. That
    # stationarity warning does not apply to the time-dependent subtraction here.
    notes = [note for note in notes if not note.startswith("Equilibrium ψ drift")]
    field, base = perturbed["psi"], reference["psi"]
    n = field.shape[1]
    scale = np.maximum(np.max(np.abs(field), axis=1), np.max(np.abs(base), axis=1))
    return dict(
        time=perturbed["time"].copy(),
        raw=dct(field, type=2, norm="ortho", axis=1),
        difference=dct(field - base, type=2, norm="ortho", axis=1),
        length=n * float(np.mean(np.diff(perturbed["z"]))),
        floor=100 * np.finfo(float).eps * np.sqrt(n) * scale,
        reference_drift=drift, notes=notes,
    )


def analyze_window(prepared, start, end, mode_min, mode_max):
    """Raw perturbed modal fits and the endpoint rate of the paired band norm.

    Both endpoints must be actual saved times. At least two frames define the
    norm rate; modal log fits require five frames. Zero endpoints have no
    finite logarithmic rate and are never replaced with epsilon.
    """
    all_time = prepared["time"]
    if not np.isfinite([start, end]).all() or start >= end:
        raise ValueError("Select a time window with start < end.")
    if start not in all_time or end not in all_time:
        raise ValueError("Window endpoints must be saved times present in every loaded pair.")
    n = prepared["raw"].shape[1]
    if (not isinstance(mode_min, (int, np.integer))
            or not isinstance(mode_max, (int, np.integer))
            or not 0 <= mode_min <= mode_max < n):
        raise ValueError("Select an inclusive integer DCT mode range within 0 … N−1.")
    take = (all_time >= start) & (all_time <= end)
    time = all_time[take]
    modes = np.arange(mode_min, mode_max + 1)
    raw = prepared["raw"][take][:, modes]
    difference = prepared["difference"][take][:, modes]
    floor = float(np.max(prepared["floor"][take]))
    rows, fits = [], []
    for index, mode in enumerate(modes):
        fit = fit_log_mode(raw[:, index], time, amplitude_floor=floor) if len(time) >= 5 else None
        fits.append(fit)
        rows.append(dict(
            mode=int(mode),
            wavelength_cm=200 * prepared["length"] / mode if mode else np.nan,
            growth_s_inv=fit.rate if fit else np.nan,
            rate_std_s_inv=fit.rate_std if fit else np.nan,
            log_relative_rmse=fit.relative_rmse if fit else np.nan,
            status=fit.status if fit else "fewer than five saved frames",
        ))
    table = pd.DataFrame(rows)
    growing = table[(table["status"] == "ok") & (table["growth_s_inv"] > 0) & (table["mode"] > 0)]
    fastest = growing.loc[growing["growth_s_inv"].idxmax()] if len(growing) else None
    # hypot.reduce computes sqrt(sum(abs(c_m)**2)) without squaring overflow.
    distance = np.hypot.reduce(np.abs(difference), axis=1)
    norm_floor = np.sqrt(len(modes)) * prepared["floor"][take]
    endpoint_distance = distance[[0, -1]]
    endpoint_floor = norm_floor[[0, -1]]
    rate = np.nan
    if not np.isfinite(endpoint_distance).all():
        norm_status = "non-finite endpoint norm"
    elif np.any(endpoint_distance == 0):
        norm_status = "zero endpoint norm; logarithmic rate undefined"
    elif np.any(endpoint_distance <= endpoint_floor):
        norm_status = "endpoint norm below numerical precision"
    else:
        rate = float((np.log(distance[-1]) - np.log(distance[0])) / (time[-1] - time[0]))
        norm_status = "net growth" if rate > 0 else "net decay" if rate < 0 else "neutral"
    summary = dict(
        start_s=float(time[0]), end_s=float(time[-1]), samples=len(time),
        mode_min=int(mode_min), mode_max=int(mode_max),
        fastest_mode=int(fastest["mode"]) if fastest is not None else np.nan,
        fastest_wavelength_cm=float(fastest["wavelength_cm"]) if fastest is not None else np.nan,
        fastest_growth_s_inv=float(fastest["growth_s_inv"]) if fastest is not None else np.nan,
        wavelength_status="resolved growth" if fastest is not None else "no resolved growing mode",
        resolved_modes=int((table["status"] == "ok").sum()),
        D_start=float(distance[0]), D_end=float(distance[-1]),
        lambda_eff_s_inv=rate, norm_status=norm_status,
        reference_drift=prepared["reference_drift"],
    )
    return dict(summary=summary, table=table, fits=fits, time=time, modes=modes,
                raw=raw, difference=difference, distance=distance, norm_floor=norm_floor)


def demo_sweep():
    """Small analytic sweep with directly excited growing and decaying cosines."""
    pairs, prepared = [], {}
    n, length = 256, 0.07
    time = np.linspace(0, 30, 301)
    modes = np.arange(n)
    k = np.pi * modes / length
    for mu in [-3.6e-37, -4.5e-37, -5.4e-37]:
        critical = np.sqrt(-4 * 2e-5 * (30 / length) * mu / (7.4e23 * 0.03))
        for factor in [0.8, 1.0, 1.2, 1.5]:
            nu = -factor * critical
            rate = -2e-5 * 30 / length - 7.4e23 * 0.03 * nu * k**2 + 7.4e23 * 0.03 * mu * k**4
            coeff = np.zeros((len(time), n))
            coeff[:, 1:101] = 2e-7 * np.exp(time[:, None] * rate[None, 1:101])
            common = dict(
                time=time, z=np.arange(n) * length / n, rho=np.arange(n),
                model=dict(modelType="TAYL", gradientType="LINEAR_FULL", alpha=2e-5, beta=7.4e23),
                variant=dict(NU=nu, MU=mu), run=dict(DT=0.005), kernel=None,
            )
            base = dict(common, psi=np.full((len(time), n), 0.03))
            perturbed = dict(common, psi=base["psi"] + idct(coeff, type=2, norm="ortho", axis=1))
            run_id = f"demo-{len(pairs) + 1}"
            pairs.append(dict(run_id=run_id, reference_id=f"{run_id}-base", NU=nu, MU=mu,
                              psi_avg=0.03, seed=0, epsilon=1e-5, mode_min=1, mode_max=100,
                              status="ready"))
            prepared[run_id] = prepare_pair(perturbed, base)
    return pairs, prepared
