"""Adaptive early-prefix DCT diagnostics for paired Taylor sweeps."""

import numpy as np
import pandas as pd
from scipy.fft import dct, idct

from .dct_dispersion import validate_pair
from .paired_mode_growth import early_prefix


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
    if notes:
        raise ValueError("Cannot verify paired dynamics: " + " ".join(notes))
    field = np.asarray(perturbed["psi"], dtype=np.float64)
    base = np.asarray(reference["psi"], dtype=np.float64)
    n = field.shape[1]
    length = n * float(np.mean(np.diff(perturbed["z"])))
    scale = max(np.max(np.abs(field)), np.max(np.abs(base)))
    model = perturbed["model"]
    alpha, beta = float(model.get("alpha", np.nan)), float(model.get("beta", np.nan))
    mean_psi = float(np.mean(base[0]))
    # ν_c² = C (-μ), for the negative ν, μ branch and continuous k > 0.
    boundary_coefficient = np.nan
    if (model.get("modelType") in ("TAYL", b"TAYL")
            and model.get("gradientType") in ("LINEAR_FULL", b"LINEAR_FULL")
            and np.isfinite([alpha, beta, mean_psi]).all()
            and alpha > 0 and beta > 0 and mean_psi > 0):
        boundary_coefficient = 4 * alpha * (30 / length) / (beta * mean_psi)
    return dict(
        time=perturbed["time"].copy(),
        z=perturbed["z"].copy(),
        raw=dct(field, type=2, norm="ortho", axis=1),
        difference=dct(field - base, type=2, norm="ortho", axis=1),
        length=length, boundary_coefficient=boundary_coefficient,
        theory_mobility=beta * mean_psi,
        theory_nu=float(perturbed["variant"].get("NU", np.nan)),
        theory_mu=float(perturbed["variant"].get("MU", np.nan)),
        floor=float(100 * np.finfo(float).eps * np.sqrt(n) * scale),
        reference_drift=drift, notes=notes,
    )


def analyze_adaptive(prepared, mode_min, mode_max, *, min_samples=8, log_tolerance=0.05):
    """Fit signed paired-difference modes with independent initial prefixes.

    Use the same early_prefix fitter as the local ±epsilon notebook. Existing
    sweeps have a perturbed/unperturbed pair, so their response is one-sided.
    A constant epsilon normalization changes amplitudes but not log slopes.
    Mode zero is retained for inspection but excluded from growth summaries.
    """
    time = prepared["time"]
    n = prepared["raw"].shape[1]
    if (not isinstance(mode_min, (int, np.integer))
            or not isinstance(mode_max, (int, np.integer))
            or not 0 <= mode_min <= mode_max < n):
        raise ValueError("Select an inclusive integer DCT mode range within 0 … N−1.")
    modes = np.arange(mode_min, mode_max + 1)
    coefficients = prepared["difference"][:, modes]
    floor = prepared["floor"]
    rows, fits = [], []
    for index, mode in enumerate(modes):
        early = early_prefix(coefficients[:, index], time, floor=floor,
                             min_samples=min_samples, log_tolerance=log_tolerance)
        fit, count = early["fit"], early["count"]
        fits.append(fit)
        rows.append(dict(
            mode=int(mode),
            wavelength_cm=200 * prepared["length"] / mode if mode else np.nan,
            growth_s_inv=fit.rate if fit else np.nan,
            rate_std_s_inv=fit.rate_std if fit else np.nan,
            log_relative_rmse=fit.relative_rmse if fit else np.nan,
            r_squared=fit.r_squared if fit else np.nan,
            fit_samples=count,
            fit_start_s=float(time[0]) if fit else np.nan,
            fit_end_s=float(time[count - 1]) if fit else np.nan,
            max_abs_log_residual=float(np.max(np.abs(
                np.log(np.abs(coefficients[:count, index])) - fit.log_fitted
            ))) if fit else np.nan,
            status=fit.status if fit else early["reason"],
            window_reason=early["reason"],
        ))
    table = pd.DataFrame(rows)
    resolved = table[(table["status"] == "ok") & (table["mode"] > 0)
                     & np.isfinite(table["growth_s_inv"])]
    growing = resolved[resolved["growth_s_inv"] > 0]
    fastest = growing.loc[growing["growth_s_inv"].idxmax()] if len(growing) else None
    maximum = resolved.loc[resolved["growth_s_inv"].idxmax()] if len(resolved) else None
    unresolved_modes = int((table["mode"] > 0).sum() - len(resolved))
    growth_status = (
        "resolved growth" if fastest is not None else
        "no resolved modal rate" if maximum is None else
        "resolved modes decay; others unresolved" if unresolved_modes else
        "selected modes decay"
    )
    prediction = predict_selected_modes(
        prepared["theory_nu"], prepared["theory_mu"], prepared["length"],
        prepared["boundary_coefficient"], mode_min, mode_max,
        mobility=prepared["theory_mobility"],
    )
    table["theory_rate_s_inv"] = np.nan
    table.loc[table["mode"] > 0, "theory_rate_s_inv"] = prediction["rates"]
    summary = dict(
        saved_start_s=float(time[0]), saved_end_s=float(time[-1]), saved_samples=len(time),
        min_fit_samples=int(min_samples), log_tolerance=float(log_tolerance),
        mode_min=int(mode_min), mode_max=int(mode_max),
        fastest_mode=int(fastest["mode"]) if fastest is not None else np.nan,
        fastest_wavelength_cm=float(fastest["wavelength_cm"]) if fastest is not None else np.nan,
        fastest_growth_s_inv=float(fastest["growth_s_inv"]) if fastest is not None else np.nan,
        wavelength_status="resolved growth" if fastest is not None else "no resolved growing mode",
        max_growth_s_inv=float(maximum["growth_s_inv"]) if maximum is not None else np.nan,
        max_growth_mode=int(maximum["mode"]) if maximum is not None else np.nan,
        max_growth_fit_end_s=float(maximum["fit_end_s"]) if maximum is not None else np.nan,
        max_growth_fit_samples=int(maximum["fit_samples"]) if maximum is not None else 0,
        growth_status=growth_status,
        resolved_modes=len(resolved), unresolved_modes=unresolved_modes,
        reference_drift=prepared["reference_drift"],
        boundary_coefficient=prepared["boundary_coefficient"],
        dct_length_m=prepared["length"],
        theory_onset_nu=prediction["onset_nu"],
        theory_max_rate_s_inv=prediction["max_rate"],
        theory_growing_modes=prediction["growing_modes"],
        theory_fastest_mode=prediction["fastest_mode"],
        theory_fastest_wavelength_cm=prediction["fastest_wavelength_cm"],
    )
    return dict(summary=summary, table=table, fits=fits, time=time, modes=modes,
                coefficients=coefficients, floor=floor)


def predict_selected_modes(nu, mu, length, coefficient, mode_min, mode_max, *, mobility=1.0):
    """Evaluate the continuum dispersion on selected positive DCT wavenumbers.

    C = 4 alpha g / (beta mean_psi), so the reduced dispersion is
    -C/4 - nu*k² + mu*k⁴. The first growing mode appears as nu drops below
    max_m(mu*k_m² - C/(4*k_m²)). This samples the continuum model, rather
    than computing eigenvalues of the simulator's finite-difference operator.
    """
    modes = np.arange(max(1, mode_min), mode_max + 1)
    result = dict(rates=np.full(len(modes), np.nan), onset_nu=np.nan,
                  max_rate=np.nan, growing_modes=np.nan,
                  fastest_mode=np.nan, fastest_wavelength_cm=np.nan)
    if (not len(modes) or not np.isfinite([nu, mu, length, coefficient, mobility]).all()
            or length <= 0 or coefficient <= 0 or mobility <= 0 or mu >= 0):
        return result
    k2 = (np.pi * modes / length) ** 2
    reduced = -coefficient / 4 - nu * k2 + mu * k2**2
    precision = 64 * np.finfo(float).eps * (coefficient / 4 + abs(nu) * k2 + abs(mu) * k2**2)
    reduced[np.abs(reduced) <= precision] = 0
    rates = mobility * reduced
    fastest = int(np.argmax(rates))
    growing = int(np.count_nonzero(rates > 0))
    result.update(rates=rates, onset_nu=float(np.max(mu * k2 - coefficient / (4 * k2))),
                  max_rate=float(rates[fastest]), growing_modes=growing)
    if growing:
        result.update(fastest_mode=int(modes[fastest]),
                      fastest_wavelength_cm=float(200 * length / modes[fastest]))
    return result


def theory_overlays(phase_data, coordinates, mode_min, mode_max, wavelengths_cm):
    """Sample onset curves and continuum fastest-wavelength isolines in view.

    Wavelength contours use nu = 2 mu (2 pi / wavelength)². Retain only
    growing segments with k* inside the selected positive mode band. Physical
    configurations with different C or lengths remain separate curves.
    """
    columns = ["phase_x", "phase_y", "curve", "kind", "curve_id", "point_order",
               "wavelength_cm", "dct_length_m", "boundary_coefficient"]
    visible = phase_data[np.isfinite(phase_data[["x_low", "x_high", "y_low", "y_high"]]).all(axis=1)]
    if visible.empty:
        return pd.DataFrame(columns=columns)
    x_low, x_high = visible["x_low"].min(), visible["x_high"].max()
    y_low, y_high = visible["y_low"].min(), visible["y_high"].max()
    mu_low, mu_high = ((y_low * 1e-37, y_high * 1e-37) if coordinates == "raw"
                       else (-y_high * 1e-37, -y_low * 1e-37))
    if mu_low >= 0:
        return pd.DataFrame(columns=columns)
    mus = np.linspace(mu_low, min(mu_high, -np.finfo(float).tiny), 401)
    valid = visible[(visible["boundary_coefficient"] > 0) & (visible["dct_length_m"] > 0)]
    rows = []
    for group_index, ((coefficient, length), _) in enumerate(valid.groupby(["boundary_coefficient", "dct_length_m"])):
        critical = -np.sqrt(coefficient * -mus)
        modes = np.arange(max(1, mode_min), mode_max + 1)
        k2 = (np.pi * modes / length) ** 2
        discrete = (np.max(mus[:, None] * k2 - coefficient / (4 * k2), axis=1)
                    if len(modes) else None)
        curves = [("Continuum onset", "onset", critical, np.nan)]
        if discrete is not None:
            curves.append(("Selected-mode onset", "onset", discrete, np.nan))
        for wavelength in wavelengths_cm:
            k = 2 * np.pi / (wavelength / 100)
            if discrete is None or not k2[0] <= k**2 <= k2[-1]:
                continue
            nus = 2 * mus * k**2
            # A continuous optimum can lie inside the band while every sampled
            # mode still decays, particularly near onset for a narrow band.
            nus = np.where(nus < discrete, nus, np.nan)
            curves.append((f"ℓ* = {wavelength:g} cm", "wavelength", nus, float(wavelength)))
        for curve, kind, nus, wavelength in curves:
            xs = (nus / 1e-30 if coordinates == "raw" else (nus / 1e-30) ** 2
                  if coordinates == "squared" else nus / critical)
            ys = mus / 1e-37 if coordinates == "raw" else -mus / 1e-37
            # Preserve gaps as separate segments so no line bridges excluded
            # stable regions or excursions outside the displayed domain.
            keep = np.isfinite(xs) & (xs >= x_low) & (xs <= x_high)
            segment = np.cumsum(np.r_[True, keep[1:] != keep[:-1]])
            for index in np.flatnonzero(keep):
                rows.append(dict(
                    phase_x=float(xs[index]), phase_y=float(ys[index]),
                    curve=curve, kind=kind, curve_id=f"{group_index}:{curve}:{segment[index]}",
                    point_order=int(index), wavelength_cm=wavelength,
                    dct_length_m=float(length), boundary_coefficient=float(coefficient),
                ))
    return pd.DataFrame(rows, columns=columns)


def phase_coordinates(summary, coordinates):
    """Numeric heatmap coordinates and midpoint cell edges, without aggregation.

    Squaring is restricted to the negative branch to avoid folding positive ν
    onto negative ν. Normalization uses saved per-pair physical parameters.
    Cell edges are a visualization convention, not interpolation of measurements.
    """
    frame = summary.copy()
    nu, mu = frame["NU"], frame["MU"]
    coefficient = frame["boundary_coefficient"]
    frame["nu_critical"] = np.nan
    valid_theory = (mu < 0) & (coefficient > 0) & np.isfinite(coefficient)
    frame.loc[valid_theory, "nu_critical"] = -np.sqrt(
        coefficient[valid_theory] * -mu[valid_theory]
    )
    frame["critical_ratio"] = np.nan
    negative = (nu < 0) & (mu < 0)
    frame.loc[negative, "critical_ratio"] = nu[negative] / frame.loc[negative, "nu_critical"]
    if coordinates == "raw":
        frame["phase_x"], frame["phase_y"] = nu / 1e-30, mu / 1e-37
        titles = ("ν / 10⁻³⁰", "μ / 10⁻³⁷")
    elif coordinates == "squared":
        frame["phase_x"] = (nu / 1e-30) ** 2
        frame["phase_y"] = -mu / 1e-37
        frame.loc[~negative, ["phase_x", "phase_y"]] = np.nan
        titles = ("ν² / 10⁻⁶⁰", "−μ / 10⁻³⁷")
    elif coordinates == "normalized":
        frame["phase_x"], frame["phase_y"] = frame["critical_ratio"], -mu / 1e-37
        frame.loc[~negative, ["phase_x", "phase_y"]] = np.nan
        titles = ("|ν| / |ν_c(μ)|", "−μ / 10⁻³⁷")
    else:
        raise ValueError("Unknown phase coordinates.")

    def bounds(values):
        unique = np.unique(values)
        if len(unique) == 1:
            half = max(abs(unique[0]) * 0.025, 0.025)
            edges = np.array([unique[0] - half, unique[0] + half])
        else:
            mid = (unique[:-1] + unique[1:]) / 2
            edges = np.r_[unique[0] - (mid[0] - unique[0]),
                          mid, unique[-1] + (unique[-1] - mid[-1])]
        index = np.searchsorted(unique, values)
        return edges[index], edges[index + 1]

    valid = np.isfinite(frame[["phase_x", "phase_y"]]).all(axis=1)
    for column in ("x_low", "x_high", "y_low", "y_high"):
        frame[column] = np.nan
    if valid.any():
        frame.loc[valid, "y_low"], frame.loc[valid, "y_high"] = bounds(frame.loc[valid, "phase_y"])
        # ν values depend on μ in this sweep; compute x edges within each row.
        for _, group in frame[valid].groupby("phase_y"):
            # Round only for cell geometry: equal prescribed factors can differ
            # at floating point precision after normalization. IDs stay exact.
            values = group["phase_x"].to_numpy()
            if coordinates == "normalized":
                values = np.round(values, 12)
            low, high = bounds(values)
            frame.loc[group.index, "x_low"] = low
            frame.loc[group.index, "x_high"] = high
    return frame, titles


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
