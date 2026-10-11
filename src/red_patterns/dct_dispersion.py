"""Growth dispersion of independently evolving spatial DCT-II modes."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.fft import dct, idct

from red_patterns.linear_growth import fit_dct_mode


def read_run(path):
    """Keep saved float64 precision before subtracting small perturbations."""
    with h5py.File(path, "r") as handle:
        result = {name: np.asarray(handle[key], dtype=np.float64) for name, key in
                  (("psi", "fields/psi"), ("time", "time"), ("z", "coords/z"))}
        result["path"] = str(Path(path).resolve())
        result["rho"] = np.asarray(handle["coords/rho"]) if "coords/rho" in handle else None
        result["model"] = dict(handle["config/model"].attrs)
        result["run"] = dict(handle["config/run"].attrs)
        variant = "tayl" if "config/model/variant/tayl" in handle else "conv"
        result["variant"] = {
            key: value for key, value in handle[f"config/model/variant/{variant}"].attrs.items()
            if key != "kernelSource"
        }
        kernel_key = "config/model/variant/conv/kernel_values"
        result["kernel"] = np.asarray(handle[kernel_key]) if kernel_key in handle else None
    return result


def validate_pair(perturbed, equilibrium):
    """Reject incompatible runs; return diagnostics about the fixed reference."""
    for run, minimum in ((perturbed, 5), (equilibrium, 1)):
        t, z, psi = run["time"], run["z"], run["psi"]
        if (t.ndim != 1 or z.ndim != 1 or len(t) < minimum or len(z) < 2
                or psi.shape != (len(t), len(z))):
            raise ValueError("Need ψ(time, z), at least five perturbed frames and one equilibrium frame.")
        if (not all(np.all(np.isfinite(a)) for a in (t, z, psi))
                or np.any(np.diff(t) <= 0) or np.any(np.diff(z) <= 0)):
            raise ValueError("All fields must be finite; time and z must increase strictly.")
    if not np.array_equal(perturbed["z"], equilibrium["z"]):
        raise ValueError("The runs must use identical z grids.")
    spacing = np.diff(perturbed["z"])
    if not np.allclose(spacing, np.mean(spacing), rtol=1e-7, atol=1e-14):
        raise ValueError("DCT-II wavelengths require uniformly spaced z samples.")
    notes, differences = [], []
    for group, keys in (("model", ("modelType", "gradientType", "alpha", "beta")),
                        ("variant", set(perturbed["variant"]) | set(equilibrium["variant"]))):
        for key in sorted(keys):
            if key not in perturbed[group] or key not in equilibrium[group]:
                notes.append(f"Missing {key} metadata; check that both runs have the same dynamics.")
            elif not np.array_equal(perturbed[group][key], equilibrium[group][key]):
                differences.append(key)
    model_type = perturbed["model"].get("modelType")
    if model_type in ("CONV", b"CONV"):
        if perturbed["kernel"] is None or equilibrium["kernel"] is None:
            notes.append("Saved convolution kernel unavailable; kernel equality could not be verified.")
        elif not np.array_equal(perturbed["kernel"], equilibrium["kernel"]):
            differences.append("convolution kernel")
    if differences:
        raise ValueError("Runs use different dynamics: " + ", ".join(differences))
    if perturbed["rho"] is not None and equilibrium["rho"] is not None:
        if not np.array_equal(perturbed["rho"], equilibrium["rho"]):
            raise ValueError("The runs must use identical rho grids.")
    else:
        notes.append("Missing rho grid; equality of the full spatial grid could not be verified.")
    if perturbed["run"].get("DT") != equilibrium["run"].get("DT"):
        notes.append("Solver DT differs between the runs.")
    eq = equilibrium["psi"]
    drift = float(np.max(np.linalg.norm(eq - eq[0], axis=1))
                  / max(np.linalg.norm(eq[0]), np.finfo(float).tiny))
    if drift > 1e-6:
        notes.append(f"Equilibrium ψ drift is {drift:.3e} relative to its first profile. "
                     "Verify stationarity; subtraction holds that first profile fixed.")
    return drift, notes


def analyze_modes(perturbed, equilibrium, end_index, mode_min, mode_max):
    """Fit signed coefficients on original times, including the selected endpoint."""
    validate_pair(perturbed, equilibrium)
    n = len(perturbed["z"])
    if not 4 <= end_index < len(perturbed["time"]):
        raise ValueError("The selected interval must contain at least five saved frames.")
    if not 0 <= mode_min <= mode_max < n:
        raise ValueError("Select an inclusive DCT mode range within 0 … N−1.")
    time = perturbed["time"][:end_index + 1]
    field = perturbed["psi"][:end_index + 1]
    reference = equilibrium["psi"][0]
    coefficients = dct(field - reference, type=2, norm="ortho", axis=1)
    floor = (100 * np.finfo(float).eps * np.sqrt(n)
             * max(float(np.max(np.abs(field))), float(np.max(np.abs(reference)))))
    modes = np.arange(mode_min, mode_max + 1)
    fits = [fit_dct_mode(coefficients[:, m], time, amplitude_floor=floor) for m in modes]
    length = n * float(np.mean(np.diff(perturbed["z"])))
    wavenumber = np.pi * modes / length
    wavelength = np.full(len(modes), np.inf)
    np.divide(2 * length, modes, out=wavelength, where=modes > 0)
    table = pd.DataFrame({
        "mode": modes, "wavenumber_rad_m": wavenumber, "wavelength_m": wavelength,
        "growth_rate_s^-1": [fit.rate for fit in fits],
        "rate_std_s^-1": [fit.rate_std for fit in fits],
        "relative_fit_error": [fit.relative_rmse for fit in fits],
        "R_squared": [fit.r_squared for fit in fits],
        "max_abs_coefficient": [fit.max_abs_coefficient for fit in fits],
        "status": [fit.status for fit in fits],
    })
    return time, coefficients, fits, table, length, floor


def dispersion_features(table):
    """Use resolved nonzero modes; interpolate only adjacent resolved brackets.

    The maximum is a measured discrete mode, never a fitted continuous maximum.
    Every resolved zero crossing is retained, including long-wave band edges.
    A + to − crossing with increasing k is a short-wave cutoff.
    """
    ordered = table.sort_values("mode").reset_index(drop=True)
    valid = ((ordered["status"] == "ok") & (ordered["mode"] > 0)
             & np.isfinite(ordered["growth_rate_s^-1"]))
    growing = ordered[valid & (ordered["growth_rate_s^-1"] > 0)]
    fastest = None if growing.empty else growing.loc[growing["growth_rate_s^-1"].idxmax()].to_dict()
    crossings = []
    for i in range(len(ordered) - 1):
        left, right = ordered.iloc[i], ordered.iloc[i + 1]
        if not (valid.iloc[i] and valid.iloc[i + 1] and right["mode"] == left["mode"] + 1):
            continue
        g0, g1 = left["growth_rate_s^-1"], right["growth_rate_s^-1"]
        if g0 * g1 >= 0:
            continue
        fraction = -g0 / (g1 - g0)
        kc = float(left["wavenumber_rad_m"] + fraction
                   * (right["wavenumber_rad_m"] - left["wavenumber_rad_m"]))
        crossings.append({
            "kind": "short-wave cutoff" if g0 > 0 else "long-wave band edge",
            "mode_left": int(left["mode"]), "mode_right": int(right["mode"]),
            "wavenumber_rad_m": kc, "wavelength_m": 2 * np.pi / kc,
            "wavelength_bracket_min_m": float(right["wavelength_m"]),
            "wavelength_bracket_max_m": float(left["wavelength_m"]),
        })
    return fastest, pd.DataFrame(crossings, columns=[
        "kind", "mode_left", "mode_right", "wavenumber_rad_m", "wavelength_m",
        "wavelength_bracket_min_m", "wavelength_bracket_max_m",
    ])


def demo_pair():
    """Independent cosine eigenmodes: known maximum m=5, cutoff between 7 and 8."""
    time = np.linspace(0, 18, 121)
    n, dz = 64, 1e-3
    z = np.arange(n) * dz
    modes = np.arange(n)
    rates = 0.16 - 0.02 * (modes - 5) ** 2
    coefficients = np.zeros((len(time), n))
    excited = np.arange(1, 13)
    # Integrated exponential observable: offsets allow initially zero δψ.
    coefficients[:, excited] = (2e-5 * (-1.0) ** excited
                                * np.expm1(time[:, None] * rates[excited]))
    equilibrium = np.full((len(time), n), 0.02)
    common = {"time": time, "z": z, "rho": np.arange(n),
              "model": {"modelType": "TAYL", "gradientType": "LINEAR_FULL",
                        "alpha": 1.0, "beta": 1.0},
              "variant": {"NU": 1.0, "MU": 1.0}, "run": {"DT": 0.001}, "kernel": None}
    return (
        dict(common, path="Synthetic perturbed run", psi=equilibrium + idct(coefficients, type=2, norm="ortho", axis=1)),
        dict(common, path="Synthetic equilibrium run", psi=equilibrium),
    )
