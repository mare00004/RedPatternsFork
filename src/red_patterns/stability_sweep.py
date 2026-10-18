"""Pair and analyze constant-ridge Taylor stability sweeps."""

import json

import numpy as np

from .dct_dispersion import analyze_modes, dispersion_features, validate_pair
from .models import TaylorRun
from .types import Gradient, PhiType


def ridge_pairs(catalog):
    """Match all shared settings exactly, excluding displacement parameters.

    Never use default float tolerances for NU/MU: their magnitudes are tiny.
    Missing and ambiguous references remain visible in the returned rows.
    """
    def key(run):
        payload = run.model_dump(mode="json")
        payload.pop("run_id")
        for name in ("phi_type", "epsilon", "seed", "mode_min", "mode_max"):
            payload["phi"]["params"].pop(name, None)
        return json.dumps(payload, sort_keys=True)

    entries = [entry for entry in catalog.entries
               if isinstance(entry.run, TaylorRun)
               and entry.run.gradient == Gradient.LINEAR_FULL]
    references = {}
    for entry in entries:
        if entry.run.phi.params.phi_type == PhiType.LINEAR_FULL_RIDGE:
            references.setdefault(key(entry.run), []).append(entry)
    pairs = []
    for entry in entries:
        run = entry.run
        if run.phi.params.phi_type != PhiType.PERTURBED_LINEAR_FULL_RIDGE:
            continue
        matches = references.get(key(run), [])
        reference = matches[0] if len(matches) == 1 else None
        status = "ready"
        if not matches:
            status = "missing reference definition"
        elif len(matches) > 1:
            status = "ambiguous reference definitions"
        elif not entry.h5_exists or not reference.h5_exists:
            status = "missing result file"
        pairs.append(dict(
            run_id=run.run_id, reference_id=reference.run_id if reference else "",
            NU=run.NU, MU=run.MU, psi_avg=run.phi.params.psi_avg,
            epsilon=run.phi.params.epsilon, seed=run.phi.params.seed,
            mode_min=run.phi.params.mode_min, mode_max=run.phi.params.mode_max,
            perturbed_path=str(entry.run_h5),
            equilibrium_path=str(reference.run_h5) if reference else "",
            status=status,
        ))
    return pairs


def analyze_pair(perturbed, equilibrium, end_time, mode_min, mode_max):
    """Analyze one inclusive time window; use saved metadata for theory."""
    drift, notes = validate_pair(perturbed, equilibrium)
    end_index = int(np.searchsorted(perturbed["time"], end_time, side="right") - 1)
    time, coefficients, fits, table, length, floor = analyze_modes(
        perturbed, equilibrium, end_index, mode_min, mode_max,
    )
    fastest, crossings = dispersion_features(table)
    model, variant = perturbed["model"], perturbed["variant"]
    if (model.get("modelType") not in ("TAYL", b"TAYL")
            or model.get("gradientType") not in ("LINEAR_FULL", b"LINEAR_FULL")):
        raise ValueError("This browser requires Taylor LINEAR_FULL runs.")
    alpha, beta = float(model["alpha"]), float(model["beta"])
    nu, mu = float(variant["NU"]), float(variant["MU"])
    avg = float(np.mean(equilibrium["psi"][0]))
    if not all(np.isfinite([alpha, beta, nu, mu, avg])) or beta <= 0 or avg <= 0:
        raise ValueError("Theory requires finite metadata and positive beta and mean psi.")
    g = 30 / length
    k = table["wavenumber_rad_m"].to_numpy()
    table["theory_rate_s^-1"] = -alpha * g - beta * avg * nu * k**2 + beta * avg * mu * k**4
    critical = -np.sqrt(-4 * alpha * g * mu / (beta * avg)) if mu < 0 and alpha >= 0 else np.nan
    resolved = table[table["status"] == "ok"]
    measured_max = float(resolved["growth_rate_s^-1"].max()) if len(resolved) else np.nan
    if drift > 1e-6 or notes:
        status = "check reference / metadata"
    elif measured_max > 0:
        status = "resolved growth"
    elif len(resolved) == len(table):
        status = "selected modes decay"
    else:
        status = "inconclusive"
    return dict(
        time=time, coefficients=coefficients, fits=fits, table=table,
        length=length, floor=floor, fastest=fastest, crossings=crossings,
        notes=notes, summary=dict(
            status=status, max_growth_s_inv=measured_max,
            resolved_modes=len(resolved), selected_modes=len(table),
            equilibrium_drift=drift, fit_end_s=float(time[-1]),
            nu_critical=critical, NU=nu, MU=mu, psi_avg=avg,
            alpha=alpha, beta=beta, g=g,
            fastest_mode=fastest["mode"] if fastest else np.nan,
            fastest_wavelength_cm=100 * fastest["wavelength_m"] if fastest else np.nan,
        ),
    )
