"""Central-difference cosine response and conservative early-prefix fits."""

import numpy as np
from scipy.fft import dct

from .dct_dispersion import fit_log_mode, validate_pair


def paired_response(positive, negative, epsilon):
    """Subtract matching saved frames, preserving float64 before subtraction."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive.")
    # Reuse operator/grid validation, but discard fixed-equilibrium diagnostics:
    # neither member of this pair is an unperturbed equilibrium reference.
    _, notes = validate_pair(positive, negative)
    incomplete = [note for note in notes if note.startswith(("Missing ", "Saved convolution"))]
    if incomplete:
        raise ValueError("Cannot verify paired dynamics: " + " ".join(incomplete))
    if not np.array_equal(positive["time"], negative["time"]):
        raise ValueError("The ±epsilon runs must have identical saved times; no interpolation is applied.")
    if positive["run"].get("DT") != negative["run"].get("DT"):
        raise ValueError("The ±epsilon runs must use identical solver DT.")
    for run in (positive, negative):
        if run["rho"] is None:
            raise ValueError("Missing rho grid; cannot verify the paired experiment.")
    delta = (np.asarray(positive["psi"], dtype=np.float64)
             - np.asarray(negative["psi"], dtype=np.float64)) / (2 * epsilon)
    coefficients = dct(delta, type=2, norm="ortho", axis=1)
    # Conservative floating-point cancellation scale, not a physical noise model.
    scale = max(np.max(np.abs(positive["psi"])), np.max(np.abs(negative["psi"])))
    floor = 100 * np.finfo(float).eps * np.sqrt(delta.shape[1]) * scale / epsilon
    return delta, coefficients, float(floor)


def early_prefix(values, time, *, floor=0., min_samples=8, log_tolerance=0.05):
    """Extend the initial contiguous prefix until its first log-linear failure.

    Every prefix must remain above the floor, retain one sign and have maximum
    absolute log residual <= log_tolerance. No late-window search or saturation
    model is used. A successful prefix is a diagnostic, not proof of linearity.
    """
    values, time = np.asarray(values, dtype=float), np.asarray(time, dtype=float)
    if min_samples < 5 or log_tolerance <= 0 or not np.isfinite(log_tolerance):
        raise ValueError("Need at least five samples and a positive finite log tolerance.")
    if values.ndim != 1 or values.shape != time.shape or len(time) < min_samples:
        return {"fit": None, "count": 0, "reason": f"Need at least {min_samples} matching saved samples."}
    best = None
    count = 0
    reason = "All saved frames pass; no departure from log-linearity was detected."
    for end in range(min_samples, len(time) + 1):
        v = values[:end]
        if np.any(np.abs(v) <= floor) or np.any(np.sign(v) != np.sign(v[0])):
            reason = "Stopped at a noise-floor sample or a sign change."
            break
        fit = fit_log_mode(v, time[:end], amplitude_floor=floor)
        if not np.all(np.isfinite(fit.log_fitted)):
            reason = "Initial prefix is constant or unresolved."
            break
        residual = np.max(np.abs(np.log(np.abs(v)) - fit.log_fitted))
        if residual > log_tolerance:
            reason = f"Next prefix exceeds max |log residual| = {log_tolerance:g}."
            break
        best, count = fit, end
    return {"fit": best, "count": count, "reason": reason}
