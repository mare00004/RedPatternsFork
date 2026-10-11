"""Individual DCT-mode fits and block-Hankel estimates of growth rates.

Observations have shape (saved time, spatial DCT mode). No centering or mode
removal is performed. Irregular observations can be interpolated onto a uniform
grid before the Hankel SVD. Rates belong to the observed signal;
identifying them with eigenvalues of a phi linearization needs a suitable
equilibrium reference and observable perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd
from scipy.fft import idct
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class HankelSubspace:
    left_vectors: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    shape: tuple[int, int]
    delays: int
    n_modes: int
    numerical_rank: int


@dataclass(frozen=True)
class GrowthEstimate:
    discrete_eigenvalues: NDArray[np.complex128]
    exponents: NDArray[np.complex128]
    spatial_amplitudes: NDArray[np.complex128]
    reconstruction: NDArray[np.complex128]
    retained_energy: float
    shift_residual: float
    shift_condition: float
    reconstruction_error: float


@dataclass(frozen=True)
class DynamicModes:
    """Observable mode shapes, with b[:, q] = dct_shapes[:, q] * amplitudes[q].

    Shapes have unit Euclidean norm and their largest spatial entry is real
    and positive. This fixes an arbitrary scale/phase, retaining the excitation
    in amplitudes. Zero padding outside the observed DCT range reconstructs
    only the selected-band projection of a psi mode, not a full phi eigenvector.
    conjugate_partners uses zero-based component indices, or -1 for no partner.
    """

    dct_shapes: NDArray[np.complex128]
    psi_shapes: NDArray[np.complex128]
    amplitudes: NDArray[np.complex128]
    conjugate_partners: NDArray[np.int64]


def recover_dynamic_modes(
    growth: GrowthEstimate,
    selected_modes: NDArray[np.int64],
    n_spatial_points: int,
) -> DynamicModes:
    """Lift fitted ESPRIT residues to normalized, complex spatial psi shapes.

    The exponents come from shift invariance; least squares at those exponents
    supplies b. Distinct, resolved poles can describe observable eigenmode
    projections in a linear autonomous regime. Repeated poles only identify
    their combined excited projection in a single trajectory.
    """
    modes = np.asarray(selected_modes)
    b = np.asarray(growth.spatial_amplitudes, dtype=np.complex128)
    if (
        not isinstance(n_spatial_points, (int, np.integer)) or n_spatial_points < 1
        or modes.ndim != 1 or not np.issubdtype(modes.dtype, np.integer)
        or modes.size == 0 or np.unique(modes).size != modes.size
        or np.any(modes < 0) or np.any(modes >= n_spatial_points)
        or b.shape != (len(modes), len(growth.exponents))
        or not np.all(np.isfinite(b))
    ):
        raise ValueError("Mode indices and finite fitted amplitudes must match the full spatial grid.")
    # Stable unit normalization before fixing the spatial phase.
    peaks = np.max(np.abs(b), axis=0)
    scaled = np.divide(b, peaks, out=np.zeros_like(b), where=peaks > 0)
    norms = np.linalg.norm(scaled, axis=0)
    shapes = np.divide(scaled, norms, out=np.zeros_like(b), where=norms > 0)
    full_dct = np.zeros((n_spatial_points, b.shape[1]), dtype=np.complex128)
    full_dct[modes] = shapes
    spatial = idct(full_dct, type=2, norm="ortho", axis=0)
    pivots = spatial[np.argmax(np.abs(spatial), axis=0), np.arange(b.shape[1])]
    phase = np.exp(1j * np.angle(pivots))
    shapes = shapes / phase
    spatial = spatial / phase
    amplitudes = peaks * norms * phase
    partners = np.full(b.shape[1], -1, dtype=np.int64)
    poles = growth.discrete_eigenvalues
    for q, pole in enumerate(poles):
        if partners[q] >= 0 or abs(pole.imag) <= 1e-10 * max(1.0, abs(pole)):
            continue
        candidates = np.flatnonzero((partners < 0) & (np.arange(len(poles)) != q))
        if candidates.size:
            p = candidates[np.argmin(np.abs(poles[candidates] - pole.conjugate()))]
            if abs(poles[p] - pole.conjugate()) <= 1e-7 * max(1.0, abs(pole)):
                partners[q], partners[p] = p, q
    return DynamicModes(shapes, spatial, amplitudes, partners)


@dataclass(frozen=True)
class UniformObservations:
    time: NDArray[np.float64]
    values: NDArray[np.float64]
    resampled: bool
    max_relative_step_deviation: float


@dataclass(frozen=True)
class DCTModeFit:
    rate: float
    rate_std: float
    r_squared: float
    relative_rmse: float
    fitted: NDArray[np.float64]
    max_abs_coefficient: float
    status: str


def fit_dct_mode(
    values: NDArray[np.float64],
    time: NDArray[np.float64],
    *,
    amplitude_floor: float = 0.0,
) -> DCTModeFit:
    """Fit one signed coefficient to A exp(gamma*(t-t0)) + C.

    Use the original physical times, including irregular saves. Variable
    projection solves the amplitude/offset linearly for each candidate rate,
    then minimizes a one-dimensional residual. Normalize time and coefficient
    variation to preserve small perturbations and condition the optimization.
    The equivalent endpoint parameterization is c0 + change*expm1(k*x)/expm1(k)
    where x=(t-t0)/(t_end-t0), k=gamma*(t_end-t0). Its k=0 limit is linear;
    a linear signal does not establish an exponential growth rate.

    Search |k| <= 50. Weak/constant signals have no reported rate; poor fits,
    unresolved curvature, and search-limit fits are explicitly marked. Standard
    errors assume uncorrelated residuals and are fit diagnostics, not a physical
    confidence guarantee.
    """
    values, time = np.asarray(values, dtype=np.float64), np.asarray(time, dtype=np.float64)
    if values.ndim != 1 or time.shape != values.shape or time.size < 5:
        raise ValueError("Each DCT-mode fit needs at least five matching coefficients and saved times.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("DCT coefficients must be finite and saved times must increase strictly.")
    if not np.isfinite(amplitude_floor) or amplitude_floor < 0:
        raise ValueError("The coefficient precision floor must be finite and nonnegative.")
    variation = float(np.ptp(values))
    max_abs = float(np.max(np.abs(values)))
    floor = max(amplitude_floor, 100 * np.finfo(float).eps * max_abs)
    if variation <= floor:
        return DCTModeFit(np.nan, np.nan, np.nan, np.nan, np.full_like(values, np.mean(values)), max_abs, "constant or below precision")
    duration = float(time[-1] - time[0])
    x = (time - time[0]) / duration
    center = float(np.mean(values))
    y = (values - center) / variation

    def shape(k):
        return x if k == 0 else np.expm1(k * x) / np.expm1(k)

    def project(k):
        design = np.column_stack((np.ones_like(x), shape(k)))
        parameters = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design @ parameters
        return float(residual @ residual), parameters, residual

    # A coarse global scan avoids relying on a gamma=0 starting guess, where
    # the conventional exponential amplitude and offset become degenerate.
    positive = np.geomspace(1e-4, 50, 48)
    grid = np.r_[-positive[::-1], 0.0, positive]
    losses = np.array([project(k)[0] for k in grid])
    best = int(np.argmin(losses))
    candidates = [float(grid[best]), 0.0]
    if 0 < best < len(grid) - 1:
        optimum = minimize_scalar(
            lambda k: project(k)[0], bounds=(grid[best - 1], grid[best + 1]),
            method="bounded", options={"xatol": 1e-10},
        )
        if optimum.success:
            candidates.append(float(optimum.x))
    k = min(candidates, key=lambda candidate: project(candidate)[0])
    ss, parameters, residual = project(k)
    fitted = center + variation * (y - residual)
    total_ss = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss / total_ss
    relative_rmse = float(np.sqrt(ss / total_ss))
    derivative_step = 1e-4 * max(1.0, abs(k))
    derivative = (shape(k + derivative_step) - shape(k - derivative_step)) / (2 * derivative_step)
    jacobian = np.column_stack((np.ones_like(x), shape(k), parameters[1] * derivative))
    _, jacobian_values, jacobian_right = np.linalg.svd(jacobian, full_matrices=False)
    if jacobian_values[-1] > 1e-12 * jacobian_values[0]:
        k_variance = ss / (len(time) - 3) * np.sum((jacobian_right[:, 2] / jacobian_values) ** 2)
        rate_std = float(np.sqrt(k_variance) / duration)
    else:
        rate_std = np.inf
    rate = k / duration
    if abs(k) >= 50 * (1 - 1e-6):
        status = "search limit; rate unresolved"
        rate = np.nan
    elif abs(k) < 1e-6 or not np.isfinite(rate_std) or abs(rate) <= 2 * rate_std:
        status = "nearly linear or uncertain; rate unresolved"
        rate = np.nan
    elif relative_rmse > 0.1:
        status = "poor single-exponential fit"
    else:
        status = "ok"
    return DCTModeFit(rate, rate_std, r_squared, relative_rmse, fitted, max_abs, status)


def uniform_step(time: NDArray[np.float64]) -> float:
    """Validate strictly increasing, uniformly saved physical times in seconds."""
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or time.size < 3 or not np.all(np.isfinite(time)):
        raise ValueError("Need at least three finite saved times.")
    differences = np.diff(time)
    dt = float(np.mean(differences))
    tolerance = 100 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(time))))
    if np.any(differences <= 0) or not np.allclose(
        differences, dt, rtol=1e-7, atol=tolerance
    ):
        raise ValueError(
            "Saved times must be strictly increasing and uniform. "
            "Interpolate irregular observations onto a uniform grid before Hankel analysis."
        )
    return dt


def prepare_uniform_observations(
    observations: NDArray[np.float64], time: NDArray[np.float64]
) -> UniformObservations:
    """Preserve uniform data, or linearly interpolate every mode onto N times.

    The target grid spans the same first and last timestamps and has the same
    number of samples. Saved observations remain untouched. This accommodates
    simulation saving at an elapsed-time threshold, which can produce intervals
    differing by one solver step. It also handles a partial final save without
    dropping the user's selected endpoint. Interpolation is an approximation,
    especially for dynamics near the saved-time Nyquist frequency.
    """
    time = np.asarray(time, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    if time.ndim != 1 or time.size < 3 or not np.all(np.isfinite(time)):
        raise ValueError("Need at least three finite saved times.")
    differences = np.diff(time)
    if np.any(differences <= 0):
        raise ValueError("Saved times must be strictly increasing; interpolation cannot fix repeated or reversed times.")
    if observations.ndim != 2 or observations.shape[0] != time.size or observations.shape[1] == 0:
        raise ValueError("Observations must have shape (saved time, mode).")
    if not np.all(np.isfinite(observations)):
        raise ValueError("Observations contain NaN or infinite values.")
    deviation = float(np.max(np.abs(differences / np.mean(differences) - 1)))
    try:
        uniform_step(time)
    except ValueError:
        target_time = np.linspace(time[0], time[-1], time.size)
        # One set of interpolation weights serves all DCT modes; no extrapolation.
        left = np.clip(np.searchsorted(time, target_time, side="right") - 1, 0, time.size - 2)
        weights = ((target_time - time[left]) / (time[left + 1] - time[left]))[:, None]
        interpolated = observations[left] + weights * (observations[left + 1] - observations[left])
        interpolated[0], interpolated[-1] = observations[0], observations[-1]
        return UniformObservations(target_time, interpolated, True, deviation)
    return UniformObservations(time, observations, False, deviation)


def block_hankel(observations: NDArray[np.float64], delays: int) -> NDArray[np.float64]:
    """Return H[(ell*M):(ell+1)*M, j] = observations[j+ell, :]."""
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim != 2 or observations.shape[1] == 0:
        raise ValueError("Observations must have shape (time, mode), with at least one mode.")
    if not np.all(np.isfinite(observations)):
        raise ValueError("Observations contain NaN or infinite values.")
    n_times, n_modes = observations.shape
    if not isinstance(delays, (int, np.integer)) or not 2 <= delays <= n_times - 1:
        raise ValueError("Delay blocks must lie between 2 and N - 1.")
    n_columns = n_times - delays + 1
    hankel = np.empty((delays * n_modes, n_columns), dtype=np.float64)
    for ell in range(delays):
        hankel[ell * n_modes : (ell + 1) * n_modes] = observations[
            ell : ell + n_columns
        ].T
    return hankel


def hankel_subspace(
    observations: NDArray[np.float64], delays: int, *, relative_cutoff: float = 1e-10
) -> HankelSubspace:
    """Compute the thin SVD once; subsequent rank changes reuse this subspace."""
    if not 0 < relative_cutoff < 1:
        raise ValueError("The relative singular-value cutoff must be between 0 and 1.")
    hankel = block_hankel(observations, delays)
    left, values, _ = svd(hankel, full_matrices=False, check_finite=False)
    numerical_rank = int(np.count_nonzero(values > relative_cutoff * values[0]))
    return HankelSubspace(
        left, values, hankel.shape, delays, observations.shape[1], numerical_rank
    )


def estimate_growth(
    observations: NDArray[np.float64],
    time: NDArray[np.float64],
    subspace: HankelSubspace,
    rank: int,
) -> GrowthEstimate:
    """Use U_top R ≈ U_bottom and log(eig(R))/saved_dt (block ESPRIT).

    Removing one M-row block at each end shifts the embedding by one *saved*
    time step. A global exponential least-squares fit supplies reconstruction
    diagnostics and spatial amplitudes; it does not change the extracted poles.
    """
    observations = np.asarray(observations, dtype=np.float64)
    dt = uniform_step(time)
    if observations.shape != (len(time), subspace.n_modes):
        raise ValueError("Time and observation shapes differ from the Hankel subspace.")
    if subspace.shape[1] != len(time) - subspace.delays + 1:
        raise ValueError("Hankel columns do not match the number of saved times.")
    max_rank = min(subspace.numerical_rank, (subspace.delays - 1) * subspace.n_modes)
    if not isinstance(rank, (int, np.integer)) or not 1 <= rank <= max_rank:
        raise ValueError(f"Retained rank must lie between 1 and {max_rank}.")
    basis = subspace.left_vectors[:, :rank]
    top, bottom = basis[:-subspace.n_modes], basis[subspace.n_modes:]
    shift, _, top_rank, _ = np.linalg.lstsq(top, bottom, rcond=None)
    if top_rank != rank:
        raise ValueError("The shifted subspace is rank deficient; reduce rank or add delay blocks.")
    poles = np.linalg.eigvals(shift).astype(np.complex128)
    # Zero poles have no finite continuous-time logarithm; preserve and flag them.
    exponents = np.full(rank, complex(-np.inf, np.nan), dtype=np.complex128)
    nonzero = np.abs(poles) > 0
    exponents[nonzero] = np.log(poles[nonzero]) / dt
    order = np.argsort(-exponents.real, kind="stable")
    poles, exponents = poles[order], exponents[order]

    powers = poles[None, :] ** np.arange(len(time))[:, None]
    if not np.all(np.isfinite(powers)):
        raise ValueError("Extracted poles overflow the exponential fit; reduce rank or the time window.")
    # Normalize each temporal column before fitting to limit scaling differences.
    scales = np.max(np.abs(powers), axis=0)
    amplitudes = np.linalg.lstsq(powers / scales, observations, rcond=None)[0] / scales[:, None]
    reconstruction = powers @ amplitudes
    weights = subspace.singular_values / subspace.singular_values[0]
    return GrowthEstimate(
        discrete_eigenvalues=poles,
        exponents=exponents,
        spatial_amplitudes=amplitudes.T,
        reconstruction=reconstruction,
        retained_energy=float(np.sum(weights[:rank] ** 2) / np.sum(weights ** 2)),
        shift_residual=float(np.linalg.norm(top @ shift - bottom) / np.linalg.norm(bottom)),
        shift_condition=float(np.linalg.cond(top)),
        reconstruction_error=float(
            np.linalg.norm(reconstruction - observations) / np.linalg.norm(observations)
        ),
    )
