"""Float64 Taylor FVM RHS and its classical Jacobian in stored populations.

Source: CuKernelInte, CuKernelTaylFace, CuKernelComputeFluxFVM in
src/cuda_kernel.cu. U[j, i] already includes density-bin quadrature; psi=sum_j U.
This is the semidiscrete operator, before Euler stepping and CFL rejection.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaylorFVM:
    n: int = 64
    length: float = 0.07
    nu: float = -4e-30
    mu: float = -4e-37
    alpha: float = 0.0
    beta: float = 7.4e23

    def __post_init__(self):
        if self.n < 2 or self.length <= 0:
            raise ValueError("Need n >= 2 and positive domain length")

    @property
    def h(self):
        return self.length / self.n

    @property
    def rho(self):
        return np.linspace(1085.0, 1115.0, self.n)

    @property
    def z(self):
        return (np.arange(self.n) + 0.5) * self.h

    @property
    def percoll(self):
        # LINEAR_FULL; for ZERO alpha=0 makes this profile irrelevant.
        return 1115.0 - 30.0 * self.z / self.length

    def interaction(self, psi):
        """Interior faces only; endpoint fluxes are prescribed separately."""
        psi = np.asarray(psi, dtype=np.float64)
        f = np.arange(1, self.n)
        a, b, c, d = (psi[np.clip(f + shift, 0, self.n - 1)]
                      for shift in (-2, -1, 0, 1))
        return (self.nu / (24 * self.h) * (a - 27*b + 27*c - d)
                + self.mu / self.h**3 * (d - 3*c + 3*b - a))

    def velocity(self, u):
        p_face = (self.percoll[:-1] + self.percoll[1:]) / 2
        return (-self.alpha * (self.rho[:, None] - p_face)
                - self.beta * self.interaction(np.asarray(u, dtype=np.float64).sum(axis=0)))

    def flux(self, u):
        u = np.asarray(u, dtype=np.float64)
        v = self.velocity(u)
        flux = np.zeros((self.n, self.n + 1))
        flux[:, 1:-1] = v * np.where(v > 0, u[:, :-1], u[:, 1:])
        return flux

    def rhs(self, u):
        return -np.diff(self.flux(u), axis=1) / self.h

    def jacobian_action(self, base, q):
        """DF(base) q; refuse unresolved upwind kinks rather than freeze a tie.

        At v=0 equal adjacent base populations give a unique derivative.
        With no interaction, delta-v=0 identically and a tie is harmless too.
        """
        base, q = np.asarray(base, dtype=float), np.asarray(q, dtype=float)
        v = self.velocity(base)
        if self.beta != 0 and (self.nu != 0 or self.mu != 0):
            if np.any((v == 0) & (base[:, :-1] != base[:, 1:])):
                raise ValueError("No classical flux Jacobian at a zero-velocity population jump")
        up = np.where(v > 0, base[:, :-1], base[:, 1:])
        q_up = np.where(v > 0, q[:, :-1], q[:, 1:])
        dv = -self.beta * self.interaction(q.sum(axis=0))
        dflux = np.zeros((self.n, self.n + 1))
        dflux[:, 1:-1] = up * dv + v * q_up
        return -np.diff(dflux, axis=1) / self.h


def homogeneous_state(model, psi=0.02):
    # Nonuniform density weights exercise the rank-one density coupling.
    weights = np.exp(-0.5 * ((model.rho - 1100.0) / 4.0)**2)
    weights /= weights.sum()
    return np.broadcast_to(psi * weights[:, None], (model.n, model.n)).copy()


def ridge_state(model, psi=0.02):
    """Use the production initializer, including its nearest-bin convention."""
    from .phi import phi_linear_full_ridge

    return phi_linear_full_ridge(model.rho, model.z, psi, 1100.0)


def continuum_rates(model, psi=0.02, modes=None, *, ridge=False):
    modes = np.arange(model.n) if modes is None else np.asarray(modes)
    k = modes * np.pi / model.length
    rates = model.beta * psi * (-model.nu * k**2 + model.mu * k**4)
    return rates - model.alpha * 30 / model.length if ridge else rates


def discrete_rates(model, psi=0.02, modes=None):
    modes = np.arange(model.n) if modes is None else np.asarray(modes)
    khat = 2 / model.h * np.sin(modes * np.pi / (2 * model.n))
    return model.beta * psi * (
        -model.nu * khat**2 * (1 + model.h**2 * khat**2 / 24)
        + model.mu * khat**4
    )


def homogeneous_matrix(model, psi=0.02):
    """Assemble marginal Jacobian from flux stencils, without using its symbol."""
    if model.alpha != 0:
        raise ValueError("Homogeneous benchmark requires alpha=0")
    columns = []
    for direction in np.eye(model.n):
        dflux = np.zeros(model.n + 1)
        dflux[1:-1] = -model.beta * psi * model.interaction(direction)
        columns.append(-np.diff(dflux) / model.h)
    return np.column_stack(columns)


def dense_jacobian(model, base):
    """Small-grid reference only; production-size matrices scale as N^4."""
    if model.n > 24:
        raise ValueError("Use Jacobian actions or density blocks for n > 24")
    return np.column_stack([
        model.jacobian_action(base, q.reshape(base.shape)).ravel()
        for q in np.eye(base.size)
    ])


def fixed_mass_basis(n):
    """Orthonormal columns spanning sum_i q[j,i]=0 for every density class."""
    modes = np.arange(1, n)
    spatial = np.sqrt(2 / n) * np.cos(np.pi * (np.arange(n)[:, None] + 0.5) * modes / n)
    return np.kron(np.eye(n), spatial)


def ridge_rates(model):
    """Exact full spectrum from triangular transport blocks, including N zeros.

    For class j the occupied cell is r=N-1-j. Every other cell flows toward r.
    Returns [j,i] eigenvalues labelled by departure cell, NOT by cosine mode.
    """
    if model.alpha <= 0:
        raise ValueError("Ridge transport spectrum requires alpha > 0")
    r = (model.n - 1 - np.arange(model.n))[:, None]
    i = np.arange(model.n)[None, :]
    outflow = np.where(i < r, r / (model.n - 1) - (i + 1) / model.n,
                      np.where(i > r, i / model.n - r / (model.n - 1), 0.0))
    return -model.alpha * 30 / model.h * outflow


def derivative_sweep(model, base, q, amplitudes):
    """Signed FD diagnostic; epsilon*q has the units of stored populations.

    Centered differences can leave the nonnegative cone at a ridge. The forward
    difference is physical if q is nonnegative off the occupied support.
    """
    action = model.jacobian_action(base, q)
    norm = np.linalg.norm(action)
    if norm == 0:
        raise ValueError("Choose a direction with a nonzero Jacobian action")
    v0 = model.velocity(base)
    f0 = model.rhs(base)
    rows = []
    for eps in amplitudes:
        plus, minus = base + eps*q, base - eps*q
        fp, fm = model.rhs(plus), model.rhs(minus)
        rows.append({
            "epsilon": float(eps),
            "forward_relative_error": float(np.linalg.norm((fp-f0)/eps-action) / norm),
            "centered_relative_error": float(np.linalg.norm((fp-fm)/(2*eps)-action) / norm),
            "plus_nonnegative": bool(np.min(plus) >= 0),
            "changed_face_signs": int(np.count_nonzero(
                np.sign(model.velocity(plus)) != np.sign(v0))),
            "minus_changed_face_signs": int(np.count_nonzero(
                np.sign(model.velocity(minus)) != np.sign(v0))),
        })
    return rows


def ridge_direction(model, seed=42):
    """A fixed-mass, one-sided admissible redistribution; max absolute entry=1."""
    q = np.random.default_rng(seed).uniform(0.1, 1.0, (model.n, model.n))
    sinks = model.n - 1 - np.arange(model.n)
    q[np.arange(model.n), sinks] = 0
    q[np.arange(model.n), sinks] = -q.sum(axis=1)
    return q / np.max(np.abs(q))
