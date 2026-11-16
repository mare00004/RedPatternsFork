"""Independent spectral, conservation, and nonlinear-difference benchmarks."""

from dataclasses import replace

import numpy as np
import pytest

from red_patterns.jacobian import (
    TaylorFVM, continuum_rates, dense_jacobian, derivative_sweep,
    discrete_rates, fixed_mass_basis, homogeneous_matrix, homogeneous_state,
    ridge_direction, ridge_rates, ridge_state,
)


@pytest.mark.parametrize("n", [8, 31, 64, 256])
def test_homogeneous_cosines_diagonalize_actual_stencil(n):
    model = TaylorFVM(n=n)
    matrix = homogeneous_matrix(model)
    modes = np.arange(n)
    cosines = np.cos(np.pi * (np.arange(n)[:, None] + 0.5) * modes / n)
    expected = discrete_rates(model)
    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(matrix @ cosines, cosines * expected, atol=scale*2e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(matrix), np.sort(expected), atol=scale*2e-13)
    np.testing.assert_allclose(matrix.sum(axis=0), 0, atol=scale*2e-15)


def test_full_homogeneous_jacobian_and_neutral_composition():
    model = TaylorFVM(n=8)
    base = homogeneous_state(model)
    matrix = dense_jacobian(model, base)
    expected = np.r_[discrete_rates(model)[1:], np.zeros(model.n**2-model.n+1)]
    eigenvalues = np.linalg.eigvals(matrix)
    np.testing.assert_allclose(eigenvalues.imag, 0, atol=1e-14)
    np.testing.assert_allclose(np.sort(eigenvalues.real), np.sort(expected), atol=1e-14)
    q = np.random.default_rng(1).normal(size=base.shape)
    q -= q.mean(axis=0)
    np.testing.assert_allclose(model.jacobian_action(base, q), 0, atol=1e-14)
    np.testing.assert_allclose(model.jacobian_action(base, np.ones_like(base)).sum(axis=1), 0, atol=1e-14)


def test_continuum_convergence_at_fixed_physical_mode():
    errors = []
    for n in (128, 256, 512, 1024):
        model = TaylorFVM(n=n)
        errors.append(abs(discrete_rates(model, modes=[40])[0]
                          - continuum_rates(model, modes=[40])[0]))
    assert np.all(np.diff(errors) < 0)
    assert 3.7 < errors[-2] / errors[-1] < 4.3


@pytest.mark.parametrize("n", [2, 7, 32, 64, 256])
def test_ridge_is_equilibrium_with_empty_upstream_cells(n):
    model = TaylorFVM(n=n, alpha=2e-5)
    base = ridge_state(model)
    np.testing.assert_array_equal(np.argmax(base, axis=1), n-1-np.arange(n))
    v = model.velocity(base)
    assert np.min(np.abs(v)) > 0
    assert np.max(np.where(v > 0, base[:, :-1], base[:, 1:])) == 0
    np.testing.assert_array_equal(model.rhs(base), 0)
    q = ridge_direction(model)
    np.testing.assert_allclose(model.jacobian_action(base, q).sum(axis=1), 0, atol=1e-15)
    np.testing.assert_allclose(model.jacobian_action(base, q),
                               replace(model, nu=0, mu=0).jacobian_action(base, q), atol=1e-14)


def test_ridge_full_and_fixed_mass_spectra():
    model = TaylorFVM(n=9, alpha=2e-5)
    base = ridge_state(model)
    matrix = dense_jacobian(model, base)
    expected = ridge_rates(model).ravel()
    np.testing.assert_allclose(np.sort(np.linalg.eigvals(matrix)), np.sort(expected), atol=2e-14)
    basis = fixed_mass_basis(model.n)
    reduced = basis.T @ matrix @ basis
    # Invariance of the fixed-mass space is checked independently of eigenvalues.
    np.testing.assert_allclose(matrix @ basis, basis @ reduced, atol=1e-15)
    values = np.linalg.eigvals(reduced)
    np.testing.assert_allclose(values.imag, 0, atol=1e-14)
    np.testing.assert_allclose(np.sort(values.real), np.sort(expected[expected < 0]), atol=1e-13)
    assert max(values.real) == pytest.approx(-model.alpha*30/(model.length*(model.n-1)))


def test_nonlinear_rhs_derivatives_converge_on_both_benchmarks():
    model = TaylorFVM(n=32)
    base = homogeneous_state(model)
    q = np.random.default_rng(2).normal(size=base.shape)
    q -= q.mean(axis=1, keepdims=True)
    q /= np.max(np.abs(q))
    rows = derivative_sweep(model, base, q, [1e-7, 1e-8, 1e-9])
    assert rows[-1]["forward_relative_error"] < rows[0]["forward_relative_error"] / 50
    assert rows[-1]["centered_relative_error"] < 1e-6

    forced = replace(model, alpha=2e-5)
    rows = derivative_sweep(forced, ridge_state(forced), ridge_direction(forced), [1e-5, 1e-6, 1e-7])
    assert all(row["plus_nonnegative"] and row["changed_face_signs"] == 0 for row in rows)
    assert rows[-1]["forward_relative_error"] < rows[0]["forward_relative_error"] / 50
    assert rows[-1]["centered_relative_error"] < 1e-8


def test_zero_velocity_jump_is_not_silently_given_a_jacobian():
    model = TaylorFVM(n=8)  # Constant marginal, zero buoyancy, discontinuous populations.
    base = ridge_state(model, psi=0.125)  # Exactly representable constant gives v=0.
    with pytest.raises(ValueError, match="No classical"):
        model.jacobian_action(base, np.ones_like(base))


def test_euler_amplification_of_admissible_homogeneous_mode():
    model = TaylorFVM(n=64)
    base = homogeneous_state(model)
    mode, dt, eps = 20, 0.005, 1e-6
    cosine = np.cos(mode * np.pi * model.z / model.length)
    q = base * cosine
    advanced = base + eps*q + dt*model.rhs(base + eps*q)
    coefficient = np.dot((advanced-base).sum(axis=0), cosine) / np.dot(cosine, cosine)
    expected = eps * base.sum(axis=0)[0] * (1 + dt*discrete_rates(model)[mode])
    assert coefficient == pytest.approx(expected, rel=1e-8)
