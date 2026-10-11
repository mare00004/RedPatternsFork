"""Physical-rate and sampling checks for block-Hankel growth extraction."""

import numpy as np
import pytest
from scipy.fft import dct, idct

from red_patterns.linear_growth import (
    block_hankel,
    estimate_growth,
    fit_dct_mode,
    hankel_subspace,
    prepare_uniform_observations,
    recover_dynamic_modes,
    uniform_step,
)


def test_recovers_signed_growth_decay_and_constant_from_paired_psi():
    time = np.arange(81) * 0.25 + 13.0
    elapsed = time - time[0]
    coefficients = np.zeros((len(time), 16))
    coefficients[:, 3] = 1e-7 * np.expm1(0.17 * elapsed)
    coefficients[:, 8] = 2e-7 * np.expm1(-0.09 * elapsed)
    no_interaction = np.full_like(coefficients, 0.02)
    interaction = no_interaction + idct(coefficients, type=2, norm="ortho", axis=1)
    vectors = dct(interaction - no_interaction, type=2, norm="ortho", axis=1)

    subspace = hankel_subspace(vectors, delays=10)
    estimate = estimate_growth(vectors, time, subspace, rank=3)

    np.testing.assert_allclose(estimate.exponents.real, [0.17, 0.0, -0.09], atol=1e-9)
    np.testing.assert_allclose(estimate.exponents.imag, 0.0, atol=1e-9)
    assert estimate.reconstruction_error < 1e-9
    assert estimate.shift_residual < 1e-9
    assert estimate.retained_energy > 1 - 1e-12


@pytest.mark.parametrize("rate", [0.17, -0.09])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_individual_dct_rates_with_zero_initial_difference_and_irregular_times(rate, sign):
    time = 13 + np.r_[0, np.cumsum(np.tile([0.25, 0.26, 0.25], 30))]
    values = sign * 1e-10 * np.expm1(rate * (time - time[0]))
    result = fit_dct_mode(values, time)
    assert result.status == "ok"
    assert result.rate == pytest.approx(rate, abs=1e-8)
    assert result.relative_rmse < 1e-8
    np.testing.assert_allclose(result.fitted, values, atol=1e-17)


def test_individual_mode_fit_recovers_exponential_plus_arbitrary_offset():
    time = np.linspace(7, 40, 150)
    values = 0.003 - 2e-5 * np.exp(0.045 * (time - time[0]))
    result = fit_dct_mode(values, time)
    assert result.status == "ok"
    assert result.rate == pytest.approx(0.045, abs=1e-8)


@pytest.mark.parametrize("values", [np.zeros(30), np.full(30, 0.1)])
def test_unexcited_or_constant_dct_modes_have_no_reported_rate(values):
    result = fit_dct_mode(values, np.arange(30))
    assert result.status == "constant or below precision"
    assert np.isnan(result.rate)


def test_below_precision_mode_has_no_reported_rate():
    values = 1e-15 * np.arange(30)
    result = fit_dct_mode(values, np.arange(30), amplitude_floor=1e-12)
    assert np.isnan(result.rate)


def test_linear_trace_does_not_invent_an_exponential_rate():
    time = np.arange(30)
    result = fit_dct_mode(3e-7 * time, time)
    assert result.status == "nearly linear or uncertain; rate unresolved"
    assert np.isnan(result.rate)


def test_oscillating_dct_coefficient_is_marked_as_a_poor_or_unresolved_fit():
    time = np.linspace(0, 40, 150)
    result = fit_dct_mode(1e-6 * np.exp(0.01 * time) * np.cos(time), time)
    assert result.status != "ok"
    assert result.relative_rmse > 0.5


def test_rate_above_search_limit_is_marked_unresolved():
    time = np.linspace(0, 1, 120)
    result = fit_dct_mode(np.exp(70 * time), time)
    assert result.status == "search limit; rate unresolved"
    assert np.isnan(result.rate)


def test_noisy_exponential_reports_rate_uncertainty():
    time = np.linspace(0, 20, 100)
    random = np.random.default_rng(1)
    result = fit_dct_mode(np.expm1(0.12 * time) + random.normal(0, 0.01, time.size), time)
    assert result.status == "ok"
    assert result.rate == pytest.approx(0.12, abs=1e-3)
    assert np.isfinite(result.rate_std) and result.rate_std > 0


def test_delay_embedding_recovers_two_rates_from_one_spatial_observable():
    time = np.arange(75) * 0.3
    observations = (np.exp(0.04 * time) + 0.3 * np.exp(-0.12 * time))[:, None]
    subspace = hankel_subspace(observations, delays=12)
    estimate = estimate_growth(observations, time, subspace, rank=2)
    np.testing.assert_allclose(estimate.exponents.real, [0.04, -0.12], atol=1e-10)
    assert estimate.reconstruction_error < 1e-10


def test_recovers_growth_and_angular_frequency_of_oscillatory_pair():
    time = np.arange(100) * 0.15
    observations = (np.exp(0.06 * time) * np.cos(0.8 * time))[:, None]
    subspace = hankel_subspace(observations, delays=15)
    estimate = estimate_growth(observations, time, subspace, rank=2)
    np.testing.assert_allclose(estimate.exponents.real, [0.06, 0.06], atol=1e-10)
    np.testing.assert_allclose(np.sort(estimate.exponents.imag), [-0.8, 0.8], atol=1e-10)
    assert estimate.reconstruction_error < 1e-10


def test_hankel_overlap_uses_every_frame_and_preserves_mode_order():
    observations = np.arange(14).reshape(7, 2)
    hankel = block_hankel(observations, delays=3)
    assert hankel.shape == (6, 5)
    np.testing.assert_array_equal(hankel[:, 0], observations[:3].reshape(-1))
    np.testing.assert_array_equal(hankel[:, -1], observations[-3:].reshape(-1))
    np.testing.assert_array_equal(hankel[2:, :-1], hankel[:-2, 1:])


@pytest.mark.parametrize("time", [[0, 1, 2.1], [0, 1, 1], [2, 1, 0], [0, 1, np.nan], [0, 1]])
def test_rejects_invalid_saved_sampling(time):
    with pytest.raises(ValueError):
        uniform_step(np.array(time))


def test_uniform_sampling_tolerates_float64_accumulation():
    time = np.cumsum(np.full(1200, 0.03))
    assert uniform_step(time) == pytest.approx(0.03)


def test_irregular_threshold_saves_preserve_originals_and_selected_endpoint():
    time = np.r_[0, np.cumsum(np.tile([1.0, 1.0, 1.01, 1.01], 15))]
    observations = np.column_stack([2 * time + 1, -3 * time + 4])
    original = observations.copy()
    prepared = prepare_uniform_observations(observations, time)
    assert prepared.resampled
    assert prepared.max_relative_step_deviation > 0.004
    assert prepared.values.shape == observations.shape
    assert prepared.time[0] == time[0]
    assert prepared.time[-1] == time[-1]
    np.testing.assert_array_equal(observations, original)
    np.testing.assert_array_equal(prepared.values[[0, -1]], observations[[0, -1]])
    np.testing.assert_allclose(prepared.values[:, 0], 2 * prepared.time + 1)
    np.testing.assert_allclose(prepared.values[:, 1], -3 * prepared.time + 4)
    assert uniform_step(prepared.time) == pytest.approx((time[-1] - time[0]) / (len(time) - 1))


def test_uniform_observations_do_not_interpolate():
    time = np.arange(30) * 0.1
    observations = np.random.default_rng(4).normal(size=(30, 3))
    prepared = prepare_uniform_observations(observations, time)
    assert not prepared.resampled
    np.testing.assert_array_equal(prepared.time, time)
    np.testing.assert_array_equal(prepared.values, observations)


@pytest.mark.parametrize("rate", [0.08, -0.06])
def test_rates_use_uniform_physical_time_after_threshold_save_interpolation(rate):
    time = np.r_[0, np.cumsum(np.tile([1.0, 1.01, 1.01], 20))]
    observations = np.exp(rate * time)[:, None]
    prepared = prepare_uniform_observations(observations, time)
    subspace = hankel_subspace(prepared.values, delays=10)
    estimate = estimate_growth(prepared.values, prepared.time, subspace, rank=1)
    assert estimate.exponents[0].real == pytest.approx(rate, abs=1e-5)
    assert estimate.reconstruction_error < 0.001


def test_final_partial_save_is_interpolated_without_dropping_endpoint():
    time = np.array([0, 1, 2, 3, 3.4])
    prepared = prepare_uniform_observations(time[:, None], time)
    assert prepared.resampled
    np.testing.assert_allclose(prepared.time, [0, 0.85, 1.7, 2.55, 3.4])
    np.testing.assert_allclose(prepared.values[:, 0], prepared.time)


@pytest.mark.parametrize("time", [[0, 1, 1], [0, 2, 1], [0, 1, np.nan]])
def test_interpolation_rejects_unusable_timestamps(time):
    with pytest.raises(ValueError):
        prepare_uniform_observations(np.ones((3, 2)), np.array(time))


def test_zero_signal_has_no_identifiable_rates():
    observations = np.zeros((20, 4))
    subspace = hankel_subspace(observations, delays=5)
    assert subspace.numerical_rank == 0
    with pytest.raises(ValueError, match="Retained rank"):
        estimate_growth(observations, np.arange(20), subspace, rank=1)


def test_rank_above_identifiable_subspace_is_rejected():
    observations = np.exp(0.05 * np.arange(30))[:, None]
    subspace = hankel_subspace(observations, delays=5)
    with pytest.raises(ValueError, match="Retained rank"):
        estimate_growth(observations, np.arange(30), subspace, rank=2)


@pytest.mark.parametrize("delays", [0, 1, 6, 7, 2.5])
def test_invalid_embedding_is_rejected(delays):
    with pytest.raises(ValueError, match="Delay blocks"):
        block_hankel(np.ones((6, 2)), delays)


def test_nonfinite_signal_is_rejected():
    with pytest.raises(ValueError, match="NaN or infinite"):
        block_hankel(np.array([[0], [np.nan], [1]]), delays=2)


def test_notebook_hdf5_reader_preserves_perturbations_below_float32_precision(tmp_path):
    import importlib.util
    from pathlib import Path

    import h5py

    notebook_path = Path(__file__).resolve().parents[1] / "analysis/analyze_linear_growth.py"
    spec = importlib.util.spec_from_file_location("linear_growth_notebook", notebook_path)
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    psi = np.full((5, 4), 0.02)
    psi[-1, 2] += 1e-11
    path = tmp_path / "run.h5"
    with h5py.File(path, "w") as handle:
        handle["time"] = np.arange(5, dtype=float)
        handle["coords/z"] = np.arange(4, dtype=float)
        handle["fields/psi"] = psi
        handle.create_group("config/model")
        handle.create_group("config/run")
        taylor = handle.create_group("config/model/variant/tayl")
        taylor.attrs["NU"] = 0.0
        taylor.attrs["MU"] = 0.0

    loaded = notebook.read_psi_run(path)
    np.testing.assert_array_equal(loaded["psi"], psi)
    assert loaded["psi"].dtype == np.float64
    assert loaded["psi"][-1, 2] > loaded["psi"][0, 2]
    assert loaded["zero_interaction"] is True


def test_dynamic_modes_recover_mixed_cosine_shapes_and_shared_rates():
    time = np.arange(101) * 0.2 + 9
    modes = np.array([3, 5, 7])
    coefficients = np.array([[1, 0.3], [0.6, -0.5], [-0.3, 1.0]])
    observations = np.exp((time - time[0])[:, None] * [0.12, 0.06]) @ coefficients.T
    growth = estimate_growth(observations, time, hankel_subspace(observations, 15), 2)
    recovered = recover_dynamic_modes(growth, modes, 32)
    np.testing.assert_allclose(growth.exponents, [0.12, 0.06], atol=1e-11)
    np.testing.assert_allclose(recovered.dct_shapes * recovered.amplitudes, coefficients, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(recovered.psi_shapes, axis=0), 1, atol=1e-12)
    full = np.zeros((32, 2))
    full[modes] = coefficients
    np.testing.assert_allclose(recovered.psi_shapes * recovered.amplitudes, idct(full, type=2, norm="ortho", axis=0), atol=1e-10)
    transformed = dct(recovered.psi_shapes, type=2, norm="ortho", axis=0)
    np.testing.assert_allclose(transformed[np.setdiff1d(np.arange(32), modes)], 0, atol=1e-14)
    pivots = recovered.psi_shapes[np.argmax(np.abs(recovered.psi_shapes), axis=0), np.arange(2)]
    np.testing.assert_allclose(pivots.imag, 0, atol=1e-14)
    assert np.all(pivots.real > 0)
    np.testing.assert_array_equal(recovered.conjugate_partners, [-1, -1])


def test_complex_dynamic_modes_preserve_phase_and_pair_real_pattern():
    time = np.arange(151) * 0.15
    shape = np.array([1.0 + 0.2j, -0.4 + 0.8j, 0.6 - 0.3j])
    exponent = 0.08 + 0.9j
    observations = 2 * (np.exp(exponent * time[:, None]) * shape).real
    growth = estimate_growth(observations, time, hankel_subspace(observations, 20), 2)
    recovered = recover_dynamic_modes(growth, np.array([2, 4, 6]), 16)
    np.testing.assert_array_equal(recovered.conjugate_partners, [1, 0])
    np.testing.assert_allclose(recovered.dct_shapes * recovered.amplitudes, growth.spatial_amplitudes, atol=1e-12)
    full = np.zeros((len(time), 16))
    full[:, [2, 4, 6]] = observations
    powers = growth.discrete_eigenvalues[None, :] ** np.arange(len(time))[:, None]
    reconstruction = (powers * recovered.amplitudes) @ recovered.psi_shapes.T
    np.testing.assert_allclose(reconstruction.real, idct(full, type=2, norm="ortho", axis=1), atol=1e-11)
    np.testing.assert_allclose(reconstruction.imag, 0, atol=1e-11)
    np.testing.assert_allclose(recovered.psi_shapes[:, 0], recovered.psi_shapes[:, 1].conjugate(), atol=1e-11)


@pytest.mark.parametrize("modes", [[-1, 2], [1, 1], [1, 4], [1.0, 2.0]])
def test_dynamic_modes_reject_invalid_dct_indices(modes):
    time = np.arange(30) * 0.1
    x = np.exp(0.1 * time)[:, None] * np.array([[1, 2]])
    growth = estimate_growth(x, time, hankel_subspace(x, 5), 1)
    with pytest.raises(ValueError, match="Mode indices"):
        recover_dynamic_modes(growth, np.array(modes), 4)


def test_notebook_compares_dynamics_independently_of_initial_conditions():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "analysis/analyze_linear_growth.py"
    spec = importlib.util.spec_from_file_location("dynamic_modes_notebook", path)
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    perturbed, equilibrium = notebook.demo_pair()
    differences, missing = notebook.compare_dynamics(perturbed, equilibrium)
    assert differences == missing == []
    assert not np.array_equal(perturbed["psi"][0], equilibrium["psi"][0])
    different_operator = dict(equilibrium, variant={"NU": 0.0, "MU": 0.0})
    differences, _ = notebook.compare_dynamics(perturbed, different_operator)
    assert set(differences) == {"NU", "MU"}
    conv = dict(equilibrium, model=dict(equilibrium["model"], modelType="CONV"), variant={}, kernel=np.ones(3))
    differences, missing = notebook.compare_dynamics(conv, dict(conv, kernel=np.zeros(3)))
    assert differences == ["convolution kernel values"]
    _, missing = notebook.compare_dynamics(conv, dict(conv, kernel=None))
    assert missing == ["convolution kernel values"]
