"""Analytic checks of the distinct raw-field and paired-norm measurements."""

import numpy as np
import pandas as pd
import pytest
from scipy.fft import idct

from red_patterns.dct_dispersion import demo_pair
from red_patterns.finite_time_stability import (
    analyze_window, demo_sweep, phase_coordinates, prepare_pair,
    predict_selected_modes, theory_overlays,
)


def test_fastest_raw_mode_wavelength_and_mixed_norm_endpoint_rate():
    p, e = demo_pair()
    data = prepare_pair(p, e)
    result = analyze_window(data, 0, 6, 1, 9)
    summary = result["summary"]
    assert summary["fastest_mode"] == 5
    assert summary["fastest_wavelength_cm"] == pytest.approx(2.56)
    assert summary["fastest_growth_s_inv"] == pytest.approx(0.16, abs=1e-9)
    rates = 0.16 - 0.02 * (np.arange(1, 10) - 5) ** 2
    expected = 2e-5 * np.sqrt(np.exp(2 * result["time"][:, None] * rates).sum(axis=1))
    np.testing.assert_allclose(result["distance"], expected, rtol=1e-9)
    assert summary["lambda_eff_s_inv"] == pytest.approx(
        np.log(expected[-1] / expected[0]) / 6, abs=1e-10
    )
    # Mixtures need not have the fastest modal growth rate.
    assert summary["lambda_eff_s_inv"] < summary["fastest_growth_s_inv"]


def test_moving_reference_cancels_in_difference_but_not_raw_field():
    p, e = demo_pair()
    drift_coefficients = np.zeros_like(e["psi"])
    drift_coefficients[:, 2] = 1e-3 * np.exp(0.3 * p["time"])
    drift = idct(drift_coefficients, type=2, norm="ortho", axis=1)
    moving_p = dict(p, psi=p["psi"] + drift)
    moving_e = dict(e, psi=e["psi"] + drift)
    original = analyze_window(prepare_pair(p, e), 0, 6, 1, 9)
    moving = analyze_window(prepare_pair(moving_p, moving_e), 0, 6, 1, 9)
    np.testing.assert_allclose(moving["distance"], original["distance"], rtol=1e-9)
    assert moving["summary"]["fastest_mode"] == 2
    assert moving["summary"]["reference_drift"] > 1e-6
    assert not prepare_pair(moving_p, moving_e)["notes"]


@pytest.mark.parametrize("mode, expected", [(5, 0.16), (9, -0.16)])
def test_single_mode_nonzero_start_irregular_times(mode, expected):
    p, e = demo_pair()
    take = [0, 1, 4, 7, 10, 17, 25, 40, 60]
    p = dict(p, time=p["time"][take], psi=p["psi"][take])
    e = dict(e, time=e["time"][take], psi=e["psi"][take])
    result = analyze_window(prepare_pair(p, e), p["time"][2], p["time"][-2], mode, mode)
    assert result["summary"]["lambda_eff_s_inv"] == pytest.approx(expected, abs=1e-9)
    np.testing.assert_array_equal(result["time"], p["time"][2:-1])
    if expected < 0:
        assert np.isnan(result["summary"]["fastest_wavelength_cm"])
        assert result["summary"]["norm_status"] == "net decay"


def test_full_band_norm_equals_spatial_difference_norm_by_parseval():
    p, e = demo_pair()
    result = analyze_window(prepare_pair(p, e), 0, 6, 0, 63)
    np.testing.assert_allclose(result["distance"], np.linalg.norm((p["psi"] - e["psi"])[:41], axis=1))
    assert np.isnan(result["table"].iloc[0]["wavelength_cm"])


def test_zero_and_subprecision_endpoints_are_not_regularized():
    p, e = demo_pair()
    p["psi"][0] = e["psi"][0]
    result = analyze_window(prepare_pair(p, e), 0, 6, 1, 9)
    assert result["distance"][0] == 0
    assert np.isnan(result["summary"]["lambda_eff_s_inv"])
    assert "zero endpoint" in result["summary"]["norm_status"]
    data = prepare_pair(p, e)
    data["difference"][0, 1:10] = data["floor"][0] / 10
    result = analyze_window(data, 0, 6, 1, 9)
    assert np.isnan(result["summary"]["lambda_eff_s_inv"])
    assert "below numerical precision" in result["summary"]["norm_status"]


def test_short_window_has_norm_rate_without_inventing_modal_fits():
    p, e = demo_pair()
    result = analyze_window(prepare_pair(p, e), 0, p["time"][1], 5, 5)
    assert result["summary"]["lambda_eff_s_inv"] == pytest.approx(0.16, abs=1e-8)
    assert result["fits"] == [None]
    assert np.isnan(result["summary"]["fastest_wavelength_cm"])


def test_mismatched_time_and_dynamics_and_invalid_selections_rejected():
    p, e = demo_pair()
    with pytest.raises(ValueError, match="identical saved-time"):
        prepare_pair(p, dict(e, time=e["time"] + 0.001))
    with pytest.raises(ValueError, match="different dynamics"):
        prepare_pair(p, dict(e, variant={"NU": -1e-30, "MU": -1e-37}))
    data = prepare_pair(p, e)
    for low, high in [(-1, 5), (5, 64), (6, 5), (1.5, 5)]:
        with pytest.raises(ValueError, match="mode range"):
            analyze_window(data, 0, 6, low, high)
    with pytest.raises(ValueError, match="start < end"):
        analyze_window(data, 6, 6, 1, 9)
    with pytest.raises(ValueError, match="saved times"):
        analyze_window(data, 0, 6.01, 1, 9)


def test_saved_continuum_boundary_and_normalized_sweep_factors():
    pairs, prepared = demo_sweep()
    summary = pd.DataFrame([
        dict(pair, boundary_coefficient=prepared[pair["run_id"]]["boundary_coefficient"])
        for pair in pairs
    ])
    coefficient = 4 * 2e-5 * (30 / 0.07) / (7.4e23 * 0.03)
    np.testing.assert_allclose(summary["boundary_coefficient"], coefficient, rtol=1e-12, atol=0)
    frame, _ = phase_coordinates(summary, "normalized")
    np.testing.assert_allclose(frame["phase_x"], np.tile([0.8, 1, 1.2, 1.5], 3))
    assert (frame["x_low"] < frame["phase_x"]).all()
    assert (frame["phase_x"] < frame["x_high"]).all()
    assert (frame["y_low"] < frame["phase_y"]).all()
    assert (frame["phase_y"] < frame["y_high"]).all()
    squared, _ = phase_coordinates(summary, "squared")
    boundary_rows = np.isclose(frame["phase_x"], 1)
    np.testing.assert_allclose(
        squared.loc[boundary_rows, "phase_x"],
        coefficient * 1e23 * squared.loc[boundary_rows, "phase_y"],
    )
    np.testing.assert_array_equal(frame["run_id"], summary["run_id"])


def test_transforms_do_not_fold_wrong_signs_or_invent_missing_metadata():
    frame = pd.DataFrame({
        "NU": [-1e-30, 1e-30, -1e-30, -1e-30],
        "MU": [-4e-37, -4e-37, 4e-37, -4e-37],
        "boundary_coefficient": [1e-24, 1e-24, 1e-24, np.nan],
    })
    normalized, _ = phase_coordinates(frame, "normalized")
    assert np.isfinite(normalized.loc[0, "phase_x"])
    assert normalized.loc[1:, "phase_x"].isna().all()
    squared, _ = phase_coordinates(frame, "squared")
    assert squared.loc[[1, 2], "phase_x"].isna().all()
    assert squared.loc[3, "phase_x"] == 1
    raw, _ = phase_coordinates(frame, "raw")
    np.testing.assert_allclose(raw["phase_x"], [-1, 1, -1, -1])
    assert np.isfinite(raw[["x_low", "x_high", "y_low", "y_high"]]).all().all()


def test_discrete_onset_is_neutral_and_restricting_modes_changes_prediction():
    pairs, prepared = demo_sweep()
    pair = pairs[6]  # μ=-4.5e-37, |ν|=1.2 |ν_c|.
    data = prepared[pair["run_id"]]
    arguments = (pair["NU"], pair["MU"], data["length"], data["boundary_coefficient"])
    broad = predict_selected_modes(*arguments, 1, 100, mobility=data["theory_mobility"])
    narrow = predict_selected_modes(*arguments, 14, 14, mobility=data["theory_mobility"])
    assert broad["growing_modes"] > 0
    assert narrow["growing_modes"] == 0
    assert np.isnan(narrow["fastest_wavelength_cm"])
    assert narrow["onset_nu"] < broad["onset_nu"]
    continuous_onset = -np.sqrt(data["boundary_coefficient"] * -pair["MU"])
    assert broad["onset_nu"] <= continuous_onset
    k = np.pi * 14 / data["length"]
    assert narrow["onset_nu"] == pytest.approx(
        pair["MU"] * k**2 - data["boundary_coefficient"] / (4 * k**2), rel=1e-12, abs=0,
    )
    neutral = predict_selected_modes(
        narrow["onset_nu"], *arguments[1:], 14, 14, mobility=data["theory_mobility"],
    )
    assert neutral["max_rate"] == 0
    assert neutral["growing_modes"] == 0
    growing = predict_selected_modes(
        1.01 * narrow["onset_nu"], *arguments[1:], 14, 14, mobility=data["theory_mobility"],
    )
    assert growing["fastest_mode"] == 14
    assert growing["fastest_wavelength_cm"] == pytest.approx(1.0)


def test_selected_mode_predictions_match_analytic_synthetic_growth_rates():
    pairs, prepared = demo_sweep()
    for pair in pairs:
        data = prepared[pair["run_id"]]
        result = analyze_window(data, 0, 5, 1, 100)
        table = result["table"]
        k = np.pi * table["mode"].to_numpy() / data["length"]
        expected = (-2e-5 * 30 / data["length"]
                    - 7.4e23 * .03 * pair["NU"] * k**2
                    + 7.4e23 * .03 * pair["MU"] * k**4)
        np.testing.assert_allclose(table["theory_rate_s_inv"], expected, atol=1e-12)
        summary = result["summary"]
        if summary["theory_growing_modes"]:
            assert summary["theory_fastest_mode"] == summary["fastest_mode"]
            assert summary["theory_fastest_wavelength_cm"] == pytest.approx(summary["fastest_wavelength_cm"])
        else:
            assert np.isnan(summary["theory_fastest_wavelength_cm"])
    # A selection containing only the mean has no wavelength or onset prediction.
    assert np.isnan(predict_selected_modes(-1e-30, -4e-37, .07, 1e-24, 0, 0)["onset_nu"])


@pytest.mark.parametrize("coordinates", ["raw", "squared", "normalized"])
def test_overlay_curves_obey_onset_and_wavelength_equations_in_every_view(coordinates):
    pairs, prepared = demo_sweep()
    summary = pd.DataFrame([
        dict(pair, boundary_coefficient=prepared[pair["run_id"]]["boundary_coefficient"],
             dct_length_m=prepared[pair["run_id"]]["length"])
        for pair in pairs
    ])
    frame, _ = phase_coordinates(summary, coordinates)
    curves = theory_overlays(frame, coordinates, 1, 100, [.3, .5, .7, 1.0])
    assert set(curves["kind"]) == {"onset", "wavelength"}
    assert {"Continuum onset", "Selected-mode onset"} <= set(curves["curve"])
    coefficient = curves["boundary_coefficient"].to_numpy()
    mu = curves["phase_y"].to_numpy() * (1e-37 if coordinates == "raw" else -1e-37)
    nu = (curves["phase_x"].to_numpy() * 1e-30 if coordinates == "raw" else
          -np.sqrt(curves["phase_x"].to_numpy()) * 1e-30 if coordinates == "squared" else
          -curves["phase_x"].to_numpy() * np.sqrt(coefficient * -mu))
    modes = np.arange(1, 101)
    k2 = (np.pi * modes / .07) ** 2
    selected_onset = np.max(mu[:, None] * k2 - coefficient[:, None] / (4 * k2), axis=1)
    mask = curves["curve"] == "Selected-mode onset"
    np.testing.assert_allclose(nu[mask], selected_onset[mask], rtol=1e-12, atol=0)
    mask = curves["kind"] == "wavelength"
    k = 2 * np.pi / (curves.loc[mask, "wavelength_cm"].to_numpy() / 100)
    np.testing.assert_allclose(nu[mask], 2 * mu[mask] * k**2, rtol=1e-12, atol=0)
    assert np.all(nu[mask] < selected_onset[mask])
    assert curves["phase_x"].between(frame["x_low"].min(), frame["x_high"].max()).all()
    # No 1 cm or 0.3 cm growing contour is predicted inside this synthetic view.
    assert not set(curves.loc[mask, "wavelength_cm"]) & {.3, 1.0}
