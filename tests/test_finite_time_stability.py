"""Analytic checks of adaptive paired-mode sweep diagnostics."""

import numpy as np
import pandas as pd
import pytest
from scipy.fft import idct

from red_patterns.dct_dispersion import demo_pair
from red_patterns.paired_mode_growth import early_prefix
from red_patterns.finite_time_stability import (
    analyze_adaptive, demo_sweep, phase_coordinates, prepare_pair,
    predict_selected_modes, theory_overlays,
)


def test_fastest_paired_mode_wavelength_and_analytic_rates():
    p, e = demo_pair()
    result = analyze_adaptive(prepare_pair(p, e), 1, 9)
    summary = result["summary"]
    assert summary["fastest_mode"] == 5
    assert summary["fastest_wavelength_cm"] == pytest.approx(2.56)
    assert summary["fastest_growth_s_inv"] == pytest.approx(0.16, abs=1e-9)
    assert summary["max_growth_s_inv"] == pytest.approx(0.16, abs=1e-9)
    rates = 0.16 - 0.02 * (np.arange(1, 10) - 5) ** 2
    np.testing.assert_allclose(result["table"]["growth_s_inv"], rates, atol=1e-9)
    assert (result["table"]["fit_samples"] == len(p["time"])).all()
    assert (result["table"]["fit_end_s"] == p["time"][-1]).all()


def test_moving_reference_cancels_before_adaptive_modal_fits():
    p, e = demo_pair()
    drift_coefficients = np.zeros_like(e["psi"])
    drift_coefficients[:, 2] = 1e-3 * np.exp(0.3 * p["time"])
    drift = idct(drift_coefficients, type=2, norm="ortho", axis=1)
    moving_p = dict(p, psi=p["psi"] + drift)
    moving_e = dict(e, psi=e["psi"] + drift)
    original = analyze_adaptive(prepare_pair(p, e), 1, 9)
    moving = analyze_adaptive(prepare_pair(moving_p, moving_e), 1, 9)
    np.testing.assert_allclose(moving["coefficients"], original["coefficients"], atol=1e-15)
    np.testing.assert_allclose(moving["table"]["growth_s_inv"], original["table"]["growth_s_inv"], atol=1e-8)
    assert moving["summary"]["fastest_mode"] == 5
    assert moving["summary"]["reference_drift"] > 1e-6


@pytest.mark.parametrize("mode, expected", [(5, 0.16), (9, -0.16)])
def test_single_mode_irregular_saved_times(mode, expected):
    p, e = demo_pair()
    take = [0, 1, 4, 7, 10, 17, 25, 40, 60]
    p = dict(p, time=p["time"][take], psi=p["psi"][take])
    e = dict(e, time=e["time"][take], psi=e["psi"][take])
    result = analyze_adaptive(prepare_pair(p, e), mode, mode)
    assert result["summary"]["max_growth_s_inv"] == pytest.approx(expected, abs=1e-9)
    np.testing.assert_array_equal(result["time"], p["time"])
    if expected < 0:
        assert np.isnan(result["summary"]["fastest_wavelength_cm"])
        assert result["summary"]["growth_status"] == "selected modes decay"


def test_each_mode_uses_the_local_notebooks_adaptive_prefix():
    p, e = demo_pair()
    data = prepare_pair(p, e)
    time = data["time"]
    data["difference"][40:, 5] *= np.exp(0.8 * (time[40:] - time[40]))
    data["difference"][80:, 6] *= -1
    for tolerance, minimum in [(0.05, 8), (0.01, 5)]:
        result = analyze_adaptive(data, 5, 6, min_samples=minimum, log_tolerance=tolerance)
        for index, mode in enumerate([5, 6]):
            expected = early_prefix(data["difference"][:, mode], time, floor=data["floor"],
                                    min_samples=minimum, log_tolerance=tolerance)
            row = result["table"].iloc[index]
            assert row["fit_samples"] == expected["count"]
            assert row["fit_end_s"] == time[expected["count"] - 1]
            assert row["growth_s_inv"] == expected["fit"].rate
            assert row["max_abs_log_residual"] <= tolerance
        assert 40 <= result["table"].iloc[0]["fit_samples"] < 50
        assert result["table"].iloc[1]["fit_samples"] == 80


def test_initial_failure_and_subprecision_are_not_replaced_or_searched_past():
    p, e = demo_pair()
    data = prepare_pair(p, e)
    for initial in [0, data["floor"] / 10, 100 * data["difference"][0, 5]]:
        coefficients = data["difference"].copy()
        coefficients[0, 5] = initial
        result = analyze_adaptive(dict(data, difference=coefficients), 5, 5)
        assert result["fits"] == [None]
        assert result["table"].iloc[0]["fit_samples"] == 0
        assert np.isnan(result["summary"]["max_growth_s_inv"])
        assert result["summary"]["growth_status"] == "no resolved modal rate"


def test_unresolved_modes_do_not_establish_selected_band_decay():
    p, e = demo_pair()
    data = prepare_pair(p, e)
    data["difference"][:, 10] = 0
    result = analyze_adaptive(data, 9, 10)
    assert result["summary"]["max_growth_s_inv"] == pytest.approx(-0.16)
    assert result["summary"]["resolved_modes"] == 1
    assert result["summary"]["unresolved_modes"] == 1
    assert result["summary"]["growth_status"] == "resolved modes decay; others unresolved"


def test_mode_zero_is_inspectable_but_excluded_from_growth_maps():
    p, e = demo_pair()
    data = prepare_pair(p, e)
    data["difference"][:, 0] = 1e-5 * np.exp(data["time"])
    result = analyze_adaptive(data, 0, 9)
    assert result["summary"]["max_growth_mode"] == 5
    assert np.isnan(result["table"].iloc[0]["wavelength_cm"])
    result = analyze_adaptive(data, 0, 0)
    assert np.isnan(result["summary"]["max_growth_s_inv"])


def test_fewer_than_minimum_samples_has_no_adaptive_fit():
    p, e = demo_pair()
    p = dict(p, time=p["time"][:5], psi=p["psi"][:5])
    e = dict(e, time=e["time"][:5], psi=e["psi"][:5])
    result = analyze_adaptive(prepare_pair(p, e), 5, 5)
    assert result["fits"] == [None]
    assert result["table"].iloc[0]["fit_samples"] == 0
    assert np.isnan(result["summary"]["max_growth_s_inv"])


def test_mismatched_pair_and_invalid_selections_rejected():
    p, e = demo_pair()
    with pytest.raises(ValueError, match="identical saved-time"):
        prepare_pair(p, dict(e, time=e["time"] + 0.001))
    with pytest.raises(ValueError, match="different dynamics"):
        prepare_pair(p, dict(e, variant={"NU": -1e-30, "MU": -1e-37}))
    with pytest.raises(ValueError, match="DT differs"):
        prepare_pair(p, dict(e, run=dict(DT=123)))
    with pytest.raises(ValueError, match="Missing rho"):
        prepare_pair(p, dict(e, rho=None))
    data = prepare_pair(p, e)
    for low, high in [(-1, 5), (5, 64), (6, 5), (1.5, 5)]:
        with pytest.raises(ValueError, match="mode range"):
            analyze_adaptive(data, low, high)
    for options in [dict(min_samples=4), dict(log_tolerance=0), dict(log_tolerance=np.nan)]:
        with pytest.raises(ValueError, match="at least five samples"):
            analyze_adaptive(data, 1, 9, **options)


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
        result = analyze_adaptive(data, 1, 100)
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
