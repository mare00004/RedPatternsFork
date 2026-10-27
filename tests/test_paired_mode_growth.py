"""Validate central differences, mode purity and early-window rejection."""

import numpy as np
import pytest
from scipy.fft import dct, idct

from red_patterns.paired_mode_growth import early_prefix, paired_response
from red_patterns.phi import phi_field_from_params
from red_patterns.models import PHI_PARAMS_ADAPTER


def test_displaced_ridge_pair_dominantly_excites_chosen_dct_mode():
    fields = []
    epsilon = 1e-7
    for amplitude in (epsilon, -epsilon):
        params = PHI_PARAMS_ADAPTER.validate_python(dict(
            phi_type="single_mode_linear_full_ridge", N=64, wing_z=0, wing_r=0,
            rho_center=1100., rho_span=30., dz=0.01, psi_avg=0.02,
            amplitude=amplitude, mode_number=7,
        ))
        fields.append(phi_field_from_params(params).compute().phi_values)
    response = (fields[0] - fields[1]) / (2 * epsilon)
    c = dct(response.sum(axis=0), type=2, norm="ortho")
    assert abs(c[7]) > 0.01
    # Finite conservative remapping can generate harmonics; its infinitesimal
    # marginal tends to -C * xi'(z), with the expected negative cosine sign.
    expected = -0.02 * (2 / 0.01) * np.sin(np.pi * 7 / (2 * 64)) * np.sqrt(64 / 2)
    assert c[7] == pytest.approx(expected, rel=1e-4)
    assert np.max(np.abs(np.delete(c, 7))) < 1e-4 * abs(c[7])


def make_pair():
    time = np.linspace(0, 20, 101)
    coeff = np.zeros((101, 32))
    coeff[:, 7] = np.exp(0.12 * time)
    response = idct(coeff, type=2, norm="ortho", axis=1)
    common = dict(time=time, z=np.arange(32.), rho=np.arange(32.),
                  model=dict(modelType="TAYL", gradientType="LINEAR_FULL", alpha=0., beta=1.),
                  variant=dict(NU=1., MU=1.), run=dict(DT=0.001), kernel=None)
    return dict(common, psi=1. + 0.001*response), dict(common, psi=1. - 0.001*response), coeff


def test_signed_central_difference_and_rate():
    plus, minus, expected = make_pair()
    _, coeff, floor = paired_response(plus, minus, 0.001)
    np.testing.assert_allclose(coeff, expected, atol=1e-12)
    result = early_prefix(coeff[:, 7], plus["time"], floor=floor)
    assert result["count"] == 101
    assert result["fit"].rate == pytest.approx(0.12, abs=1e-12)
    _, reversed_coeff, _ = paired_response(minus, plus, 0.001)
    np.testing.assert_allclose(reversed_coeff, -coeff)


def test_reject_mismatched_time_and_dynamics():
    plus, minus, _ = make_pair()
    with pytest.raises(ValueError, match="saved times"):
        paired_response(plus, dict(minus, time=minus["time"] + 0.01), 0.001)
    with pytest.raises(ValueError, match="different dynamics"):
        paired_response(plus, dict(minus, variant=dict(NU=2., MU=1.)), 0.001)


@pytest.mark.parametrize("rate", [-0.12, 0.12])
def test_prefix_stops_before_bending_or_sign_change(rate):
    t = np.linspace(0, 20, 101)
    a = np.exp(rate * t)
    a[40:] *= np.exp(0.8 * (t[40:] - t[40]))
    result = early_prefix(a, t)
    assert 40 <= result["count"] < 50
    signed = np.exp(rate * t)
    signed[40:] *= -1
    result = early_prefix(signed, t)
    assert result["count"] == 40
    assert result["fit"].rate == pytest.approx(rate)


def test_no_late_search_and_no_floor_filling():
    t = np.arange(30.)
    a = np.exp(0.1*t)
    a[0] = 0
    assert early_prefix(a, t)["fit"] is None
    a[0] = 100
    assert early_prefix(a, t)["fit"] is None
