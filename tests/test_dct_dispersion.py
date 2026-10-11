"""Check modal wavelengths, signed rates, and conservative cutoff extraction."""

import h5py
import numpy as np
import pandas as pd
import pytest

from red_patterns.dct_dispersion import (
    analyze_modes,
    demo_pair,
    dispersion_features,
    read_run,
    validate_pair,
)


def test_known_cosine_dispersion_and_wavelengths():
    perturbed, equilibrium = demo_pair()
    time, coefficients, fits, table, length, _ = analyze_modes(
        perturbed, equilibrium, 40, 1, 9,
    )
    assert len(time) == 41
    assert coefficients.shape == (41, 64)
    assert np.all(table["status"] == "ok")
    modes = table["mode"].to_numpy()
    np.testing.assert_allclose(table["growth_rate_s^-1"], 0.16 - 0.02 * (modes - 5) ** 2, atol=1e-7)
    assert length == pytest.approx(0.064)
    np.testing.assert_allclose(table["wavelength_m"], 2 * 64 * 0.001 / modes)
    np.testing.assert_allclose(table["wavenumber_rad_m"], np.pi * modes / length)
    for mode, fit in zip(modes, fits):
        np.testing.assert_allclose(fit.fitted, coefficients[:, mode], atol=1e-11)
    fastest, crossings = dispersion_features(table)
    assert fastest["mode"] == 5
    assert fastest["wavelength_m"] == pytest.approx(0.0256)
    cutoff = crossings[crossings["kind"] == "short-wave cutoff"].iloc[0]
    assert (cutoff["mode_left"], cutoff["mode_right"]) == (7, 8)
    # Linear interpolation in k, not interpolation in wavelength.
    assert cutoff["wavelength_m"] == pytest.approx(0.128 / 7.8, rel=1e-6)
    assert len(crossings) == 2  # Also retain the long-wave edge between 2 and 3.


def test_irregular_original_times_and_inclusive_endpoint():
    perturbed, equilibrium = demo_pair()
    # Preserve analytic samples at irregular physical times by removing frames.
    take = np.r_[0, 1, 4, 8, 11, 15, 22, 29, 35, 40]
    perturbed = dict(perturbed, time=perturbed["time"][take], psi=perturbed["psi"][take])
    equilibrium = dict(equilibrium, time=np.array([0.0]), psi=equilibrium["psi"][:1])
    # Corrupt the later frame: it must not enter the chosen fit window.
    perturbed["psi"][-1] += 0.1 * np.arange(64)
    time, _, _, table, _, _ = analyze_modes(perturbed, equilibrium, 8, 5, 5)
    np.testing.assert_array_equal(time, perturbed["time"][:9])
    assert table.iloc[0]["growth_rate_s^-1"] == pytest.approx(0.16, abs=1e-7)


def test_mean_and_unexcited_modes_do_not_acquire_wavelength_features():
    perturbed, equilibrium = demo_pair()
    _, _, _, table, _, _ = analyze_modes(perturbed, equilibrium, 40, 0, 0)
    assert np.isinf(table.iloc[0]["wavelength_m"])
    assert np.isnan(table.iloc[0]["growth_rate_s^-1"])
    fastest, crossings = dispersion_features(table)
    assert fastest is None
    assert crossings.empty
    _, _, _, table, _, _ = analyze_modes(perturbed, equilibrium, 40, 20, 23)
    assert not np.any(table["status"] == "ok")


@pytest.mark.parametrize("modes,status", [([7, 9], ["ok", "ok"]), ([7, 8], ["ok", "poor single-exponential fit"])])
def test_no_cutoff_across_missing_or_rejected_modes(modes, status):
    table = pd.DataFrame({
        "mode": modes, "status": status, "growth_rate_s^-1": [0.1, -0.1],
        "wavenumber_rad_m": modes, "wavelength_m": 2 * np.pi / np.asarray(modes),
    })
    _, crossings = dispersion_features(table)
    assert crossings.empty


def test_all_stable_modes_have_no_fastest_growing_wavelength():
    perturbed, equilibrium = demo_pair()
    _, _, _, table, _, _ = analyze_modes(perturbed, equilibrium, 40, 8, 9)
    fastest, crossings = dispersion_features(table)
    assert fastest is None
    assert crossings.empty


def test_validation_rejects_grid_and_operator_mismatches():
    perturbed, equilibrium = demo_pair()
    with pytest.raises(ValueError, match="different dynamics"):
        validate_pair(perturbed, dict(equilibrium, variant={"NU": 2, "MU": 1}))
    with pytest.raises(ValueError, match="identical z grids"):
        validate_pair(perturbed, dict(equilibrium, z=equilibrium["z"] * 1.1))
    with pytest.raises(ValueError, match="five saved frames"):
        analyze_modes(perturbed, equilibrium, 3, 1, 9)


def test_saved_run_loading_preserves_small_perturbations(tmp_path):
    perturbed, _ = demo_pair()
    path = tmp_path / "run.h5"
    values = perturbed["psi"].copy()
    values[0, 0] += 1e-10  # Smaller than a float32 ulp at this equilibrium.
    with h5py.File(path, "w") as f:
        for key, value in (("fields/psi", values), ("time", perturbed["time"]),
                           ("coords/z", perturbed["z"]), ("coords/rho", perturbed["rho"])):
            f.create_dataset(key, data=value)
        for key, attrs in (("config/model", perturbed["model"]),
                           ("config/run", perturbed["run"]),
                           ("config/model/variant/tayl", perturbed["variant"])):
            f.require_group(key).attrs.update(attrs)
    loaded = read_run(path)
    assert loaded["psi"].dtype == np.float64
    np.testing.assert_array_equal(loaded["psi"], values)
    validate_pair(loaded, perturbed)
