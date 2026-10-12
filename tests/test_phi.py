"""Focused tests for registered initial phi fields."""

from __future__ import annotations

import unittest

import numpy as np
import h5py
import pytest
from pydantic import ValidationError
from scipy.fft import dct

from red_patterns.models import PHI_PARAMS_ADAPTER
from red_patterns.phi import (
    PHI_FIELD_TYPES,
    build_export_parser,
    phi_field_from_params,
)
from red_patterns.types import PhiType


COMMON_PARAMS = {
    "psi_avg": 0.02,
    "N": 32,
    "wing_z": 2,
    "wing_r": 2,
    "rho_center": 1100.0,
    "rho_span": 30.0,
    "dz": 0.01,
}


class SingleModeLinearFullRidgePhiTests(unittest.TestCase):
    def test_registered_payload_and_cli(self) -> None:
        payload = {
            **COMMON_PARAMS,
            "phi_type": PhiType.SINGLE_MODE_LINEAR_FULL_RIDGE.value,
            "amplitude": 0.1,
            "mode_number": 3,
        }
        params = PHI_PARAMS_ADAPTER.validate_python(payload)
        field = phi_field_from_params(params)

        self.assertIsInstance(field, PHI_FIELD_TYPES[params.phi_type])
        with self.assertRaises(ValidationError):
            PHI_PARAMS_ADAPTER.validate_python({**payload, "rho_range": 5.0})
        args = build_export_parser().parse_args(
            [
                "--output",
                "initial_phi.h5",
                "--phi-type",
                PhiType.SINGLE_MODE_LINEAR_FULL_RIDGE.value,
                "--psi-avg",
                "0.02",
                "--amplitude",
                "0.1",
                "--mode-number",
                "3",
            ]
        )
        self.assertEqual(args.mode_number, 3)

    def test_applies_cosine_to_linear_full_ridge_and_normalizes(self) -> None:
        base = phi_field_from_params(
            {**COMMON_PARAMS, "phi_type": PhiType.LINEAR_FULL_RIDGE.value}
        )
        perturbed = phi_field_from_params(
            {
                **COMMON_PARAMS,
                "phi_type": PhiType.SINGLE_MODE_LINEAR_FULL_RIDGE.value,
                "amplitude": 0.1,
                "mode_number": 3,
            }
        )
        result = perturbed.compute()
        rho, z = result.rho, result.z
        raw_base = base.build(rho, z)
        raw_perturbed = perturbed.build(rho, z)
        active = slice(perturbed.wing_z, perturbed.N - perturbed.wing_z)
        x = (z[active] - z[active][0]) / (z[active][-1] - z[active][0])
        multiplier = 1.0 + perturbed.amplitude * np.cos(
            np.pi * perturbed.mode_number * x
        )

        self.assertTrue(
            np.array_equal(raw_base != 0.0, raw_perturbed != 0.0)
        )
        np.testing.assert_allclose(
            raw_perturbed[:, active], raw_base[:, active] * multiplier[np.newaxis, :]
        )

        phi_values = result.phi_values
        self.assertTrue(np.all(phi_values[: perturbed.wing_r, :] == 0.0))
        self.assertTrue(np.all(phi_values[:, : perturbed.wing_z] == 0.0))
        active_average = phi_values[:, active].sum() / (
            perturbed.N - 2 * perturbed.wing_z
        )
        self.assertAlmostEqual(active_average, perturbed.psi_avg)


class PerturbedLinearFullGaussianRidgePhiTests(unittest.TestCase):
    def test_seeded_conservative_displacement(self) -> None:
        base_payload = {
            **COMMON_PARAMS,
            "phi_type": PhiType.PERTURBED_LINEAR_FULL_GAUSSIAN_RIDGE.value,
            "gaussian_mu": 1100.0,
            "gaussian_sigma": 4.0,
            "epsilon": 0.0,
            "seed": 7,
            "mode_min": 1,
            "mode_max": 4,
        }
        base = phi_field_from_params(base_payload)
        perturbed = phi_field_from_params({**base_payload, "epsilon": 1e-6})
        rho, z = base.compute().rho, base.compute().z
        raw_base = base.build(rho, z)
        raw_perturbed = perturbed.build(rho, z)

        self.assertFalse(np.array_equal(raw_base, raw_perturbed))
        self.assertTrue(np.all(raw_perturbed >= 0.0))
        np.testing.assert_allclose(raw_perturbed.sum(axis=1), raw_base.sum(axis=1))

        result = perturbed.compute()
        active = slice(perturbed.wing_z, perturbed.N - perturbed.wing_z)
        self.assertTrue(np.all(result.phi_values[:, : perturbed.wing_z] == 0.0))
        self.assertTrue(np.all(result.phi_values[:, perturbed.N - perturbed.wing_z :] == 0.0))
        self.assertAlmostEqual(
            result.phi_values[:, active].sum() / (perturbed.N - 2 * perturbed.wing_z),
            perturbed.psi_avg,
        )

    def test_rejects_inverted_mode_range(self) -> None:
        with self.assertRaises(ValidationError):
            PHI_PARAMS_ADAPTER.validate_python(
                {
                    **COMMON_PARAMS,
                    "phi_type": PhiType.PERTURBED_LINEAR_FULL_GAUSSIAN_RIDGE.value,
                    "gaussian_mu": 1100.0,
                    "gaussian_sigma": 4.0,
                    "epsilon": 1e-6,
                    "seed": 0,
                    "mode_min": 4,
                    "mode_max": 1,
                }
            )


class PerturbedLinearFullRidgePhiTests(unittest.TestCase):
    def test_constant_ridge_remap_is_nonnegative_and_row_conservative(self) -> None:
        payload = {
            **COMMON_PARAMS,
            "phi_type": PhiType.PERTURBED_LINEAR_FULL_RIDGE.value,
            "epsilon": 1e-6,
            "seed": 7,
            "mode_min": 1,
            "mode_max": 4,
        }
        field = phi_field_from_params(payload)
        base = phi_field_from_params(
            {**COMMON_PARAMS, "phi_type": PhiType.LINEAR_FULL_RIDGE.value}
        )
        rho, z = field.compute().rho, field.compute().z
        raw_base = base.build(rho, z)
        raw_perturbed = field.build(rho, z)

        self.assertFalse(np.array_equal(raw_base, raw_perturbed))
        self.assertTrue(np.all(raw_perturbed >= 0.0))
        np.testing.assert_allclose(raw_perturbed.sum(axis=1), raw_base.sum(axis=1))


@pytest.mark.parametrize("mode", [1, 7, 50])
def test_displacement_index_excites_matching_dct_cosine(mode):
    params = {
        **COMMON_PARAMS, "N": 256, "dz": 0.07 / 256,
        "wing_z": 0, "wing_r": 0,
        "phi_type": PhiType.PERTURBED_LINEAR_FULL_RIDGE.value,
        "epsilon": 1e-8, "seed": 0, "mode_min": mode, "mode_max": mode,
    }
    field = phi_field_from_params(params)
    result = field.compute()
    baseline = phi_field_from_params({**params, "epsilon": 0}).compute()
    coefficients = dct((result.phi_values - baseline.phi_values).sum(axis=0), norm="ortho")
    assert np.argmax(np.abs(coefficients)) == mode
    # Linearized conservative face displacement gives this cell-averaged cosine.
    b = np.random.default_rng(params["seed"]).normal()
    expected = (-params["psi_avg"] * params["epsilon"] * b
                * 2 * np.sin(np.pi * mode / (2 * params["N"])) / params["dz"]
                * np.sqrt(params["N"] / 2))
    assert coefficients[mode] == pytest.approx(expected, rel=2e-4)
    assert np.linalg.norm(np.delete(coefficients, mode)) < 1e-3 * abs(expected)
    assert np.all(result.phi_values >= 0)
    np.testing.assert_allclose(result.phi_values.sum(axis=1), baseline.phi_values.sum(axis=1), rtol=1e-11)


@pytest.mark.parametrize("phi_type,extras", [
    (PhiType.PERTURBED_LINEAR_FULL_RIDGE, {}),
    (PhiType.PERTURBED_LINEAR_FULL_GAUSSIAN_RIDGE, {"gaussian_mu": 1100., "gaussian_sigma": 4.}),
])
def test_displacement_folding_check_and_export_use_pi_basis(tmp_path, phi_type, extras):
    mode, seed = 3, 0
    length = COMMON_PARAMS["N"] * COMMON_PARAMS["dz"]
    b = abs(np.random.default_rng(seed).normal())
    critical_epsilon = length / (np.pi * mode * b)
    params = {
        **COMMON_PARAMS, **extras, "phi_type": phi_type.value,
        "wing_z": 0, "wing_r": 0,
        "epsilon": 0.75 * critical_epsilon,
        "seed": seed, "mode_min": mode, "mode_max": mode,
    }
    field = phi_field_from_params(params)
    assert not field.validate()  # The old 2*pi check would incorrectly reject this.
    result = field.compute()
    baseline = phi_field_from_params({**params, "epsilon": 0}).compute()
    np.testing.assert_allclose(result.phi_values.sum(axis=1), baseline.phi_values.sum(axis=1), rtol=1e-12)
    field.write_phi_h5(tmp_path / "phi.h5", result)
    with h5py.File(tmp_path / "phi.h5") as handle:
        assert handle["phi"].attrs["displacement_basis"] == "sin(n*pi*z/L)"
    folded = phi_field_from_params({**params, "epsilon": 1.1 * critical_epsilon})
    with pytest.raises(ValueError, match="folding"):
        folded.compute()


if __name__ == "__main__":
    unittest.main()
