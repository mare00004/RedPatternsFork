"""Focused tests for registered initial phi fields."""

from __future__ import annotations

import unittest

import numpy as np
from pydantic import ValidationError

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


if __name__ == "__main__":
    unittest.main()
