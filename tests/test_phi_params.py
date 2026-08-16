from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import h5py
import numpy as np
from pydantic import ValidationError

from red_patterns.models import (
    GaussianPhiParams,
    HomogeneousPhiParams,
    PHI_PARAMS_ADAPTER,
    PerturbedSmoothHomogeneousPhiParams,
    PhiGenerateParams,
    SingleBinPhiParams,
    SingleModeSmoothHomogeneousPhiParams,
)
from red_patterns.phi import (
    PHI_FIELD_TYPES,
    build_export_parser,
    phi_field_from_params,
    phi_field_from_ui,
    make_phi_ui,
    validate_export_namespace,
)
from red_patterns.sweep_jobs import PhiSweep
from red_patterns.types import PhiType

N = 256
BASE = {
    "psi_avg": 0.02,
    "N": N,
    "wing_z": 30,
    "wing_r": 30,
    "rho_center": 1100.0,
    "rho_span": 30.0,
    "dz": 0.000267651,
}

PER_TYPE_ARGS = {
    PhiType.GAUSSIAN: {"gaussian_mu": 1100.0, "gaussian_sigma": 4.0},
    PhiType.GAUSSIAN_BLOB: {
        "gaussian_mu": 1100.0,
        "gaussian_sigma": 4.0,
        "gaussian_blob_mu_z": 0.035,
        "gaussian_blob_sigma_z": 0.01,
    },
    PhiType.HOMOGENEOUS: {},
    PhiType.SMOOTH_HOMOGENEOUS: {"rho_range": 5.0},
    PhiType.PERTURBED_SMOOTH_HOMOGENEOUS: {
        "rho_range": 5.0,
        "seed": 3,
        "amplitude": 1e-3,
    },
    PhiType.SINGLE_MODE_SMOOTH_HOMOGENEOUS: {
        "rho_range": 5.0,
        "amplitude": 1e-3,
        "mode_number": 7,
    },
    PhiType.SINGLE_BIN: {"single_bin_idx": 100},
}


def params_for(phi_type: PhiType) -> dict:
    return dict(BASE, phi_type=phi_type.value, **PER_TYPE_ARGS[phi_type])


class AdapterMixin:
    def assert_valid(self, payload: dict):
        return PHI_PARAMS_ADAPTER.validate_python(payload)


class PhiParamsUnionTests(unittest.TestCase):
    def test_registry_has_all_phi_types(self):
        self.assertEqual(set(PHI_FIELD_TYPES), set(PhiType))

    def test_each_type_round_trips_through_json(self):
        for phi_type in PhiType:
            payload = params_for(phi_type)
            model = PHI_PARAMS_ADAPTER.validate_python(payload)
            self.assertEqual(model.phi_type, phi_type, msg=phi_type)
            dumped = PHI_PARAMS_ADAPTER.dump_json(model)
            reloaded = PHI_PARAMS_ADAPTER.validate_json(dumped)
            self.assertEqual(reloaded.model_dump(), model.model_dump(), msg=phi_type)

    def test_union_selects_concrete_member(self):
        self.assertIsInstance(
            PHI_PARAMS_ADAPTER.validate_python(params_for(PhiType.GAUSSIAN)),
            GaussianPhiParams,
        )
        self.assertIsInstance(
            PHI_PARAMS_ADAPTER.validate_python(params_for(PhiType.HOMOGENEOUS)),
            HomogeneousPhiParams,
        )
        self.assertIsInstance(
            PHI_PARAMS_ADAPTER.validate_python(params_for(PhiType.SINGLE_BIN)),
            SingleBinPhiParams,
        )
        self.assertIsInstance(
            PHI_PARAMS_ADAPTER.validate_python(
                params_for(PhiType.PERTURBED_SMOOTH_HOMOGENEOUS)
            ),
            PerturbedSmoothHomogeneousPhiParams,
        )
        self.assertIsInstance(
            PHI_PARAMS_ADAPTER.validate_python(
                params_for(PhiType.SINGLE_MODE_SMOOTH_HOMOGENEOUS)
            ),
            SingleModeSmoothHomogeneousPhiParams,
        )

    def test_concrete_model_supplies_its_discriminator_default(self):
        params = GaussianPhiParams(
            psi_avg=0.02,
            gaussian_mu=1100.0,
            gaussian_sigma=4.0,
        )
        self.assertEqual(params.phi_type, PhiType.GAUSSIAN)

    def test_rejects_param_mismatched_to_type(self):
        payload = params_for(PhiType.HOMOGENEOUS)
        payload["gaussian_mu"] = 1100.0
        with self.assertRaises(ValidationError):
            PHI_PARAMS_ADAPTER.validate_python(payload)

    def test_rejects_missing_required_per_type_param(self):
        payload = params_for(PhiType.SINGLE_BIN)
        del payload["single_bin_idx"]
        with self.assertRaises(ValidationError):
            PHI_PARAMS_ADAPTER.validate_python(payload)

    def test_rejects_missing_phi_type(self):
        payload = params_for(PhiType.GAUSSIAN)
        del payload["phi_type"]
        with self.assertRaises(ValidationError):
            PHI_PARAMS_ADAPTER.validate_python(payload)

    def test_rejects_unknown_phi_type(self):
        payload = params_for(PhiType.GAUSSIAN)
        payload["phi_type"] = "bogus"
        with self.assertRaises(ValidationError):
            PHI_PARAMS_ADAPTER.validate_python(payload)


class PhiFieldFromParamsTests(unittest.TestCase):
    def test_from_params_maps_wing_and_extras(self):
        for phi_type in PhiType:
            params = PHI_PARAMS_ADAPTER.validate_python(params_for(phi_type))
            field = phi_field_from_params(params)
            self.assertEqual(field.phi_type, phi_type, msg=phi_type)
            self.assertEqual(field.N, N)
            self.assertEqual(field.wing_z, 30)
            self.assertEqual(field.wing_r, 30)
            self.assertEqual(field.psi_avg, BASE["psi_avg"])
            for key, value in PER_TYPE_ARGS[phi_type].items():
                self.assertEqual(getattr(field, key), value, msg=f"{phi_type}.{key}")

    def test_from_params_accepts_raw_dict(self):
        field = phi_field_from_params(params_for(PhiType.GAUSSIAN))
        self.assertEqual(field.phi_type, PhiType.GAUSSIAN)


class PhiComputeTests(unittest.TestCase):
    def test_compute_and_write_phi_shape_and_attrs_for_each_type(self):
        for phi_type in PhiType:
            params = PHI_PARAMS_ADAPTER.validate_python(params_for(phi_type))
            field = phi_field_from_params(params)
            result = field.compute()
            self.assertEqual(result.phi_values.shape, (N, N), msg=phi_type)
            with tempfile.TemporaryDirectory() as tmpdir:
                out = field.write_phi_h5(Path(tmpdir) / "phi.h5", result)
                with h5py.File(out, "r") as f:
                    self.assertEqual(f["phi/values"].shape, (N, N), msg=phi_type)
                    self.assertEqual(f["phi"].attrs["phi_type"], phi_type.label)
                    if phi_type == PhiType.SINGLE_BIN:
                        self.assertEqual(
                            f["phi"].attrs["single_bin_idx"],
                            PER_TYPE_ARGS[PhiType.SINGLE_BIN]["single_bin_idx"],
                        )
                    if phi_type == PhiType.GAUSSIAN:
                        self.assertEqual(
                            f["phi"].attrs["gaussian_sigma"],
                            PER_TYPE_ARGS[PhiType.GAUSSIAN]["gaussian_sigma"],
                        )
                    if phi_type == PhiType.SINGLE_MODE_SMOOTH_HOMOGENEOUS:
                        self.assertEqual(
                            f["phi"].attrs["mode_number"],
                            PER_TYPE_ARGS[phi_type]["mode_number"],
                        )

    def test_single_mode_has_the_requested_active_domain_cosine(self):
        base = phi_field_from_params(params_for(PhiType.SMOOTH_HOMOGENEOUS)).compute()
        field = phi_field_from_params(
            params_for(PhiType.SINGLE_MODE_SMOOTH_HOMOGENEOUS)
        )
        result = field.compute()

        active = slice(field.wing_z, field.N - field.wing_z)
        z_active = result.z[active]
        x = (z_active - z_active[0]) / (z_active[-1] - z_active[0])
        raw_multiplier = 1.0 + field.amplitude * np.cos(
            np.pi * field.mode_number * x
        )
        expected_multiplier = raw_multiplier / np.mean(raw_multiplier)
        np.testing.assert_allclose(
            result.phi_values[:, active],
            base.phi_values[:, active] * expected_multiplier[np.newaxis, :],
        )
        np.testing.assert_allclose(result.phi_values[:, : field.wing_z], 0.0)
        np.testing.assert_allclose(result.phi_values[:, field.N - field.wing_z :], 0.0)
        self.assertAlmostEqual(
            result.phi_values.sum(axis=0)[active].mean(), field.psi_avg
        )

    def test_single_mode_zero_is_accepted(self):
        payload = params_for(PhiType.SINGLE_MODE_SMOOTH_HOMOGENEOUS)
        payload["mode_number"] = 0
        self.assertEqual(PHI_PARAMS_ADAPTER.validate_python(payload).mode_number, 0)


class PhiSweepRowsTests(unittest.TestCase):
    def test_all_phi_types_sweep_to_valid_params(self):
        sweep = PhiSweep(phi_type=list(PhiType))
        for row in sweep.rows():
            PHI_PARAMS_ADAPTER.validate_python(row)

    def test_sweep_emits_gaussian_params_only_for_gaussian(self):
        sweep = PhiSweep(phi_type=[PhiType.HOMOGENEOUS])
        rows = sweep.rows()
        self.assertEqual(len(rows), 1)
        self.assertNotIn("gaussian_mu", rows[0])


class PhiCliTests(unittest.TestCase):
    def test_shared_amplitude_option_is_registered_once(self):
        parser = build_export_parser()
        amplitude_actions = [
            action for action in parser._actions if "--amplitude" in action.option_strings
        ]
        self.assertEqual(len(amplitude_actions), 1)

    def test_registry_builds_perturbed_cli(self):
        parser = build_export_parser()
        args = parser.parse_args(
            [
                "--output", "phi.h5",
                "--phi-type", PhiType.PERTURBED_SMOOTH_HOMOGENEOUS.value,
                "--psi-avg", "0.02",
                "--rho-range", "5",
                "--seed", "4",
                "--amplitude", "0.001",
            ]
        )
        field = validate_export_namespace(parser, args)
        self.assertEqual(field.phi_type, PhiType.PERTURBED_SMOOTH_HOMOGENEOUS)
        self.assertEqual(field.seed, 4)

    def test_registry_builds_single_mode_cli(self):
        parser = build_export_parser()
        args = parser.parse_args(
            [
                "--output", "phi.h5",
                "--phi-type", PhiType.SINGLE_MODE_SMOOTH_HOMOGENEOUS.value,
                "--psi-avg", "0.02",
                "--rho-range", "5",
                "--amplitude", "0.001",
                "--mode-number", "7",
            ]
        )
        field = validate_export_namespace(parser, args)
        self.assertEqual(field.phi_type, PhiType.SINGLE_MODE_SMOOTH_HOMOGENEOUS)
        self.assertEqual(field.mode_number, 7)

    def test_cli_rejects_an_option_from_another_type(self):
        parser = build_export_parser()
        args = parser.parse_args(
            [
                "--output", "phi.h5",
                "--phi-type", PhiType.HOMOGENEOUS.value,
                "--psi-avg", "0.02",
                "--seed", "4",
            ]
        )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                validate_export_namespace(parser, args)


class PhiUiTests(unittest.TestCase):
    def test_nested_ui_keeps_common_and_variant_values_separate(self):
        ui = make_phi_ui()
        self.assertEqual(set(ui.value), {"common", "phi_type", "variants"})
        self.assertEqual(set(ui.value["common"]), {"psi_avg", "N", "wing_z", "wing_r"})
        self.assertEqual(set(ui.value["variants"]), {phi_type.value for phi_type in PhiType})

        field = phi_field_from_ui(ui.value)
        self.assertEqual(field.phi_type, PhiType.GAUSSIAN)


if __name__ == "__main__":
    unittest.main()
