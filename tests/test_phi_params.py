from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import h5py
from pydantic import ValidationError

from red_patterns.models import (
    GaussianPhiParams,
    HomogeneousPhiParams,
    PHI_PARAMS_ADAPTER,
    PhiGenerateParams,
    SingleBinPhiParams,
)
from red_patterns.phi import (
    PHI_FIELD_TYPES,
    PhiConfig,
    compute_phi,
    write_phi_h5,
)
from red_patterns.sweep_jobs import PhiSweep
from red_patterns.types import PhiType

N = 256
BASE = {
    "psi_avg": 0.02,
    "N": N,
    "wing": 30,
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


class PhiConfigFromParamsTests(unittest.TestCase):
    def test_from_params_maps_wing_and_extras(self):
        for phi_type in PhiType:
            params = PHI_PARAMS_ADAPTER.validate_python(params_for(phi_type))
            cfg = PhiConfig.from_params(params, output_path="out.h5")
            self.assertEqual(cfg.phi_type, phi_type, msg=phi_type)
            self.assertEqual(cfg.N, N)
            self.assertEqual(cfg.wing_z, 30)
            self.assertEqual(cfg.wing_r, 30)
            self.assertEqual(cfg.psi_avg, BASE["psi_avg"])
            for key, value in PER_TYPE_ARGS[phi_type].items():
                self.assertEqual(getattr(cfg, key), value, msg=f"{phi_type}.{key}")

    def test_from_params_accepts_raw_dict(self):
        cfg = PhiConfig.from_params(params_for(PhiType.GAUSSIAN), output_path="out.h5")
        self.assertEqual(cfg.phi_type, PhiType.GAUSSIAN)


class PhiComputeTests(unittest.TestCase):
    def test_compute_and_write_phi_shape_and_attrs_for_each_type(self):
        for phi_type in PhiType:
            params = PHI_PARAMS_ADAPTER.validate_python(params_for(phi_type))
            cfg = PhiConfig.from_params(params, output_path="initial_phi.h5")
            result = compute_phi(cfg)
            self.assertEqual(result.phi_values.shape, (N, N), msg=phi_type)
            with tempfile.TemporaryDirectory() as tmpdir:
                out = write_phi_h5(Path(tmpdir) / "phi.h5", result, cfg)
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


if __name__ == "__main__":
    unittest.main()