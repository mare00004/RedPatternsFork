from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from pydantic import TypeAdapter, ValidationError

from red_patterns.models import (
    BaseRun,
    ConvRun,
    GenerateParams,
    RunPayload,
    TaylorRun,
    run_payload_adapter,
)
from red_patterns.types import Gradient, Variant


def taylor_payload() -> dict:
    return {
        "variant": "taylor",
        "N": 256,
        "T": 1.0,
        "DT": 0.1,
        "storeTime": 1.0,
        "gradient": "linear",
        "phi": {
            "mode": "generate",
            "params": {
                "phi_type": "gaussian",
                "psi_avg": 0.02,
                "N": 256,
                "wing": 30,
                "rho_center": 1100.0,
                "rho_span": 30.0,
                "dz": 0.000267651,
                "gaussian_mu": 1100.0,
                "gaussian_sigma": 4.0,
            },
        },
        "NU": -1e-30,
        "MU": -1e-37,
    }


class GenerateParamsTests(unittest.TestCase):
    def test_defaults(self):
        params = GenerateParams()
        self.assertEqual(params.mode, "generate")
        self.assertEqual(params.params, {})


class DiscriminatedUnionTests(unittest.TestCase):
    def test_validates_taylor_by_variant(self):
        run = run_payload_adapter.validate_python(taylor_payload())
        self.assertIsInstance(run, TaylorRun)
        self.assertEqual(run.variant, Variant.TAYLOR)
        self.assertEqual(run.run_id, "")
        self.assertEqual(run.gradient, Gradient.LINEAR)

    def test_type_adapter_rejects_bad_discriminator(self):
        payload = taylor_payload()
        payload["variant"] = "bogus"
        with self.assertRaises(ValidationError):
            run_payload_adapter.validate_python(payload)

    def test_rejects_extra_fields(self):
        payload = taylor_payload()
        payload["bogus"] = 42
        with self.assertRaises(ValidationError):
            run_payload_adapter.validate_python(payload)

    def test_conv_requires_kernel(self):
        payload = taylor_payload()
        payload["variant"] = "convolution"
        del payload["NU"]
        del payload["MU"]
        payload["kernel"] = {
            "mode": "generate",
            "params": {"U": 1.0e-16},
        }
        run = run_payload_adapter.validate_python(payload)
        self.assertIsInstance(run, ConvRun)

    def test_direct_union_alias_is_not_a_model(self):
        self.assertFalse(hasattr(RunPayload, "validate_python"))
        self.assertFalse(hasattr(RunPayload, "model_dump"))


class RunIdTests(unittest.TestCase):
    def test_run_id_defaults_empty(self):
        run = run_payload_adapter.validate_python(taylor_payload())
        self.assertEqual(run.run_id, "")

    def test_model_copy_updates_run_id(self):
        run = run_payload_adapter.validate_python(taylor_payload())
        renamed = run.model_copy(update={"run_id": "r000001"})
        self.assertEqual(renamed.run_id, "r000001")
        self.assertEqual(run.run_id, "")


class JsonRoundTripTests(unittest.TestCase):
    def test_jsonl_round_trip_preserves_discriminator(self):
        adapter = TypeAdapter(BaseRun)
        run = run_payload_adapter.validate_python(taylor_payload())
        run = run.model_copy(update={"run_id": "r000001"})
        dumped = run.model_dump_json()
        loaded = run_payload_adapter.validate_json(dumped)
        self.assertIsInstance(loaded, TaylorRun)
        self.assertEqual(loaded.run_id, "r000001")
        self.assertEqual(loaded.model_dump(), run.model_dump())

    def test_conv_jsonl_round_trip(self):
        payload = taylor_payload()
        payload["variant"] = "convolution"
        del payload["NU"]
        del payload["MU"]
        payload["kernel"] = {"mode": "generate", "params": {"U": 1.0e-16}}
        run = run_payload_adapter.validate_python(payload)
        loaded = run_payload_adapter.validate_json(run.model_dump_json())
        self.assertIsInstance(loaded, ConvRun)


if __name__ == "__main__":
    unittest.main()
