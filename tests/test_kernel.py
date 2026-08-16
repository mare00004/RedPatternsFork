from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from red_patterns.kernel import KernelConfig, compute_kernel, effective_morse_kernel, kernel_config_from_ui, write_kernel_h5
from red_patterns.types import KernelType


class HNCKernelTests(unittest.TestCase):
    def config(self, output: Path) -> KernelConfig:
        return KernelConfig(output_path=output, kernel_type=KernelType.HNC, kernel_n=31, dz=1e-6, subdiv=2, a=1e-16, b=1e-16, c=1.0, alpha=1e-16, beta=6e-6, gamma=1e6)

    def test_matches_effective_morse_formula_and_is_odd(self):
        result = compute_kernel(self.config(Path("kernel.h5")))
        expected = effective_morse_kernel(result.x_sample, 1e-16, 1e-16, 1.0, 1e-16, 6e-6, 1e6)
        np.testing.assert_allclose(result.K_sample, expected)
        np.testing.assert_allclose(result.K_sample, -result.K_sample[::-1])
        self.assertEqual(result.K_sample[len(result.K_sample) // 2], 0.0)

    def test_writes_hnc_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(Path(directory) / "kernel.h5")
            write_kernel_h5(cfg.output_path, compute_kernel(cfg), cfg)
            with h5py.File(cfg.output_path) as h5:
                self.assertEqual(h5["kernel"].attrs["kernel_type"], "hnc")
                self.assertEqual(h5["kernel"].attrs["alpha"], cfg.alpha)

    def test_ui_values_create_validated_hnc_config(self):
        value = {"kernel_type": "HNC effective Morse", "kernel_n": 31, "dz": 1e-6, "subdiv": 2, "a": 1.0, "b": 100.0, "c": 1.0, "alpha": 100.0, "beta": 6.0, "gamma": 1.0}
        cfg = kernel_config_from_ui(value)
        self.assertEqual(cfg.kernel_type, KernelType.HNC)
        self.assertEqual(cfg.a, 1e-10)
        self.assertEqual(cfg.beta, 6e-6)
        self.assertEqual(cfg.gamma, 1e6)


if __name__ == "__main__":
    unittest.main()
