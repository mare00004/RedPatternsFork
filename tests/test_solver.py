"""CUDA integration regressions (opt in with RED_PATTERNS_BINARY).

Run after building, on a CUDA host:
    RED_PATTERNS_BINARY=build/red-patterns uv run pytest tests/test_solver.py
Optionally set RED_PATTERNS_SANITIZER to the compute-sanitizer executable.
"""

import os
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest


@pytest.fixture
def solver(tmp_path):
    executable = os.environ.get("RED_PATTERNS_BINARY")
    if not executable:
        pytest.skip("Set RED_PATTERNS_BINARY to run CUDA integration tests")
    binary = Path(executable).resolve()
    assert binary.is_file(), binary

    def run(*, phi=None, kernel=None, kernel_attrs=None, **options):
        phi_path = tmp_path / "phi.h5"
        with h5py.File(phi_path, "w") as handle:
            handle.create_dataset(
                "phi/values", data=np.full((32, 32), 0.02 / 32) if phi is None else phi
            )
        args = {
            "use-taylor": True,
            "N": 32,
            "NU": 0,
            "MU": 0,
            "gradient": "zero",
            "T": 0.02,
            "DT": 0.01,
            "storeTime": 0.01,
            "out-dir": tmp_path,
            "phi-file": phi_path,
        }
        if kernel is not None:
            kernel_path = tmp_path / "kernel.h5"
            with h5py.File(kernel_path, "w") as handle:
                group = handle.create_group("kernel")
                group.create_dataset("values", data=kernel)
                group.attrs.update(
                    {"DZ": 0.07 / 32, "subDiv": 256, "spacing": 0.07 / 32 / 256}
                    if kernel_attrs is None else kernel_attrs
                )
            args.update({"use-taylor": None, "NU": None, "MU": None,
                         "use-convolution": True, "kernel-file": kernel_path})
        args.update(options)
        command = [str(binary)] + [
            f"--{key}" if value is True else f"--{key}={value}"
            for key, value in args.items() if value is not None
        ]
        sanitizer = os.environ.get("RED_PATTERNS_SANITIZER")
        if sanitizer:
            command = [sanitizer, "--error-exitcode=99", "--print-limit=5", *command]
        result = subprocess.run(command, text=True, capture_output=True, timeout=60)
        return result, tmp_path

    return run


def test_convolution_covers_last_coarse_cell(solver):
    result, out = solver(kernel=np.zeros(31))
    assert result.returncode == 0, result.stdout + result.stderr
    with h5py.File(out / "run.h5") as handle:
        assert handle["config/model/variant/conv"].attrs["M"] == 32 * 256
        np.testing.assert_allclose(handle["fields/phi"][:], 0.02 / 32)


def test_cuda_launch_failure_marks_run_failed(solver):
    # This launch exceeds the default dynamic shared-memory allowance.
    result, out = solver(kernel=np.zeros(7001))
    assert result.returncode != 0
    assert "CUDA Runtime Error" in result.stderr
    progress = json.loads((out / "progress.json").read_text())
    assert progress["status"] == "failed"
    assert "CUDA" in progress["error"]
