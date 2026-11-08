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
        command = [str(binary)]
        for key, value in args.items():
            for item in value if isinstance(value, tuple) else (value,):
                if item is not None:
                    command.append(f"--{key}" if item is True else f"--{key}={item}")
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


@pytest.mark.parametrize("kernel_n", [31, 1027])
def test_convolution_halo_matches_constant_field(solver, kernel_n):
    result, out = solver(kernel=np.full(kernel_n, 1e-30), store="face-velocity")
    assert result.returncode == 0, result.stdout + result.stderr
    expected_velocity = -7.4e23 * 0.02 * kernel_n * 1e-30 * (0.07 / 32 / 256)
    with h5py.File(out / "run.h5") as handle:
        np.testing.assert_allclose(
            handle["fields/face_velocity"][0, :, 16], expected_velocity, rtol=1e-12
        )


@pytest.mark.parametrize("duration, steps", [(0.01, 1), (0.02, 2), (0.025, 3), (0.07, 7)])
def test_final_snapshot_and_diagnostics_match_final_state(solver, duration, steps):
    phi = np.zeros((32, 32))
    phi[0, 2] = 0.02
    result, out = solver(phi=phi, T=duration, storeTime=0.5, gradient="linear",
                         store=("face-velocity", "face-flux"))
    assert result.returncode == 0, result.stdout + result.stderr
    with h5py.File(out / "run.h5") as handle:
        np.testing.assert_array_equal(handle["time"][:], [0.0, duration])
        assert handle["config/run"].attrs["NT"] == steps
        saved_phi = handle["fields/phi"][:]
        assert not np.array_equal(saved_phi[-1], saved_phi[0])
        np.testing.assert_allclose(handle["fields/psi"][:], saved_phi.sum(axis=1))
        # No cells upstream of the spike are populated. Its source-cell mass
        # decays by a known factor per Euler step, including a partial last step.
        velocity = handle["fields/face_velocity"][0]
        dz = handle["config/run"].attrs["DZ"]
        rate = velocity[0, 3] / dz
        full_steps = int(duration / 0.01)
        remainder = duration - full_steps * 0.01
        expected_source = 0.02 * (1 - rate * 0.01)**full_steps * (1 - rate * remainder)
        assert saved_phi[-1, 0, 2] == pytest.approx(expected_source, rel=1e-12)
        expected_flux = velocity[:, 1:-1] * np.where(
            velocity[:, 1:-1] > 0, saved_phi[-1, :, :-1], saved_phi[-1, :, 1:]
        )
        np.testing.assert_allclose(handle["fields/face_flux"][-1, :, 1:-1], expected_flux)


@pytest.mark.parametrize("missing", [{"NU": None}, {"MU": None}, {"NU": None, "MU": None}])
def test_taylor_requires_explicit_coefficients(solver, missing):
    result, out = solver(**missing)
    assert result.returncode != 0
    assert not (out / "run.h5").exists()
    assert "missing option" in result.stderr


@pytest.mark.parametrize("invalid", [{"NU": "nan"}, {"MU": "inf"}])
def test_taylor_rejects_nonfinite_coefficients(solver, invalid):
    result, out = solver(**invalid)
    assert result.returncode != 0
    assert not (out / "run.h5").exists()
    assert "finite" in result.stderr
