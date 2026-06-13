from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from red_patterns.kernel import ClosureType, PDFType
from red_patterns.phi import PhiType
from red_patterns.sweep_jobs import (
    ConvSweep,
    Gradient,
    KernelSweep,
    PhiSweep,
    TaylSweep,
    combine_sweeps,
    load_runs_jsonl,
    runs_to_jsonl,
    write_sweep_export,
)


def build_sample_runs():
    phi = PhiSweep(
        psi_avg=[0.02],
        phi_type=[PhiType.GAUSSIAN],
        gaussian_mu=[1100.0],
        gaussian_sigma=[4.0],
    )
    kernel = KernelSweep(
        closure=[ClosureType.FORCE],
        pair_distribution=[PDFType.NEAREST_NEIGHBOR],
        U=[111.15e-18],
    )
    tayl = TaylSweep(
        T=[1.0],
        DT=[0.1],
        NO=[2],
        gradient=[Gradient.LINEAR],
        phi=phi,
        NU=[-1.0e-30],
        MU=[-1.0e-37],
    )
    conv = ConvSweep(
        T=[1.0],
        DT=[0.1],
        NO=[2],
        gradient=[Gradient.SIGMOID],
        phi=phi,
        kernel=kernel,
    )
    return combine_sweeps(tayl, conv)


class SweepPipelineTests(unittest.TestCase):
    def test_exported_runs_have_expected_shape(self):
        runs = build_sample_runs()
        self.assertEqual([run["run_id"] for run in runs], ["r000001", "r000002"])

        tayl_run, conv_run = runs
        self.assertEqual(tayl_run["variant"], "taylor")
        self.assertIn("phi", tayl_run)
        self.assertNotIn("kernel", tayl_run)

        self.assertEqual(conv_run["variant"], "convolution")
        self.assertIn("phi", conv_run)
        self.assertIn("kernel", conv_run)

        jsonl_lines = runs_to_jsonl(runs).strip().splitlines()
        self.assertEqual(len(jsonl_lines), 2)
        self.assertEqual(json.loads(jsonl_lines[0])["run_id"], "r000001")
        self.assertEqual(json.loads(jsonl_lines[1])["run_id"], "r000002")

    def test_export_writes_runs_jsonl_and_run_ids_in_matching_order(self):
        runs = build_sample_runs()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            runs_jsonl, queue_path = write_sweep_export(tmp_path, runs)

            loaded_runs = load_runs_jsonl(runs_jsonl)
            self.assertEqual([run["run_id"] for run in loaded_runs], ["r000001", "r000002"])
            self.assertEqual(
                queue_path.read_text(encoding="utf-8").splitlines(),
                ["r000001", "r000002"],
            )

    def test_run_one_generates_inputs_and_invokes_expected_variant(self):
        runs = build_sample_runs()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            runs_jsonl = tmp_path / "runs.jsonl"
            fake_binary = tmp_path / "fake-red-patterns.sh"

            runs_jsonl.write_text(runs_to_jsonl(runs), encoding="utf-8")
            fake_binary.write_text(
                """#!/usr/bin/env bash
set -Eeuo pipefail
out_dir=""
phi_file=""
kernel_file=""
for arg in "$@"; do
  case "$arg" in
    --out-dir=*) out_dir="${arg#--out-dir=}" ;;
    --phi-file=*) phi_file="${arg#--phi-file=}" ;;
    --kernel-file=*) kernel_file="${arg#--kernel-file=}" ;;
  esac
done
if [[ -z "${out_dir}" || -z "${phi_file}" ]]; then
  echo "missing required args" >&2
  exit 9
fi
if [[ ! -f "${phi_file}" ]]; then
  echo "missing phi file" >&2
  exit 10
fi
if [[ -n "${kernel_file}" && ! -f "${kernel_file}" ]]; then
  echo "missing kernel file" >&2
  exit 11
fi
mkdir -p "${out_dir}"
touch "${out_dir}/run.h5"
""",
                encoding="utf-8",
            )
            fake_binary.chmod(0o755)

            for run_id, expected_flag, expect_kernel in (
                ("r000001", "--use-taylor", False),
                ("r000002", "--use-convolution", True),
            ):
                work_dir = tmp_path / run_id
                work_dir.mkdir()
                result = subprocess.run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "sweep" / "run_one.py"),
                        "--runs-jsonl",
                        str(runs_jsonl),
                        "--run-id",
                        run_id,
                        "--binary",
                        str(fake_binary),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=work_dir,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)

                command_text = work_dir.joinpath("command.txt").read_text(encoding="utf-8")
                run_spec = json.loads(
                    work_dir.joinpath("run_spec.json").read_text(encoding="utf-8")
                )

                self.assertIn(expected_flag, command_text)
                self.assertEqual(run_spec["run_id"], run_id)
                self.assertTrue(work_dir.joinpath("run.h5").exists())
                self.assertTrue(work_dir.joinpath("command.txt").exists())
                self.assertTrue(work_dir.joinpath("run_spec.json").exists())
                self.assertFalse(work_dir.joinpath("phi.h5").exists())

                kernel_path = work_dir / "kernel.h5"
                if expect_kernel:
                    self.assertTrue(
                        "--kernel-file=" in command_text
                    )
                else:
                    self.assertFalse(kernel_path.exists())
                    self.assertFalse("--kernel-file=" in command_text)

    def test_submit_file_queues_run_ids_and_keys_logs_by_run_id(self):
        submit_text = (REPO_ROOT / "cluster" / "sweep.submit").read_text(
            encoding="utf-8"
        )
        self.assertIn("queue run_id from $(RUN_IDS_FILE)", submit_text)
        self.assertIn("transfer_input_files = $(LAUNCH_SH),$(RUNS_JSONL)", submit_text)
        self.assertIn("arguments = $(run_id)", submit_text)
        self.assertIn("log = logs/$(run_id).log", submit_text)
        self.assertIn("output = logs/$(run_id).out", submit_text)
        self.assertIn("error = logs/$(run_id).err", submit_text)


if __name__ == "__main__":
    unittest.main()
