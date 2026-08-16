from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from red_patterns.kernel import (
    KernelConfig,
    compute_kernel,
    write_kernel_h5,
)
from red_patterns.models import ConvRun, TaylorRun
from red_patterns.phi import phi_field_from_params
from red_patterns.sim import build_cli_args, locate_binary
from red_patterns.sweep_jobs import find_run_by_id, load_runs_jsonl


def kernel_config_from_params(
    params: dict[str, Any], output_path: Path
) -> KernelConfig:
    return KernelConfig.from_params(params, output_path)


def cli_args_from_payload(
    *,
    run: TaylorRun | ConvRun,
    binary_path: Path,
    out_dir: Path,
    phi_path: Path,
    kernel_path: Path | None,
) -> list[str]:
    """Assemble the ``red-patterns`` CLI for a single parsed ``runs.jsonl`` payload."""
    if isinstance(run, TaylorRun):
        mode = "Taylor"
        nu = float(run.NU)
        mu = float(run.MU)
    elif isinstance(run, ConvRun):
        mode = "Convolution"
        nu = 0.0
        mu = 0.0
    else:
        raise AssertionError(f"unreachable run type: {type(run).__name__}")
    return build_cli_args(
        binary_path=binary_path,
        mode=mode,
        out_dir=out_dir,
        phi_path=phi_path,
        kernel_path=kernel_path,
        gradient=run.gradient.value,
        N=run.N,
        t_final=run.T,
        dt=run.DT,
        storeTime=run.storeTime,
        nu=nu,
        mu=mu,
    )


def _resolve_binary(binary: str | None) -> Path:
    if binary:
        return Path(binary)
    container_binary = Path("/bin/red-patterns")
    if container_binary.exists():
        return container_binary
    return locate_binary(REPO_ROOT)


def run_selected(
    *,
    runs_jsonl: Path,
    run_id: str,
    binary: str | None = None,
    work_dir: Path = Path("."),
) -> Path:
    runs = load_runs_jsonl(runs_jsonl)
    run = find_run_by_id(runs, run_id)

    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    run_spec_path = work_dir / "run_spec.json"
    command_path = work_dir / "command.txt"
    output_h5 = work_dir / "run.h5"
    phi_path = work_dir / "phi.h5"
    kernel_path: Path | None = None

    run_spec_path.write_text(
        run.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    phi_params = run.phi.params
    phi_field = phi_field_from_params(phi_params)
    phi_result = phi_field.compute()
    phi_field.write_phi_h5(phi_path, phi_result)

    if isinstance(run, ConvRun):
        kernel_params = dict(run.kernel.params)
        kernel_path = work_dir / "kernel.h5"
        kernel_cfg = kernel_config_from_params(kernel_params, kernel_path)
        kernel_result = compute_kernel(kernel_cfg)
        write_kernel_h5(kernel_path, kernel_result, kernel_cfg)

    binary_path = _resolve_binary(binary)
    args = cli_args_from_payload(
        run=run,
        binary_path=binary_path,
        out_dir=work_dir,
        phi_path=phi_path,
        kernel_path=kernel_path,
    )
    command_path.write_text(shlex.join(args) + "\n", encoding="utf-8")

    try:
        subprocess.run(args, check=True, cwd=work_dir)
    finally:
        phi_path.unlink(missing_ok=True)
        if kernel_path is not None:
            kernel_path.unlink(missing_ok=True)

    if not output_h5.exists():
        raise FileNotFoundError(f"Simulation finished without creating {output_h5}")
    return output_h5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve one run from runs.jsonl, generate inputs, and run it."
    )
    parser.add_argument("--runs-jsonl", required=True, help="Path to runs.jsonl.")
    parser.add_argument("--run-id", required=True, help="run_id to execute.")
    parser.add_argument(
        "--binary",
        help="Simulation binary to execute. Defaults to /bin/red-patterns in containers.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_h5 = run_selected(
        runs_jsonl=Path(args.runs_jsonl),
        run_id=args.run_id,
        binary=args.binary,
    )
    print(f"Completed {args.run_id}: {output_h5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
