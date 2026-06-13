from __future__ import annotations

import argparse
import json
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
    ClosureType,
    KernelConfig,
    PDFType,
    compute_kernel,
    write_kernel_h5,
)
from red_patterns.phi import PhiConfig, PhiType, compute_phi, write_phi_h5
from red_patterns.sim import build_cli_args, locate_binary
from red_patterns.sweep_jobs import find_run_by_id, load_runs_jsonl


def _phi_config_from_params(params: dict[str, Any], output_path: Path) -> PhiConfig:
    return PhiConfig(
        output_path=output_path,
        phi_type=PhiType(str(params["phi_type"])),
        psi_avg=float(params["psi_avg"]),
        N=int(params["N"]),
        wing=int(params["wing"]),
        rho_center=float(params["rho_center"]),
        rho_span=float(params["rho_span"]),
        dz=float(params["dz"]),
        gaussian_mu=(float(params["gaussian_mu"]) if "gaussian_mu" in params else None),
        gaussian_sigma=(
            float(params["gaussian_sigma"]) if "gaussian_sigma" in params else None
        ),
    )


def _kernel_config_from_params(
    params: dict[str, Any], output_path: Path
) -> KernelConfig:
    return KernelConfig(
        output_path=output_path,
        closure=ClosureType(str(params["closure"])),
        pair_distribution=PDFType(str(params["pair_distribution"])),
        U=float(params["U"]),
        sigma=float(params["sigma"]),
        kernel_n=int(params["kernel_n"]),
        dz=float(params["dz"]),
        subdiv=int(params["subdiv"]),
        g0=float(params["g0"]) if "g0" in params else None,
        nn_d=float(params["nn_d"]) if "nn_d" in params else None,
        nn_sigma=float(params["nn_sigma"]) if "nn_sigma" in params else None,
        lambda_=float(params["lambda_"]) if "lambda_" in params else None,
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
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    phi_params = dict(run["phi"]["params"])
    phi_cfg = _phi_config_from_params(phi_params, phi_path)
    phi_result = compute_phi(phi_cfg)
    write_phi_h5(phi_path, phi_result, phi_cfg)

    if run["variant"] == "convolution":
        kernel_params = dict(run["kernel"]["params"])
        kernel_path = work_dir / "kernel.h5"
        kernel_cfg = _kernel_config_from_params(kernel_params, kernel_path)
        kernel_result = compute_kernel(kernel_cfg)
        write_kernel_h5(kernel_path, kernel_result, kernel_cfg)

    binary_path = _resolve_binary(binary)
    args = build_cli_args(
        binary_path=binary_path,
        mode="Taylor" if run["variant"] == "taylor" else "Convolution",
        out_dir=work_dir,
        phi_path=phi_path,
        kernel_path=kernel_path,
        gradient=str(run["gradient"]),
        t_final=float(run["T"]),
        dt=float(run["DT"]),
        save_every=int(run["NO"]),
        nu=float(run.get("NU", 0.0)),
        mu=float(run.get("MU", 0.0)),
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
