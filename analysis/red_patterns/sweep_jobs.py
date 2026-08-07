from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

RunPayload = dict[str, Any]

DEFAULT_N = 256
DEFAULT_WING = 30
DEFAULT_RHO_CENTER = 1100.0
DEFAULT_RHO_SPAN = 30.0
DEFAULT_DZ = 0.000267651
DEFAULT_PSI_AVG = 0.02
DEFAULT_GAUSSIAN_MU = 1100.0
DEFAULT_GAUSSIAN_SIGMA = 4.0

DEFAULT_SIGMA = 5.6e-6
DEFAULT_G0 = 4.0e7
DEFAULT_SIGMA_C = 0.5e-6
DEFAULT_EQ_DIST = 6.585467201064237e-06


class PhiMode(StrEnum):
    GAUSSIAN = "gaussian"
    HOMOGENEOUS = "homogeneous"


class ClosureMode(StrEnum):
    FORCE = "force"
    POTENTIAL = "potential"


class PairDistributionMode(StrEnum):
    MEAN_FIELD = "mean-field"
    NEAREST_NEIGHBOR = "nearest-neighbor"
    EXPONENTIAL = "exponential"


class Variant(StrEnum):
    TAYLOR = "taylor"
    CONVOLUTION = "convolution"


class Gradient(StrEnum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float
    step: float

    def values(self) -> np.ndarray:
        if self.step <= 0:
            raise ValueError("step must be > 0")
        eps = self.step / 2.0
        values = np.arange(self.lo, self.hi + eps, self.step, dtype=float)
        return values[values <= self.hi + eps]


def _json_scalar(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dict_product(fields: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    names = tuple(fields)
    combos = product(*(fields[name] for name in names))
    return [
        {name: _json_scalar(value) for name, value in zip(names, combo, strict=True)}
        for combo in combos
    ]


def _assign_run_ids(runs: Iterable[RunPayload]) -> list[RunPayload]:
    assigned: list[RunPayload] = []
    for index, run in enumerate(runs, start=1):
        assigned.append({"run_id": f"r{index:06d}", **run})
    return assigned


@dataclass(frozen=True, kw_only=True)
class PhiSweep:
    psi_avg: Sequence[float] = (DEFAULT_PSI_AVG,)
    phi_type: Sequence[str] = (PhiMode.GAUSSIAN.value,)
    N: Sequence[int] = (DEFAULT_N,)
    wing: Sequence[int] = (DEFAULT_WING,)
    rho_center: Sequence[float] = (DEFAULT_RHO_CENTER,)
    rho_span: Sequence[float] = (DEFAULT_RHO_SPAN,)
    dz: Sequence[float] = (DEFAULT_DZ,)
    gaussian_mu: Sequence[float] = (DEFAULT_GAUSSIAN_MU,)
    gaussian_sigma: Sequence[float] = (DEFAULT_GAUSSIAN_SIGMA,)

    def rows(self) -> list[dict[str, Any]]:
        base_rows = _dict_product(
            {
                "psi_avg": self.psi_avg,
                "N": self.N,
                "wing": self.wing,
                "rho_center": self.rho_center,
                "rho_span": self.rho_span,
                "dz": self.dz,
            }
        )
        rows: list[dict[str, Any]] = []
        gaussian_rows = _dict_product(
            {
                "gaussian_mu": self.gaussian_mu,
                "gaussian_sigma": self.gaussian_sigma,
            }
        )
        for phi_type in self.phi_type:
            phi_type_value = PhiMode(str(_json_scalar(phi_type))).value
            for base in base_rows:
                if phi_type_value == PhiMode.GAUSSIAN.value:
                    for gaussian in gaussian_rows:
                        rows.append(
                            {
                                **base,
                                "phi_type": phi_type_value,
                                **gaussian,
                            }
                        )
                else:
                    rows.append({**base, "phi_type": phi_type_value})
        return rows


@dataclass(frozen=True, kw_only=True)
class KernelSweep:
    closure: Sequence[str] = (ClosureMode.FORCE.value,)
    pair_distribution: Sequence[str] = (PairDistributionMode.NEAREST_NEIGHBOR.value,)
    U: Sequence[float] = (111.15e-18,)
    sigma: Sequence[float] = (DEFAULT_SIGMA,)
    kernel_n: Sequence[int] = (31,)
    dz: Sequence[float] = (DEFAULT_DZ,)
    subdiv: Sequence[int] = (256,)
    g0: Sequence[float] = (DEFAULT_G0,)
    nn_d: Sequence[float] = (DEFAULT_EQ_DIST,)
    nn_sigma: Sequence[float] = (DEFAULT_SIGMA_C,)
    lambda_: Sequence[float] = (1.0,)

    def rows(self) -> list[dict[str, Any]]:
        base_rows = _dict_product(
            {
                "U": self.U,
                "sigma": self.sigma,
                "kernel_n": self.kernel_n,
                "dz": self.dz,
                "subdiv": self.subdiv,
            }
        )
        nn_rows = _dict_product(
            {
                "g0": self.g0,
                "nn_d": self.nn_d,
                "nn_sigma": self.nn_sigma,
            }
        )
        lambda_rows = _dict_product({"lambda_": self.lambda_})
        rows: list[dict[str, Any]] = []
        for closure, pair_distribution in product(self.closure, self.pair_distribution):
            closure_value = ClosureMode(str(_json_scalar(closure))).value
            pair_distribution_value = PairDistributionMode(
                str(_json_scalar(pair_distribution))
            ).value
            for base in base_rows:
                common = {
                    **base,
                    "closure": closure_value,
                    "pair_distribution": pair_distribution_value,
                }
                if (
                    pair_distribution_value
                    == PairDistributionMode.NEAREST_NEIGHBOR.value
                ):
                    for nn in nn_rows:
                        rows.append({**common, **nn})
                elif pair_distribution_value == PairDistributionMode.EXPONENTIAL.value:
                    for lambda_row in lambda_rows:
                        rows.append({**common, **lambda_row})
                else:
                    rows.append(common)
        return rows


@dataclass(frozen=True, kw_only=True)
class SimulationSweep:
    N: Sequence[int]
    T: Sequence[float]
    DT: Sequence[float]
    storeTime: Sequence[float]
    gradient: Sequence[Gradient | str]
    phi: PhiSweep

    @property
    def variant(self) -> Variant:
        raise NotImplementedError

    def model_rows(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def to_runs(self) -> list[RunPayload]:
        sim_rows = _dict_product(
            {
                "N": self.N,
                "T": self.T,
                "DT": self.DT,
                "storeTime": self.storeTime,
                "gradient": self.gradient,
            }
        )
        phi_rows = self.phi.rows()
        runs: list[RunPayload] = []
        for sim_row, model_row, phi_row in product(
            sim_rows, self.model_rows(), phi_rows
        ):
            run: RunPayload = {
                "variant": self.variant.value,
                **sim_row,
                **model_row,
                "phi": {"mode": "generate", "params": phi_row},
            }
            runs.append(run)
        return runs


@dataclass(frozen=True, kw_only=True)
class TaylSweep(SimulationSweep):
    NU: Sequence[float]
    MU: Sequence[float]

    @property
    def variant(self) -> Variant:
        return Variant.TAYLOR

    def model_rows(self) -> list[dict[str, Any]]:
        return _dict_product({"NU": self.NU, "MU": self.MU})


@dataclass(frozen=True, kw_only=True)
class ConvSweep(SimulationSweep):
    kernel: KernelSweep

    @property
    def variant(self) -> Variant:
        return Variant.CONVOLUTION

    def model_rows(self) -> list[dict[str, Any]]:
        return [
            {"kernel": {"mode": "generate", "params": row}}
            for row in self.kernel.rows()
        ]


def combine_sweeps(*sweeps: SimulationSweep) -> list[RunPayload]:
    return _assign_run_ids(run for sweep in sweeps for run in sweep.to_runs())


def normalize_runs(value: Any) -> list[RunPayload]:
    if isinstance(value, SimulationSweep):
        return combine_sweeps(value)
    if isinstance(value, list):
        validate_runs(value)
        return value
    raise TypeError(
        "Expected `sweep` to be a SimulationSweep or a list of exported run dicts."
    )


# FIX: fix to match new params
def validate_run_payload(run: RunPayload, *, line_number: int | None = None) -> None:
    prefix = "" if line_number is None else f"line {line_number}: "
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{prefix}missing non-empty string `run_id`.")

    variant = run.get("variant")
    if variant not in {Variant.TAYLOR.value, Variant.CONVOLUTION.value}:
        raise ValueError(f"{prefix}invalid `variant`: {variant!r}.")

    for key in ("T", "DT", "storeTime"):
        if not isinstance(run.get(key), (int, float)):
            raise ValueError(f"{prefix}`{key}` must be numeric.")
    if not isinstance(run.get("gradient"), str):
        raise ValueError(f"{prefix}`gradient` must be a string.")

    phi = run.get("phi")
    if not isinstance(phi, dict) or phi.get("mode") != "generate":
        raise ValueError(f"{prefix}`phi.mode` must be `generate`.")
    if not isinstance(phi.get("params"), dict):
        raise ValueError(f"{prefix}`phi.params` must be an object.")

    if variant == Variant.TAYLOR.value:
        for key in ("NU", "MU"):
            if not isinstance(run.get(key), (int, float)):
                raise ValueError(f"{prefix}`{key}` must be numeric for Taylor runs.")
        if "kernel" in run:
            raise ValueError(f"{prefix}Taylor runs must not define `kernel`.")
    else:
        kernel = run.get("kernel")
        if not isinstance(kernel, dict) or kernel.get("mode") != "generate":
            raise ValueError(f"{prefix}`kernel.mode` must be `generate`.")
        if not isinstance(kernel.get("params"), dict):
            raise ValueError(f"{prefix}`kernel.params` must be an object.")


def validate_runs(runs: Sequence[RunPayload]) -> None:
    seen: set[str] = set()
    for line_number, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ValueError(f"line {line_number}: expected a JSON object.")
        validate_run_payload(run, line_number=line_number)
        run_id = str(run["run_id"])
        if run_id in seen:
            raise ValueError(f"line {line_number}: duplicate run_id {run_id!r}.")
        seen.add(run_id)


def runs_to_jsonl(runs: Sequence[RunPayload]) -> str:
    validate_runs(runs)
    return "\n".join(json.dumps(run, separators=(",", ":")) for run in runs) + "\n"


def write_runs_jsonl(path: str | Path, runs: Sequence[RunPayload]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(runs_to_jsonl(runs), encoding="utf-8")
    return output_path


def load_runs_jsonl(path: str | Path) -> list[RunPayload]:
    source_path = Path(path)
    runs: list[RunPayload] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"line {line_number}: invalid JSON: {exc.msg}."
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number}: expected a JSON object.")
            runs.append(payload)
    validate_runs(runs)
    return runs


def run_ids_from_runs(runs: Sequence[RunPayload]) -> list[str]:
    validate_runs(runs)
    return [str(run["run_id"]) for run in runs]


def write_run_id_queue(path: str | Path, runs: Sequence[RunPayload]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ids = run_ids_from_runs(runs)
    output_path.write_text("\n".join(run_ids) + "\n", encoding="utf-8")
    return output_path


def write_sweep_export(
    directory: str | Path,
    runs: Sequence[RunPayload],
    *,
    runs_filename: str = "runs.jsonl",
    run_ids_filename: str = "run_ids.txt",
) -> tuple[Path, Path]:
    export_dir = Path(directory)
    export_dir.mkdir(parents=True, exist_ok=True)
    runs_path = write_runs_jsonl(export_dir / runs_filename, runs)
    run_ids_path = write_run_id_queue(export_dir / run_ids_filename, runs)
    return runs_path, run_ids_path


def find_run_by_id(runs: Sequence[RunPayload], run_id: str) -> RunPayload:
    for run in runs:
        if run.get("run_id") == run_id:
            return run
    raise KeyError(f"run_id {run_id!r} not found in runs.jsonl")
