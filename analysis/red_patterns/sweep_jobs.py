"""Sweep generation and JSONL I/O for simulation run payloads.

Sweep classes expand parameter ranges into Cartesian products and produce
validated Pydantic models (:class:`red_patterns.models.TaylorRun` /
:class:`red_patterns.models.ConvRun`) via the :data:`red_patterns.models.RunPayload`
discriminated union.  ``runs.jsonl`` files round-trip through the same models.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence, TypeVar

import numpy as np
from pydantic import ValidationError

from .models import BaseRun, ConvRun, RunPayload, TaylorRun, run_payload_adapter
from .types import ClosureType, Gradient, KernelType, PDFType, PhiType, Variant

RunModelT = TypeVar("RunModelT", bound=BaseRun)

DEFAULT_N = 256
DEFAULT_WING = 30
DEFAULT_RHO_CENTER = 1100.0
DEFAULT_RHO_SPAN = 30.0
DEFAULT_DZ = 0.000267651
DEFAULT_PSI_AVG = 0.02
DEFAULT_GAUSSIAN_MU = 1100.0
DEFAULT_GAUSSIAN_SIGMA = 4.0
DEFAULT_GAUSSIAN_BLOB_MU_Z = 0.035
DEFAULT_GAUSSIAN_BLOB_SIGMA_Z = 0.01
DEFAULT_RHO_RANGE = 5.0
DEFAULT_SINGLE_BIN_IDX = 256
DEFAULT_PERTURBATION_SEED = 0
DEFAULT_PERTURBATION_AMPLITUDE = 1e-3

DEFAULT_SIGMA = 5.6e-6
DEFAULT_G0 = 4.0e7
DEFAULT_SIGMA_C = 0.5e-6
DEFAULT_EQ_DIST = 6.585467201064237e-06

GENERATE = "generate"


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


def json_scalar(value: Any) -> Any:
    """Convert StrEnum members and numpy scalars to plain Python values."""
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


def dict_product(fields: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of field sequences into row dicts."""
    names = tuple(fields)
    combos = product(*(fields[name] for name in names))
    return [
        {name: json_scalar(value) for name, value in zip(names, combo, strict=True)}
        for combo in combos
    ]


@dataclass(frozen=True, kw_only=True)
class PhiSweep:
    psi_avg: Sequence[float] = (DEFAULT_PSI_AVG,)
    phi_type: Sequence[PhiType | str] = (PhiType.GAUSSIAN,)
    N: Sequence[int] = (DEFAULT_N,)
    wing_z: Sequence[int] = (DEFAULT_WING,)
    wing_r: Sequence[int] = (DEFAULT_WING,)
    rho_center: Sequence[float] = (DEFAULT_RHO_CENTER,)
    rho_span: Sequence[float] = (DEFAULT_RHO_SPAN,)
    dz: Sequence[float] = (DEFAULT_DZ,)
    gaussian_mu: Sequence[float] = (DEFAULT_GAUSSIAN_MU,)
    gaussian_sigma: Sequence[float] = (DEFAULT_GAUSSIAN_SIGMA,)
    gaussian_blob_mu_z: Sequence[float] = (DEFAULT_GAUSSIAN_BLOB_MU_Z,)
    gaussian_blob_sigma_z: Sequence[float] = (DEFAULT_GAUSSIAN_BLOB_SIGMA_Z,)
    rho_range: Sequence[float] = (DEFAULT_RHO_RANGE,)
    seed: Sequence[int] = (DEFAULT_PERTURBATION_SEED,)
    amplitude: Sequence[float] = (DEFAULT_PERTURBATION_AMPLITUDE,)
    single_bin_idx: Sequence[int] = (DEFAULT_SINGLE_BIN_IDX,)

    def rows(self) -> list[dict[str, Any]]:
        base_rows = dict_product(
            {
                "psi_avg": self.psi_avg,
                "N": self.N,
                "wing_z": self.wing_z,
                "wing_r": self.wing_r,
                "rho_center": self.rho_center,
                "rho_span": self.rho_span,
                "dz": self.dz,
            }
        )
        # The field registry declares which sweep dimensions each distribution
        # owns, avoiding a parallel type-switch here.
        from .phi import PHI_FIELD_TYPES

        rows: list[dict[str, Any]] = []
        for phi_type in self.phi_type:
            resolved_type = PhiType(json_scalar(phi_type))
            phi_value = resolved_type.value
            field_cls = PHI_FIELD_TYPES[resolved_type]
            type_rows = dict_product(
                {
                    name: getattr(self, name)
                    for name in field_cls.sweep_param_names()
                }
            )
            for base in base_rows:
                for type_values in type_rows:
                    rows.append({**base, "phi_type": phi_value, **type_values})
        return rows


@dataclass(frozen=True, kw_only=True)
class KernelSweep:
    kernel_type: Sequence[KernelType | str] = (KernelType.ORIGINAL,)
    closure: Sequence[ClosureType | str] = (ClosureType.FORCE,)
    pair_distribution: Sequence[PDFType | str] = (PDFType.NEAREST_NEIGHBOR,)
    U: Sequence[float] = (111.15e-18,)
    sigma: Sequence[float] = (DEFAULT_SIGMA,)
    kernel_n: Sequence[int] = (31,)
    dz: Sequence[float] = (DEFAULT_DZ,)
    subdiv: Sequence[int] = (256,)
    g0: Sequence[float] = (DEFAULT_G0,)
    nn_d: Sequence[float] = (DEFAULT_EQ_DIST,)
    nn_sigma: Sequence[float] = (DEFAULT_SIGMA_C,)
    lambda_: Sequence[float] = (1.0,)
    a: Sequence[float] = (1.0,)
    b: Sequence[float] = (1.0e-16,)
    c: Sequence[float] = (1.0,)
    alpha: Sequence[float] = (1.0e-16,)
    beta: Sequence[float] = (DEFAULT_EQ_DIST,)
    gamma: Sequence[float] = (6.0 / DEFAULT_EQ_DIST,)

    def rows(self) -> list[dict[str, Any]]:
        original_base_rows = dict_product(
            {
                "U": self.U,
                "sigma": self.sigma,
                "kernel_n": self.kernel_n,
                "dz": self.dz,
                "subdiv": self.subdiv,
            }
        )
        hnc_rows = dict_product({"a": self.a, "b": self.b, "c": self.c, "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "kernel_n": self.kernel_n, "dz": self.dz, "subdiv": self.subdiv})
        nn_rows = dict_product(
            {
                "g0": self.g0,
                "nn_d": self.nn_d,
                "nn_sigma": self.nn_sigma,
            }
        )
        lambda_rows = dict_product({"lambda_": self.lambda_})
        rows: list[dict[str, Any]] = []
        for kernel_type in self.kernel_type:
            if KernelType(json_scalar(kernel_type)) == KernelType.HNC:
                rows.extend({"kernel_type": KernelType.HNC.value, **row} for row in hnc_rows)
                continue
            for closure, pair_distribution in product(self.closure, self.pair_distribution):
                closure_value = ClosureType(json_scalar(closure)).value
                pair_value = PDFType(json_scalar(pair_distribution)).value
                for base in original_base_rows:
                    common = {
                        **base,
                        "kernel_type": KernelType.ORIGINAL.value,
                        "closure": closure_value,
                        "pair_distribution": pair_value,
                    }
                    if pair_value == PDFType.NEAREST_NEIGHBOR.value:
                        for nn in nn_rows:
                            rows.append({**common, **nn})
                    elif pair_value == PDFType.EXPONENTIAL.value:
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
        sim_rows = dict_product(
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
            payload = {
                **sim_row,
                **model_row,
                "phi": {"mode": GENERATE, "params": phi_row},
            }
            run = (
                TaylorRun(**payload)
                if self.variant is Variant.TAYLOR
                else ConvRun(**payload)
            )
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
        return dict_product({"NU": self.NU, "MU": self.MU})


@dataclass(frozen=True, kw_only=True)
class ConvSweep(SimulationSweep):
    kernel: KernelSweep

    @property
    def variant(self) -> Variant:
        return Variant.CONVOLUTION

    def model_rows(self) -> list[dict[str, Any]]:
        return [
            {"kernel": {"mode": GENERATE, "params": row}}
            for row in self.kernel.rows()
        ]


def assign_run_ids(runs: Iterable[RunModelT]) -> list[RunModelT]:
    return [
        run.model_copy(update={"run_id": f"r{index:06d}"})
        for index, run in enumerate(runs, start=1)
    ]


def combine_sweeps(*sweeps: SimulationSweep) -> list[RunPayload]:
    return assign_run_ids(run for sweep in sweeps for run in sweep.to_runs())


def normalize_runs(value: Any) -> list[RunPayload]:
    if isinstance(value, SimulationSweep):
        return combine_sweeps(value)
    if isinstance(value, list):
        runs: list[RunPayload] = []
        for index, item in enumerate(value, start=1):
            if isinstance(item, (TaylorRun, ConvRun)):
                runs.append(item)
            elif isinstance(item, BaseRun):
                raise TypeError(
                    f"line {index}: unsupported run model subclass "
                    f"{type(item).__name__}."
                )
            elif isinstance(item, dict):
                runs.append(run_payload_adapter.validate_python(item))
            else:
                raise TypeError(
                    f"line {index}: expected a run dict or model, "
                    f"got {type(item).__name__}."
                )
        return runs
    raise TypeError(
        "Expected `sweep` to be a SimulationSweep or a list of exported run dicts."
    )


def runs_to_jsonl(runs: Sequence[RunPayload]) -> str:
    return "\n".join(run.model_dump_json() for run in runs) + "\n"


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
                runs.append(run_payload_adapter.validate_json(line))
            except ValidationError as exc:
                raise ValueError(
                    f"line {line_number}: invalid run payload: {exc}."
                ) from exc
    return runs


def run_ids_from_runs(runs: Sequence[BaseRun]) -> list[str]:
    return [run.run_id for run in runs]


def write_run_id_queue(path: str | Path, runs: Sequence[BaseRun]) -> Path:
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


def find_run_by_id(runs: Sequence[BaseRun], run_id: str) -> BaseRun:
    for run in runs:
        if run.run_id == run_id:
            return run
    raise KeyError(f"run_id {run_id!r} not found in runs.jsonl")
