# pyright: reportUnknownVariableType=false
# pyright: reportAny=false
"""Initial phi-field generation library.

This file contains the functions used for generating the initial
$\\varphi(\\rho, z)$ field and exporting it to HDF5.

CLI usage (in a marimo notebook's script-mode cell)
---------------------------------------------------

>>> if mo.app_meta().mode == "script" and sys.argv[1:2] == ["export"]:
...     raise SystemExit(run_export_cli(sys.argv[2:], prog="phi_init.py export"))
...
Then run: ``uv run analysis/phi_init.py export --output run.h5 --phi-type gaussian ...``

Marimo notebook usage
---------------------
>>> ui = make_phi_ui()                          # cell 1
>>> phi_ui_layout(ui)                           # cell 2 — display
>>> cfg = phi_config_from_ui(ui.value, "out.h5") # cell 3
>>> result = compute_phi(cfg)                   # cell 4
>>> write_phi_h5(cfg.output_path, result, cfg)  # cell 5
>>> plot_phi(result)                            # optional cell

As with :mod:`red_patterns.kernel`, the compute path imports
neither marimo nor matplotlib; the UI factory imports them lazily.

For mathematical details see ``analysis/phi_init.py``.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from .models import PHI_PARAMS_ADAPTER, PhiParams, PhiParamsBase
from .types import PhiType

Array1F = NDArray[np.float64]
Array2F = NDArray[np.float64]

# --------------------------------------------------------------------------- #
# Defaults / labels
# --------------------------------------------------------------------------- #

DEFAULT_N = 512
DEFAULT_WING = 30 + 2
DEFAULT_RHO_CENTER = 1100.0
DEFAULT_RHO_SPAN = 30.0
DEFAULT_RHO_RANGE = 5.0
DEFAULT_DZ = 0.000267651 / 2.0
DEFAULT_PSI_AVG = 0.02
DEFAULT_GAUSSIAN_MU = 1100.0
DEFAULT_GAUSSIAN_SIGMA = 4.0
DEFAULT_GAUSSIAN_BLOB_MU_Z = 0.035
DEFAULT_GAUSSIAN_BLOB_SIGMA_Z = 0.01
DEFAULT_Z_SYSTEM_SIZE = 0.07
DEFAULT_SINGLE_BIN_IDX = 256


LABEL_MAP = {t.label: t for t in PhiType}


# --------------------------------------------------------------------------- #
# Config / result dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PhiConfig:
    """All parameters needed to build an initial phi field and export it."""

    output_path: Path
    phi_type: PhiType
    psi_avg: float
    N: int = DEFAULT_N
    wing_z: int = DEFAULT_WING
    wing_r: int = DEFAULT_WING
    rho_center: float = DEFAULT_RHO_CENTER
    rho_span: float = DEFAULT_RHO_SPAN
    rho_range: float = DEFAULT_RHO_RANGE
    dz: float = DEFAULT_DZ
    gaussian_mu: float | None = None
    gaussian_sigma: float | None = None
    gaussian_blob_mu_z: float | None = None
    gaussian_blob_sigma_z: float | None = None
    single_bin_idx: int | None = None

    @classmethod
    def from_params(
        cls, params: PhiParams, output_path: str | Path = "initial_phi.h5"
    ) -> PhiConfig:
        """Build a config from a validated :data:`PhiParams` model.

        ``params`` may also be a plain dict; it is run through
        :data:`PHI_PARAMS_ADAPTER` first.  The single wire-level ``wing`` entry
        is split into ``wing_z``/``wing_r`` for the compute path.
        """
        if not isinstance(params, PhiParamsBase):
            params = PHI_PARAMS_ADAPTER.validate_python(params)  # type: ignore[redundant-cast]
        gaussian_mu = getattr(params, "gaussian_mu", None)
        gaussian_sigma = getattr(params, "gaussian_sigma", None)
        gaussian_blob_mu_z = getattr(params, "gaussian_blob_mu_z", None)
        gaussian_blob_sigma_z = getattr(params, "gaussian_blob_sigma_z", None)
        single_bin_idx = getattr(params, "single_bin_idx", None)
        return cls(
            output_path=Path(output_path),
            phi_type=PhiType(params.phi_type),
            psi_avg=float(params.psi_avg),
            N=int(params.N),
            wing_z=int(params.wing),
            wing_r=int(params.wing),
            rho_center=float(params.rho_center),
            rho_span=float(params.rho_span),
            dz=float(params.dz),
            rho_range=float(getattr(params, "rho_range", DEFAULT_RHO_RANGE)),
            gaussian_mu=(
                float(gaussian_mu) if gaussian_mu is not None else None
            ),
            gaussian_sigma=(
                float(gaussian_sigma) if gaussian_sigma is not None else None
            ),
            gaussian_blob_mu_z=(
                float(gaussian_blob_mu_z)
                if gaussian_blob_mu_z is not None
                else None
            ),
            gaussian_blob_sigma_z=(
                float(gaussian_blob_sigma_z)
                if gaussian_blob_sigma_z is not None
                else None
            ),
            single_bin_idx=(
                int(single_bin_idx) if single_bin_idx is not None else None
            ),
        )


@dataclass
class PhiResult:
    rho: Array1F
    z: Array1F
    phi_values: Array2F  # phi[rho_idx, z_idx], float64


# --------------------------------------------------------------------------- #
# Field builders
# --------------------------------------------------------------------------- #


def phi_smooth_homogeneous(
    rho: Array1F,
    z: Array1F,
    psi_avg: float,
    rho_center: float,
    rho_range: float,
    wing_r: int,
) -> Array2F:
    r"""Constant block in ρ with cosine taper to the wing boundaries.

    The block spans physical coordinates
    ``[rho_center - rho_range, rho_center + rho_range]`` in ρ and the full
    active z‑range.  A cosine ramp bridges between the block edges and the
    wing index boundaries (``wing_r`` on the left, ``N_rho - 1 - wing_r`` on
    the right), so the field reaches zero exactly at the wing edge.
    """
    N_rho = rho.shape[0]
    N_z = z.shape[0]

    # ---- locate block edges in index space ----
    left_edge_idx = int(np.searchsorted(rho, rho_center - rho_range, side="left"))
    right_edge_idx = int(np.searchsorted(rho, rho_center + rho_range, side="right")) - 1

    # ---- build radial profile: 1 inside block, cosine taper to wing ----
    profile = np.zeros(N_rho)
    profile[left_edge_idx : right_edge_idx + 1] = 1.0

    # left taper: 0 at wing_r, 1 at left_edge_idx
    n_left = left_edge_idx - wing_r
    if n_left > 0:
        x = np.linspace(0.0, 1.0, n_left + 2)[1:-1]  # interior points
        profile[wing_r : wing_r + n_left] = 0.5 * (1 - np.cos(np.pi * x))

    # right taper: 1 at right_edge_idx, 0 at N_rho - 1 - wing_r
    n_right = (N_rho - 1 - wing_r) - right_edge_idx
    if n_right > 0:
        x = np.linspace(0.0, 1.0, n_right + 2)[1:-1]
        idx_start = right_edge_idx + 1
        profile[idx_start : idx_start + n_right] = 0.5 * (1 + np.cos(np.pi * x))

    return (psi_avg * profile[:, np.newaxis] * np.ones(N_z)).astype(np.float64)


def phi_homogeneous(rho: Array1F, z: Array1F, psi_avg: float) -> Array2F:
    r"""$\varphi(\rho, z) = \langle \psi \rangle / L_\rho$ (normalized)."""
    N_rho = rho.shape[0]
    N_z = z.shape[0]
    rho_len = rho[-1] - rho[0]
    return (psi_avg / rho_len) * np.ones((N_rho, N_z), dtype=np.float32)


def phi_gaussian(
    rho: Array1F,
    z: Array1F,
    psi_avg: float,
    mu: float,
    sigma: float,
) -> Array2F:
    r"""Gaussian-in-$\rho$ field scaled by ``psi_avg`` (normalized)."""
    N_z = z.shape[0]
    radial_profile = (
        psi_avg
        * (1 / np.sqrt(2 * np.pi * sigma**2))
        * np.exp(-((rho - mu) ** 2) / (2.0 * sigma**2))
    )
    return radial_profile[:, np.newaxis] * np.ones(N_z)


def phi_gaussian_blob(
    rho: Array1F,
    z: Array1F,
    psi_avg: float,
    mu_rho: float,
    sigma_rho: float,
    mu_z: float,
    sigma_z: float,
) -> Array2F:
    r"""2D Gaussian blob in $(\rho, z)$ scaled by ``psi_avg``."""
    norm = psi_avg / (2.0 * np.pi * sigma_rho * sigma_z)
    rho_part = np.exp(-((rho - mu_rho) ** 2) / (2.0 * sigma_rho**2))
    z_part = np.exp(-((z - mu_z) ** 2) / (2.0 * sigma_z**2))
    return (norm * rho_part[:, np.newaxis] * z_part[np.newaxis, :]).astype(np.float64)


def phi_single_bin(
    rho: Array1F,
    z: Array1F,
    psi_avg: float,
    bin_idx: int,
) -> Array2F:
    r"""Delta-function-like ridge: ``psi_avg`` at one rho bin, zero elsewhere."""
    N_rho, N_z = rho.shape[0], z.shape[0]
    phi = np.zeros((N_rho, N_z), dtype=np.float64)
    phi[bin_idx, :] = psi_avg
    return phi


def phi_add_wing(phi, wing_z, wing_r):
    """
    Zeros out phi iff
        (z_j < wing_z or z_j > N_z - 1 - wing_z)
        and (rho_i < wing_r or rho_i > N_r - 1 - wing_r)
    See `analysis/phi_init.py` for details.
    """
    N_rho, N_z = phi.shape
    rho_idx = np.arange(N_rho)
    z_idx = np.arange(N_z)
    z_mask = (z_idx < wing_z) | (z_idx > (N_z - 1 - wing_z))
    rho_mask = (rho_idx < wing_r) | (rho_idx > (N_rho - 1 - wing_r))

    result = phi.copy()
    result[:, z_mask] = 0.0
    result[rho_mask, :] = 0.0
    return result


def renormalize_phi(phi, rho, z, psi_avg, wing_z):
    """
    See `analysis/phi_init.py` under "Normalization" for details
    """
    _, N_z = phi.shape

    z_start = wing_z
    z_end = N_z - wing_z
    n_z_eff = z_end - z_start  # N_z - 2*(wing+2)
    psi_profile = phi.sum(axis=0)
    current_avg = psi_profile[z_start:z_end].sum() / n_z_eff

    if current_avg > 0:
        return phi * (psi_avg / current_avg)
    return phi.copy()


def build_phi_axes(
    *, N: int, rho_center: float, rho_span: float, dz: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds the following two axis
        `z := [0, (N-1) * dz]`   (system size ≈ 0.07 m)
        `rho := [rho_center - rho_span / 2.0, rho_center + rho_span / 2.0]`
    with N points each.
    """
    rho = np.linspace(
        rho_center - rho_span / 2.0,
        rho_center + rho_span / 2.0,
        N,
        dtype=np.float64,
    )
    z = np.linspace(0.0, (N - 1) * dz, N, dtype=np.float64)
    return rho, z


# --------------------------------------------------------------------------- #
# Phi field hierarchy
# --------------------------------------------------------------------------- #


class PhiField(ABC):
    """Base class for initial phi field generators.

    Each concrete subclass corresponds to one :class:`PhiType` member and owns
    the math (``build``), parameter validation, and HDF5 metadata for that
    distribution.  ``compute_phi``/``write_phi_h5`` dispatch to the subclass
    through :data:`PHI_FIELD_TYPES` instead of a mutable ``if/elif`` chain.
    """

    phi_type: ClassVar[PhiType]

    def __init__(self, cfg: PhiConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        """Return the raw field ``phi[rho_idx, z_idx]`` for this distribution.

        Callers are responsible for applying wings and renormalization.
        """

    def validate(self) -> list[str]:
        """Return human-readable configuration errors, empty if valid."""
        return []

    def write_metadata(self, group: h5py.Group) -> None:
        """Write type-specific attributes to the ``phi`` HDF5 group."""


class GaussianPhi(PhiField):
    phi_type = PhiType.GAUSSIAN

    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        cfg = self.cfg
        assert cfg.gaussian_mu is not None
        assert cfg.gaussian_sigma is not None
        return phi_gaussian(rho, z, cfg.psi_avg, cfg.gaussian_mu, cfg.gaussian_sigma)

    def validate(self) -> list[str]:
        errors: list[str] = []
        cfg = self.cfg
        if cfg.gaussian_mu is None:
            errors.append("gaussian_mu is required for a Gaussian phi field.")
        if cfg.gaussian_sigma is None:
            errors.append("gaussian_sigma is required for a Gaussian phi field.")
        elif cfg.gaussian_sigma <= 0.0:
            errors.append("gaussian_sigma must be positive.")
        return errors

    def write_metadata(self, group: h5py.Group) -> None:
        assert self.cfg.gaussian_mu is not None
        assert self.cfg.gaussian_sigma is not None
        group.attrs["gaussian_mu"] = float(self.cfg.gaussian_mu)
        group.attrs["gaussian_sigma"] = float(self.cfg.gaussian_sigma)


class GaussianBlobPhi(GaussianPhi):
    phi_type = PhiType.GAUSSIAN_BLOB

    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        cfg = self.cfg
        assert cfg.gaussian_mu is not None
        assert cfg.gaussian_sigma is not None
        assert cfg.gaussian_blob_mu_z is not None
        assert cfg.gaussian_blob_sigma_z is not None
        return phi_gaussian_blob(
            rho,
            z,
            cfg.psi_avg,
            cfg.gaussian_mu,
            cfg.gaussian_sigma,
            cfg.gaussian_blob_mu_z,
            cfg.gaussian_blob_sigma_z,
        )

    def validate(self) -> list[str]:
        errors = super().validate()
        cfg = self.cfg
        if cfg.gaussian_blob_mu_z is None:
            errors.append("gaussian_blob_mu_z is required for a Gaussian Blob phi field.")
        if cfg.gaussian_blob_sigma_z is None:
            errors.append(
                "gaussian_blob_sigma_z is required for a Gaussian Blob phi field."
            )
        elif cfg.gaussian_blob_sigma_z <= 0.0:
            errors.append("gaussian_blob_sigma_z must be positive.")
        return errors

    def write_metadata(self, group: h5py.Group) -> None:
        super().write_metadata(group)
        assert self.cfg.gaussian_blob_mu_z is not None
        assert self.cfg.gaussian_blob_sigma_z is not None
        group.attrs["gaussian_blob_mu_z"] = float(self.cfg.gaussian_blob_mu_z)
        group.attrs["gaussian_blob_sigma_z"] = float(self.cfg.gaussian_blob_sigma_z)


class HomogeneousPhi(PhiField):
    phi_type = PhiType.HOMOGENEOUS

    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        return phi_homogeneous(rho, z, self.cfg.psi_avg)


class SmoothHomogeneousPhi(PhiField):
    phi_type = PhiType.SMOOTH_HOMOGENEOUS

    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        cfg = self.cfg
        return phi_smooth_homogeneous(
            rho, z, cfg.psi_avg, cfg.rho_center, cfg.rho_range, cfg.wing_r
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.cfg.rho_range is None:
            errors.append("rho_range is required for a smooth homogeneous phi field.")
        elif self.cfg.rho_range <= 0.0:
            errors.append("rho_range must be positive.")
        return errors

    def write_metadata(self, group: h5py.Group) -> None:
        assert self.cfg.rho_range is not None
        group.attrs["rho_range"] = float(self.cfg.rho_range)


class SingleBinPhi(PhiField):
    phi_type = PhiType.SINGLE_BIN

    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        assert self.cfg.single_bin_idx is not None
        return phi_single_bin(rho, z, self.cfg.psi_avg, self.cfg.single_bin_idx)

    def validate(self) -> list[str]:
        errors: list[str] = []
        cfg = self.cfg
        if cfg.single_bin_idx is None:
            errors.append("single_bin_idx is required for a single-bin phi field.")
        elif not (0 <= cfg.single_bin_idx < cfg.N):
            errors.append(
                f"single_bin_idx must be in [0, {cfg.N - 1}] for N={cfg.N}."
            )
        return errors

    def write_metadata(self, group: h5py.Group) -> None:
        assert self.cfg.single_bin_idx is not None
        group.attrs["single_bin_idx"] = int(self.cfg.single_bin_idx)


PHI_FIELD_TYPES: dict[PhiType, type[PhiField]] = {
    PhiType.GAUSSIAN: GaussianPhi,
    PhiType.GAUSSIAN_BLOB: GaussianBlobPhi,
    PhiType.HOMOGENEOUS: HomogeneousPhi,
    PhiType.SMOOTH_HOMOGENEOUS: SmoothHomogeneousPhi,
    PhiType.SINGLE_BIN: SingleBinPhi,
}


def phi_field_for(cfg: PhiConfig) -> PhiField:
    """Return the :class:`PhiField` subclass matching ``cfg.phi_type``."""
    return PHI_FIELD_TYPES[cfg.phi_type](cfg)


def compute_phi(cfg: PhiConfig) -> PhiResult:
    """Build the wing-applied, renormalized initial phi field."""
    rho, z = build_phi_axes(
        N=cfg.N, rho_center=cfg.rho_center, rho_span=cfg.rho_span, dz=cfg.dz
    )

    field = PHI_FIELD_TYPES[cfg.phi_type](cfg)
    errors = field.validate()
    if errors:
        raise ValueError(
            f"invalid configuration for {cfg.phi_type.label} phi field: "
            + "; ".join(errors)
        )
    phi = field.build(rho, z)

    phi_wing = renormalize_phi(
        phi_add_wing(phi, cfg.wing_z, cfg.wing_r), rho, z, cfg.psi_avg, cfg.wing_z
    )
    # FIX: Remove
    print(phi_wing.sum(axis=0).sum() / (cfg.N - 2 * cfg.wing_z))
    return PhiResult(rho=rho, z=z, phi_values=np.asarray(phi_wing, dtype=np.float64))


def write_phi_h5(output_path: str | Path, result: PhiResult, cfg: PhiConfig) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(output_path), "w") as f:
        group = f.create_group("phi")
        _ = group.create_dataset(
            "values", data=np.asarray(result.phi_values, dtype=np.float64)
        )
        _ = group.create_dataset("rho", data=np.asarray(result.rho, dtype=np.float64))
        _ = group.create_dataset("z", data=np.asarray(result.z, dtype=np.float64))
        group.attrs["N"] = int(cfg.N)
        group.attrs["PSI"] = float(cfg.psi_avg)
        group.attrs["wing_z"] = int(cfg.wing_z)
        group.attrs["wing_r"] = int(cfg.wing_r)
        group.attrs["phi_type"] = cfg.phi_type.label
        group.attrs["storage_order"] = "phi[rho_idx, z_idx]"
        group.attrs["generated_by"] = "red_patterns/phi.py"
        group.attrs["normalization"] = "no runtime renormalization required"

        PHI_FIELD_TYPES[cfg.phi_type](cfg).write_metadata(group)

    return output_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_export_parser(prog: str = "phi export") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Export a CUDA-compatible initial phi field to HDF5.",
    )
    _ = parser.add_argument(
        "--output", required=True, help="Path to the output HDF5 file."
    )
    _ = parser.add_argument(
        "--phi-type",
        required=True,
        choices=sorted(t.value for t in PhiType),
        help="Initial phi distribution to use.",
    )
    _ = parser.add_argument(
        "--psi-avg", required=True, type=float, help="Average volume fraction."
    )
    _ = parser.add_argument(
        "--N", type=int, default=DEFAULT_N, help="Grid size in rho and z."
    )
    _ = parser.add_argument(
        "--wingz",
        type=int,
        default=DEFAULT_WING,
        help="wing size in z direction",
    )
    _ = parser.add_argument(
        "--wingr",
        type=int,
        default=DEFAULT_WING,
        help="wing size in z direction",
    )
    _ = parser.add_argument(
        "--rho-center",
        type=float,
        default=DEFAULT_RHO_CENTER,
        help="Center of the rho axis in g/L.",
    )
    _ = parser.add_argument(
        "--rho-span",
        type=float,
        default=DEFAULT_RHO_SPAN,
        help="Total rho-axis span in g/L.",
    )
    _ = parser.add_argument(
        "--rho-range",
        type=float,
        default=DEFAULT_RHO_RANGE,
        help="Half‑width of the constant block in the ρ direction.",
    )
    _ = parser.add_argument(
        "--dz", type=float, default=DEFAULT_DZ, help="Z-axis spacing in meters."
    )
    _ = parser.add_argument(
        "--gaussian-mu", type=float, help="Gaussian rho center in g/L."
    )
    _ = parser.add_argument(
        "--gaussian-sigma", type=float, help="Gaussian rho width in g/L."
    )
    _ = parser.add_argument(
        "--gaussian-blob-mu-z",
        type=float,
        help="Gaussian blob z center in meters.",
    )
    _ = parser.add_argument(
        "--gaussian-blob-sigma-z",
        type=float,
        help="Gaussian blob z width in meters.",
    )
    _ = parser.add_argument(
        "--single-bin-idx",
        type=int,
        help="Rho bin index for single-bin phi type.",
    )
    return parser


def validate_export_namespace(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> PhiConfig:
    errors: list[str] = []

    if args.N < 3:
        errors.append("--N must be an integer >= 3.")
    if args.wingz < 0:
        errors.append("--wingz must be non-negative.")
    if args.wingr < 0:
        errors.append("--wingz must be non-negative.")
    if args.psi_avg < 0.0:
        errors.append("--psi-avg must be non-negative.")
    if args.rho_span <= 0.0:
        errors.append("--rho-span must be positive.")
    if args.dz <= 0.0:
        errors.append("--dz must be positive.")

    active_z = args.N - 2 * args.wingz
    active_rho = args.N - 2 * args.wingr
    if active_z <= 0:
        errors.append(
            "--wingz is too large for --N: the active z region would be empty."
        )
    if active_rho <= 0:
        errors.append(
            "--wingr is too large for --N: the active rho region would be empty."
        )

    if args.phi_type == PhiType.GAUSSIAN:
        if args.gaussian_mu is None:
            errors.append("--gaussian-mu is required with --phi-type=gaussian.")
        if args.gaussian_sigma is None:
            errors.append("--gaussian-sigma is required with --phi-type=gaussian.")
        elif args.gaussian_sigma <= 0.0:
            errors.append("--gaussian-sigma must be positive.")
        if (
            args.gaussian_blob_mu_z is not None
            or args.gaussian_blob_sigma_z is not None
        ):
            errors.append(
                "--gaussian-blob-mu-z and --gaussian-blob-sigma-z are only valid with "
                + "--phi-type=gaussian_blob."
            )
    elif args.phi_type == PhiType.GAUSSIAN_BLOB:
        if args.gaussian_mu is None:
            errors.append("--gaussian-mu is required with --phi-type=gaussian_blob.")
        if args.gaussian_sigma is None:
            errors.append("--gaussian-sigma is required with --phi-type=gaussian_blob.")
        elif args.gaussian_sigma <= 0.0:
            errors.append("--gaussian-sigma must be positive.")
        if args.gaussian_blob_mu_z is None:
            errors.append(
                "--gaussian-blob-mu-z is required with --phi-type=gaussian_blob."
            )
        if args.gaussian_blob_sigma_z is None:
            errors.append(
                "--gaussian-blob-sigma-z is required with --phi-type=gaussian_blob."
            )
        elif args.gaussian_blob_sigma_z <= 0.0:
            errors.append("--gaussian-blob-sigma-z must be positive.")
    elif args.phi_type == PhiType.SMOOTH_HOMOGENEOUS:
        if args.rho_range is None:
            errors.append("--rho-range is required with --phi-type=smooth_homogeneous.")
        elif args.rho_range <= 0.0:
            errors.append("--rho-range must be positive.")
        # Ensure the block edges are at least wing_r indices from each boundary
        dr = args.rho_span / (args.N - 1)  # physical spacing per cell
        max_allowed = args.rho_span / 2.0 - args.wingr * dr
        if args.rho_range > max_allowed:
            errors.append(
                "--rho-range + wingr exceeds the rho domain for smooth_homogeneous."
            )
    elif args.phi_type == PhiType.SINGLE_BIN:
        if args.single_bin_idx is None:
            errors.append("--single-bin-idx is required with --phi-type=single_bin.")
        elif args.single_bin_idx < 0 or args.single_bin_idx >= args.N:
            errors.append(
                f"--single-bin-idx must be in [0, {args.N - 1}] for N={args.N}."
            )
        if args.gaussian_mu is not None or args.gaussian_sigma is not None:
            errors.append(
                "--gaussian-mu and --gaussian-sigma are only valid with "
                + "--phi-type=gaussian or --phi-type=gaussian_blob."
            )
        if (
            args.gaussian_blob_mu_z is not None
            or args.gaussian_blob_sigma_z is not None
        ):
            errors.append(
                "--gaussian-blob-mu-z and --gaussian-blob-sigma-z are only valid with "
                + "--phi-type=gaussian_blob."
            )
    else:
        if args.gaussian_mu is not None or args.gaussian_sigma is not None:
            errors.append(
                "--gaussian-mu and --gaussian-sigma are only valid with "
                + "--phi-type=gaussian or --phi-type=gaussian_blob."
            )
        if (
            args.gaussian_blob_mu_z is not None
            or args.gaussian_blob_sigma_z is not None
        ):
            errors.append(
                "--gaussian-blob-mu-z and --gaussian-blob-sigma-z are only valid with "
                + "--phi-type=gaussian_blob."
            )
        if args.single_bin_idx is not None:
            errors.append("--single-bin-idx is only valid with --phi-type=single_bin.")

    if errors:
        parser.error("\n".join(errors))

    return PhiConfig(
        output_path=Path(args.output),
        phi_type=PhiType(args.phi_type),
        psi_avg=args.psi_avg,
        N=args.N,
        wing_z=args.wingz,
        wing_r=args.wingr,
        rho_center=args.rho_center,
        rho_range=args.rho_range,
        rho_span=args.rho_span,
        dz=args.dz,
        gaussian_mu=args.gaussian_mu,
        gaussian_sigma=args.gaussian_sigma,
        gaussian_blob_mu_z=args.gaussian_blob_mu_z,
        gaussian_blob_sigma_z=args.gaussian_blob_sigma_z,
        single_bin_idx=args.single_bin_idx,
    )


def run_export_cli(argv: list[str], prog: str = "phi export") -> int:
    parser = build_export_parser(prog)
    args = parser.parse_args(argv)
    cfg = validate_export_namespace(parser, args)
    result = compute_phi(cfg)
    output_path = write_phi_h5(cfg.output_path, result, cfg)

    print(f"Exported initial phi to {output_path}")
    print(
        f"Stored shape {result.phi_values.shape} on /phi/values with storage order "
        + "phi[rho_idx, z_idx]"
    )
    summary = [
        f"phi_type={cfg.phi_type}",
        f"PSI={cfg.psi_avg:.6e}",
        f"N={cfg.N}",
        f"wingz={cfg.wing_z}",
        f"wingr={cfg.wing_r}",
        f"rho_center={cfg.rho_center:.6e}",
        f"rho_span={cfg.rho_span:.6e}",
        f"DZ={cfg.dz:.6e}",
    ]
    if cfg.phi_type == PhiType.GAUSSIAN:
        assert cfg.gaussian_mu is not None
        assert cfg.gaussian_sigma is not None
        summary.extend(
            [
                f"gaussian_mu={cfg.gaussian_mu:.6e}",
                f"gaussian_sigma={cfg.gaussian_sigma:.6e}",
            ]
        )
    elif cfg.phi_type == PhiType.GAUSSIAN_BLOB:
        assert cfg.gaussian_mu is not None
        assert cfg.gaussian_sigma is not None
        assert cfg.gaussian_blob_mu_z is not None
        assert cfg.gaussian_blob_sigma_z is not None
        summary.extend(
            [
                f"gaussian_mu={cfg.gaussian_mu:.6e}",
                f"gaussian_sigma={cfg.gaussian_sigma:.6e}",
                f"gaussian_blob_mu_z={cfg.gaussian_blob_mu_z:.6e}",
                f"gaussian_blob_sigma_z={cfg.gaussian_blob_sigma_z:.6e}",
            ]
        )
    elif cfg.phi_type == PhiType.SMOOTH_HOMOGENEOUS:
        summary.append(f"rho_range={cfg.rho_range:.6e}")
    elif cfg.phi_type == PhiType.SINGLE_BIN:
        assert cfg.single_bin_idx is not None
        summary.append(f"single_bin_idx={cfg.single_bin_idx}")
    print("Used parameters: " + ", ".join(summary))
    return 0


# --------------------------------------------------------------------------- #
# UI factory (marimo). Imports are lazy so the compute path stays headless.
# --------------------------------------------------------------------------- #


def make_phi_ui(
    *,
    psi_avg: float = DEFAULT_PSI_AVG,
    phi_type: str = PhiType.GAUSSIAN.label,
    gaussian_mu: float = DEFAULT_GAUSSIAN_MU,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
    gaussian_blob_mu_z: float = DEFAULT_GAUSSIAN_BLOB_MU_Z,
    gaussian_blob_sigma_z: float = DEFAULT_GAUSSIAN_BLOB_SIGMA_Z,
):
    """Build every phi control as a single ``mo.ui.dictionary``.

    Grid parameters (N, wing, rho_center, rho_span, dz) are kept at their
    defaults — :func:`phi_config_from_ui` fills them in. Returning one registered
    UI element keeps cross-cell ``.value`` reads reliable in marimo.
    """
    import marimo as mo

    return mo.ui.dictionary(
        {
            "psi_avg": mo.ui.number(
                start=0.0,
                stop=1.0,
                step=0.001,
                value=psi_avg,
                label="$\\langle \\psi \\rangle$",
            ),
            "N": mo.ui.dropdown(
                options=[2**x for x in range(6, 12, 1)], value=2**10, label="N"
            ),
            "phi_type": mo.ui.tabs(
                {
                    PhiType.GAUSSIAN.label: mo.md(
                        r"$$\varphi(\rho, z) = \langle \psi \rangle \,"
                        + r"\mathcal{N}(\rho;\mu_\rho,\sigma_\rho)$$"
                    ),
                    PhiType.GAUSSIAN_BLOB.label: mo.md(
                        r"$$\varphi(\rho, z) = \langle \psi \rangle \,"
                        + r"\mathcal{N}(\rho,z;\mu_\rho,\sigma_\rho,\mu_z,\sigma_z)$$"
                    ),
                    PhiType.HOMOGENEOUS.label: mo.md(
                        r"$\varphi(\rho, z) = \langle \psi \rangle / L_\rho$"
                    ),
                    PhiType.SMOOTH_HOMOGENEOUS.label: mo.md(
                        r"$\varphi(\rho, z) = \langle \psi \rangle$ in a rectangular block, "
                        + r"cosine‑tapered to zero along $\rho$"
                    ),
                    PhiType.SINGLE_BIN.label: mo.md(
                        r"$\varphi(\rho_i, z) = \langle \psi \rangle$ "
                        + r"(single bin, zero elsewhere)"
                    ),
                },
                value=phi_type,
            ),
            "gaussian_mu": mo.ui.number(
                start=0.0,
                stop=2000,
                step=0.1,
                value=gaussian_mu,
                label="$\\mu_\\rho \\; [\\frac{g}{L}]$",
            ),
            "gaussian_sigma": mo.ui.number(
                start=0.0,
                stop=15,
                step=0.1,
                value=gaussian_sigma,
                label="$\\sigma_\\rho \\; [\\frac{g}{L}]$",
            ),
            "gaussian_blob_mu_z": mo.ui.number(
                start=0.0,
                stop=1.0,
                step=0.001,
                value=gaussian_blob_mu_z,
                label="$\\mu_z \\; [m]$",
            ),
            "gaussian_blob_sigma_z": mo.ui.number(
                start=0.0,
                stop=0.1,
                step=0.001,
                value=gaussian_blob_sigma_z,
                label="$\\sigma_z \\; [m]$",
            ),
            "wingz": mo.ui.number(
                start=0,
                stop=1000,
                step=1,
                value=DEFAULT_WING,
                label="z wing size",
            ),
            "wingr": mo.ui.number(
                start=0,
                stop=1000,
                step=1,
                value=DEFAULT_WING,
                label="r wing size",
            ),
            "rho_range": mo.ui.number(
                start=0.0,
                stop=100.0,
                step=0.1,
                value=DEFAULT_RHO_RANGE,
                label="ρ half‑width (region)",
            ),
            "single_bin_idx": mo.ui.number(
                start=0,
                stop=1023,
                step=1,
                value=DEFAULT_SINGLE_BIN_IDX,
                label="ρ bin index",
            ),
        }
    )


def phi_ui_layout(phi_ui):
    """Compose the phi controls into a readable panel (Gaussian params hidden
    unless the Gaussian tab is active)."""
    import marimo as mo

    items = [
        mo.md("### Average volume fraction"),
        phi_ui["psi_avg"],
        mo.md("### Grid size"),
        phi_ui["N"],
        mo.md("### Distribution type"),
        phi_ui["phi_type"],
        mo.md("### Wing Settings"),
        mo.hstack([phi_ui["wingz"], phi_ui["wingr"]], justify="start", align="center"),
    ]
    if str(phi_ui["phi_type"].value) == PhiType.SMOOTH_HOMOGENEOUS.label:
        items.extend(
            [
                mo.md("### Smooth Homogeneous Parameters"),
                phi_ui["rho_range"],
            ]
        )
    if str(phi_ui["phi_type"].value) == PhiType.GAUSSIAN.label:
        items.extend(
            [
                mo.md("### Gaussian Parameters"),
                phi_ui["gaussian_mu"],
                phi_ui["gaussian_sigma"],
            ]
        )
    if str(phi_ui["phi_type"].value) == PhiType.GAUSSIAN_BLOB.label:
        items.extend(
            [
                mo.md("### Gaussian Blob Parameters"),
                phi_ui["gaussian_mu"],
                phi_ui["gaussian_sigma"],
                phi_ui["gaussian_blob_mu_z"],
                phi_ui["gaussian_blob_sigma_z"],
            ]
        )
    if str(phi_ui["phi_type"].value) == PhiType.SINGLE_BIN.label:
        items.extend(
            [
                mo.md("### Single Bin Parameters"),
                phi_ui["single_bin_idx"],
            ]
        )
    return mo.vstack(items, gap=0.5)


def phi_config_from_ui(
    value: dict[str, Any], output_path: str | Path = "initial_phi.h5"
) -> PhiConfig:
    """Map a ``make_phi_ui()`` value dict to a :class:`PhiConfig`."""
    phi_type = LABEL_MAP[str(value["phi_type"])]
    is_gaussian = phi_type == PhiType.GAUSSIAN
    is_blob = phi_type == PhiType.GAUSSIAN_BLOB
    is_single_bin = phi_type == PhiType.SINGLE_BIN
    return PhiConfig(
        output_path=Path(output_path),
        phi_type=phi_type,
        N=int(value["N"]),
        dz=(DEFAULT_Z_SYSTEM_SIZE / float(value["N"])),
        wing_z=int(value["wingz"]),
        wing_r=int(value["wingr"]),
        psi_avg=float(value["psi_avg"]),
        rho_range=float(value["rho_range"]),
        gaussian_mu=float(value["gaussian_mu"]) if (is_gaussian or is_blob) else None,
        gaussian_sigma=float(value["gaussian_sigma"])
        if (is_gaussian or is_blob)
        else None,
        gaussian_blob_mu_z=float(value["gaussian_blob_mu_z"]) if is_blob else None,
        gaussian_blob_sigma_z=float(value["gaussian_blob_sigma_z"])
        if is_blob
        else None,
        single_bin_idx=int(value["single_bin_idx"]) if is_single_bin else None,
    )


def plot_phi(result: PhiResult):
    """Plot the initial $\\varphi(\\rho, z)$ field as a heatmap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        result.phi_values,
        origin="lower",
        aspect="auto",
        extent=(result.z[0], result.z[-1], result.rho[0], result.rho[-1]),
        cmap="viridis",
    )
    ax.set_xlabel("$z$ [m]")
    ax.set_ylabel(r"$\rho$ [g/L]")
    fig.colorbar(im, ax=ax, label=r"$\phi$")
    ax.set_title(r"$\varphi(\rho, z)$")
    return fig
