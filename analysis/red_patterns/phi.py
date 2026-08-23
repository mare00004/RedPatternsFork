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
>>> field = phi_field_from_ui(ui.value)         # cell 3
>>> result = field.compute()                    # cell 4
>>> field.write_phi_h5("initial_phi.h5", result)  # cell 5
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

from .models import (
    PHI_PARAMS_ADAPTER,
    GaussianBlobPhiParams,
    GaussianPhiParams,
    HomogeneousPhiParams,
    LinearFullRidgePhiParams,
    PerturbedSmoothHomogeneousPhiParams,
    PhiParamsBase,
    SingleBinPhiParams,
    SingleModeSmoothHomogeneousPhiParams,
    SingleModeLinearFullRidgePhiParams,
    SmoothHomogeneousPhiParams,
)
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


def _add_argument_if_missing(
    parser: argparse.ArgumentParser, *option_strings: str, **kwargs: Any
) -> None:
    """Add an option unless already registered for the flat all-phi-type CLI.

    The export parser collects arguments from every phi type into one flat
    command-line interface.  The selected Pydantic model later validates that
    only arguments valid for that phi type were supplied.  Some types share an
    option such as ``--amplitude``, so duplicate-safe registration avoids an
    argparse conflict while preserving that later validation.
    """
    registered_options = {
        option_string
        for action in parser._actions
        for option_string in action.option_strings
    }
    if any(option_string in registered_options for option_string in option_strings):
        return
    parser.add_argument(*option_strings, **kwargs)


# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #


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


def perturb_phi_z(
    phi: Array2F,
    z: Array1F,
    wing_z: int,
    *,
    amplitude: float,
    seed: int,
    mode_min: int = 1,
    mode_max: int = 32,
) -> Array2F:
    """Apply a mass-conserving random cosine perturbation along z."""
    result = phi.copy()
    active = slice(wing_z, phi.shape[1] - wing_z)
    z_active = z[active]
    phi_active = phi[:, active]
    x = (z_active - z_active[0]) / (z_active[-1] - z_active[0])

    rng = np.random.default_rng(seed)
    modes = np.arange(mode_min, mode_max + 1)
    coefficients = rng.normal(size=len(modes))
    eta = np.sum(
        coefficients[:, np.newaxis]
        * np.cos(np.pi * modes[:, np.newaxis] * x[np.newaxis, :]),
        axis=0,
    )

    psi = phi_active.sum(axis=0)
    eta -= np.average(eta, weights=psi)
    eta /= np.sqrt(np.mean(eta**2))
    result[:, active] *= 1.0 + amplitude * eta[np.newaxis, :]
    return result


def perturb_phi_single_cosine_z(
    phi: Array2F,
    z: Array1F,
    wing_z: int,
    *,
    amplitude: float,
    mode_number: int,
) -> Array2F:
    r"""Apply one DCT-style cosine perturbation along the active z domain."""
    result = phi.copy()
    active = slice(wing_z, phi.shape[1] - wing_z)
    z_active = z[active]
    if z_active.size == 1:
        x = np.zeros_like(z_active)
    else:
        x = (z_active - z_active[0]) / (z_active[-1] - z_active[0])
    multiplier = 1.0 + amplitude * np.cos(np.pi * mode_number * x)
    result[:, active] *= multiplier[np.newaxis, :]
    return result


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


def phi_linear_full_ridge(
    rho: Array1F,
    z: Array1F,
    psi_avg: float,
    rho_center: float,
) -> Array2F:
    r"""One-bin ridge following CUDA's ``LINEAR_FULL`` gradient.

    This initial phi should be the equilibrium state of the ``LINEAR_FULL`` gradient

    The CUDA implementation uses cell centers and maps its full gradient from
    ``-15`` to ``+15`` g/L relative to the central density. This generator
    evaluates those same centers, shifts them by ``rho_center``, and places
    each column in the nearest exported rho bin.
    """
    N_rho, N_z = rho.shape[0], z.shape[0]
    if N_z == 0:
        return np.zeros((N_rho, 0), dtype=np.float64)

    # In ``p_func``: DR = 30 / N and z is the cell center.  Substituting the
    # simulator's full-domain geometry yields this density for each z index.
    z_indices = np.arange(N_z, dtype=np.float64)
    gradient_rho = rho_center + 15.0 - 30.0 * (z_indices + 0.5) / N_z
    insertion = np.searchsorted(rho, gradient_rho, side="left")
    upper = np.clip(insertion, 0, N_rho - 1)
    lower = np.clip(insertion - 1, 0, N_rho - 1)
    rho_indices = np.where(
        np.abs(rho[lower] - gradient_rho) <= np.abs(rho[upper] - gradient_rho),
        lower,
        upper,
    )

    phi = np.zeros((N_rho, N_z), dtype=np.float64)
    phi[rho_indices, np.arange(N_z)] = psi_avg
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
    """Base class for initial phi-field generators.

    The base class owns the shared grid parameters every distribution needs
    (``N``, ``psi_avg``, wing sizes, rho axis, ``dz``) and the compute/export
    pipeline.  Each concrete subclass corresponds to one :class:`PhiType`
    member and owns its distribution-specific parameters, the math
    (``build``), parameter validation, HDF5 metadata, and a CLI summary.
    """

    phi_type: ClassVar[PhiType]
    params_model: ClassVar[type[PhiParamsBase]]

    def __init__(
        self,
        *,
        N: int,
        psi_avg: float,
        wing_z: int,
        wing_r: int,
        rho_center: float,
        rho_span: float,
        dz: float,
    ) -> None:
        self.N = int(N)
        self.psi_avg = float(psi_avg)
        self.wing_z = int(wing_z)
        self.wing_r = int(wing_r)
        self.rho_center = float(rho_center)
        self.rho_span = float(rho_span)
        self.dz = float(dz)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_params(cls, params: PhiParamsBase | dict[str, Any]) -> PhiField:
        """Build this field from its concrete model or a raw parameter payload.

        A field only needs its own parameter model and does not need to inspect
        the discriminated-union tag.  Registry selection remains the
        responsibility of :func:`phi_field_from_params`.
        """
        if isinstance(params, cls.params_model):
            validated_params = params
        else:
            validated_params = cls.params_model.model_validate(params)
        values = validated_params.model_dump()
        return cls(
            **cls._grid_from_values(values),
            **cls._per_type_from_values(values),
        )

    @classmethod
    def from_values(cls, values: dict[str, Any]) -> PhiField:
        """Build a field from a plain value dict (UI state or CLI args)."""
        return cls.from_params(values)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> PhiField:
        """Build a field from a parsed ``argparse`` namespace."""
        return cls.from_values(vars(args))

    @classmethod
    def add_parser_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add this distribution's CLI options.

        The parser contains options for every registered type.  Pydantic's
        discriminated union rejects options that do not belong to the selected
        ``--phi-type``.
        """

    @classmethod
    def make_ui_controls(cls) -> dict[str, Any]:
        """Return this distribution's controls for a nested Marimo dictionary."""
        return {}

    @classmethod
    def ui_layout(cls, controls: Any) -> Any:
        """Render this distribution's controls."""
        return controls

    @classmethod
    def sweep_param_names(cls) -> tuple[str, ...]:
        """Names of ``PhiSweep`` attributes used only by this distribution."""
        return ()

    @classmethod
    def type_description(cls) -> str:
        """Short Markdown description used in the type picker."""
        return ""

    @classmethod
    def _grid_from_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Extract the shared grid parameters from the canonical wire format."""

        return {
            "N": int(values["N"]),
            "psi_avg": float(values["psi_avg"]),
            "wing_z": int(values["wing_z"]),
            "wing_r": int(values["wing_r"]),
            "rho_center": float(values.get("rho_center", DEFAULT_RHO_CENTER)),
            "rho_span": float(values.get("rho_span", DEFAULT_RHO_SPAN)),
            "dz": float(values["dz"]),
        }

    @classmethod
    def _per_type_from_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Extract this distribution's own parameters from a value dict."""
        return {}

    # ------------------------------------------------------------------ #
    # Compute / export pipeline
    # ------------------------------------------------------------------ #

    def compute(self) -> PhiResult:
        """Build the wing-applied, renormalized initial phi field."""
        rho, z = build_phi_axes(
            N=self.N, rho_center=self.rho_center, rho_span=self.rho_span, dz=self.dz
        )
        errors = self.validate()
        if errors:
            raise ValueError(
                f"invalid configuration for {self.phi_type.label} phi field: "
                + "; ".join(errors)
            )
        phi = self.build(rho, z)

        phi_norm_with_wing = renormalize_phi(
            phi_add_wing(phi, self.wing_z, self.wing_r),
            rho,
            z,
            self.psi_avg,
            self.wing_z,
        )
        return PhiResult(
            rho=rho, z=z, phi_values=np.asarray(phi_norm_with_wing, dtype=np.float64)
        )

    def write_phi_h5(self, output_path: str | Path, result: PhiResult) -> Path:
        """Export ``result`` to the HDF5 file at ``output_path``."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(output_path), "w") as f:
            group = f.create_group("phi")
            _ = group.create_dataset(
                "values", data=np.asarray(result.phi_values, dtype=np.float64)
            )
            _ = group.create_dataset(
                "rho", data=np.asarray(result.rho, dtype=np.float64)
            )
            _ = group.create_dataset("z", data=np.asarray(result.z, dtype=np.float64))
            group.attrs["N"] = int(self.N)
            group.attrs["PSI"] = float(self.psi_avg)
            group.attrs["wing_z"] = int(self.wing_z)
            group.attrs["wing_r"] = int(self.wing_r)
            group.attrs["phi_type"] = self.phi_type.label
            group.attrs["storage_order"] = "phi[rho_idx, z_idx]"
            group.attrs["generated_by"] = "red_patterns/phi.py"
            group.attrs["normalization"] = "no runtime renormalization required"

            self.write_metadata(group)

        return output_path

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #

    @abstractmethod
    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        """Return the raw field ``phi[rho_idx, z_idx]`` for this distribution.

        ``compute`` applies wings and renormalization afterwards.
        """

    def validate(self) -> list[str]:
        """Return human-readable configuration errors, empty if valid."""
        return []

    def write_metadata(self, group: h5py.Group) -> None:
        """Write type-specific attributes to the ``phi`` HDF5 group."""

    def summary(self) -> list[str]:
        """Per-type parameter lines for the export CLI summary."""
        return []


class GaussianPhi(PhiField):
    phi_type = PhiType.GAUSSIAN
    params_model = GaussianPhiParams

    def __init__(
        self,
        *,
        gaussian_mu: float,
        gaussian_sigma: float,
        **grid: Any,
    ) -> None:
        super().__init__(**grid)
        self.gaussian_mu = float(gaussian_mu)
        self.gaussian_sigma = float(gaussian_sigma)

    @classmethod
    def _per_type_from_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        return {
            "gaussian_mu": values["gaussian_mu"],
            "gaussian_sigma": values["gaussian_sigma"],
        }

    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        return phi_gaussian(rho, z, self.psi_avg, self.gaussian_mu, self.gaussian_sigma)

    def validate(self) -> list[str]:
        if self.gaussian_sigma <= 0.0:
            return ["gaussian_sigma must be positive."]
        return []

    def write_metadata(self, group: h5py.Group) -> None:
        group.attrs["gaussian_mu"] = float(self.gaussian_mu)
        group.attrs["gaussian_sigma"] = float(self.gaussian_sigma)

    def summary(self) -> list[str]:
        return [
            f"gaussian_mu={self.gaussian_mu:.6e}",
            f"gaussian_sigma={self.gaussian_sigma:.6e}",
        ]

    @classmethod
    def add_parser_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--gaussian-mu", type=float, default=argparse.SUPPRESS)
        parser.add_argument("--gaussian-sigma", type=float, default=argparse.SUPPRESS)

    @classmethod
    def make_ui_controls(cls) -> dict[str, Any]:
        import marimo as mo

        return {
            "gaussian_mu": mo.ui.number(
                start=0.0,
                stop=2000.0,
                step=0.1,
                value=DEFAULT_GAUSSIAN_MU,
                label="$\\mu_\\rho \\; [\\frac{g}{L}]$",
            ),
            "gaussian_sigma": mo.ui.number(
                start=0.1,
                stop=15.0,
                step=0.1,
                value=DEFAULT_GAUSSIAN_SIGMA,
                label="$\\sigma_\\rho \\; [\\frac{g}{L}]$",
            ),
        }

    @classmethod
    def ui_layout(cls, controls: Any) -> Any:
        import marimo as mo

        return mo.vstack([mo.md("### Gaussian parameters"), controls])

    @classmethod
    def sweep_param_names(cls) -> tuple[str, ...]:
        return ("gaussian_mu", "gaussian_sigma")

    @classmethod
    def type_description(cls) -> str:
        return r"$$\varphi(\rho, z) = \langle \psi \rangle\,\mathcal{N}(\rho;\mu_\rho,\sigma_\rho)$$"


class GaussianBlobPhi(GaussianPhi):
    phi_type = PhiType.GAUSSIAN_BLOB
    params_model = GaussianBlobPhiParams

    def __init__(
        self,
        *,
        gaussian_blob_mu_z: float,
        gaussian_blob_sigma_z: float,
        **grid: Any,
    ) -> None:
        super().__init__(**grid)
        self.gaussian_blob_mu_z = float(gaussian_blob_mu_z)
        self.gaussian_blob_sigma_z = float(gaussian_blob_sigma_z)

    @classmethod
    def _per_type_from_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        return {
            **super()._per_type_from_values(values),
            "gaussian_blob_mu_z": values["gaussian_blob_mu_z"],
            "gaussian_blob_sigma_z": values["gaussian_blob_sigma_z"],
        }

    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        return phi_gaussian_blob(
            rho,
            z,
            self.psi_avg,
            self.gaussian_mu,
            self.gaussian_sigma,
            self.gaussian_blob_mu_z,
            self.gaussian_blob_sigma_z,
        )

    def validate(self) -> list[str]:
        errors = super().validate()
        if self.gaussian_blob_sigma_z <= 0.0:
            errors.append("gaussian_blob_sigma_z must be positive.")
        return errors

    def write_metadata(self, group: h5py.Group) -> None:
        super().write_metadata(group)
        group.attrs["gaussian_blob_mu_z"] = float(self.gaussian_blob_mu_z)
        group.attrs["gaussian_blob_sigma_z"] = float(self.gaussian_blob_sigma_z)

    def summary(self) -> list[str]:
        return super().summary() + [
            f"gaussian_blob_mu_z={self.gaussian_blob_mu_z:.6e}",
            f"gaussian_blob_sigma_z={self.gaussian_blob_sigma_z:.6e}",
        ]

    @classmethod
    def add_parser_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--gaussian-blob-mu-z", type=float, default=argparse.SUPPRESS
        )
        parser.add_argument(
            "--gaussian-blob-sigma-z", type=float, default=argparse.SUPPRESS
        )

    @classmethod
    def make_ui_controls(cls) -> dict[str, Any]:
        import marimo as mo

        return {
            **super().make_ui_controls(),
            "gaussian_blob_mu_z": mo.ui.number(
                start=0.0,
                stop=1.0,
                step=0.001,
                value=DEFAULT_GAUSSIAN_BLOB_MU_Z,
                label="$\\mu_z \\; [m]$",
            ),
            "gaussian_blob_sigma_z": mo.ui.number(
                start=0.001,
                stop=0.1,
                step=0.001,
                value=DEFAULT_GAUSSIAN_BLOB_SIGMA_Z,
                label="$\\sigma_z \\; [m]$",
            ),
        }

    @classmethod
    def ui_layout(cls, controls: Any) -> Any:
        import marimo as mo

        return mo.vstack([mo.md("### Gaussian blob parameters"), controls])

    @classmethod
    def sweep_param_names(cls) -> tuple[str, ...]:
        return super().sweep_param_names() + (
            "gaussian_blob_mu_z",
            "gaussian_blob_sigma_z",
        )

    @classmethod
    def type_description(cls) -> str:
        return r"$$\varphi(\rho,z)=\langle\psi\rangle\,\mathcal{N}(\rho,z;\mu_\rho,\sigma_\rho,\mu_z,\sigma_z)$$"


class HomogeneousPhi(PhiField):
    phi_type = PhiType.HOMOGENEOUS
    params_model = HomogeneousPhiParams

    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        return phi_homogeneous(rho, z, self.psi_avg)

    @classmethod
    def type_description(cls) -> str:
        return r"$\varphi(\rho,z)=\langle\psi\rangle/L_\rho$"


class SmoothHomogeneousPhi(PhiField):
    phi_type = PhiType.SMOOTH_HOMOGENEOUS
    params_model = SmoothHomogeneousPhiParams

    def __init__(self, *, rho_range: float, **grid: Any) -> None:
        super().__init__(**grid)
        self.rho_range = float(rho_range)

    @classmethod
    def _per_type_from_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        return {"rho_range": values["rho_range"]}

    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        return phi_smooth_homogeneous(
            rho, z, self.psi_avg, self.rho_center, self.rho_range, self.wing_r
        )

    def validate(self) -> list[str]:
        dr = self.rho_span / (self.N - 1)
        max_allowed = self.rho_span / 2.0 - self.wing_r * dr
        if self.rho_range > max_allowed:
            return ["rho_range and wing_r exceed the rho domain."]
        return []

    def write_metadata(self, group: h5py.Group) -> None:
        group.attrs["rho_range"] = float(self.rho_range)

    def summary(self) -> list[str]:
        return [f"rho_range={self.rho_range:.6e}"]

    @classmethod
    def add_parser_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--rho-range", type=float, default=argparse.SUPPRESS)

    @classmethod
    def make_ui_controls(cls) -> dict[str, Any]:
        import marimo as mo

        return {
            "rho_range": mo.ui.number(
                start=0.1,
                stop=100.0,
                step=0.1,
                value=DEFAULT_RHO_RANGE,
                label="ρ half-width (region)",
            )
        }

    @classmethod
    def ui_layout(cls, controls: Any) -> Any:
        import marimo as mo

        return mo.vstack([mo.md("### Smooth homogeneous parameters"), controls])

    @classmethod
    def sweep_param_names(cls) -> tuple[str, ...]:
        return ("rho_range",)

    @classmethod
    def type_description(cls) -> str:
        return (
            r"$\varphi(\rho,z)=\langle\psi\rangle$ in a ρ block, cosine-tapered to zero"
        )


class PerturbedSmoothHomogeneousPhi(SmoothHomogeneousPhi):
    phi_type = PhiType.PERTURBED_SMOOTH_HOMOGENEOUS
    params_model = PerturbedSmoothHomogeneousPhiParams

    def __init__(self, *, seed: int, amplitude: float, **grid: Any) -> None:
        super().__init__(**grid)
        self.seed = int(seed)
        self.amplitude = float(amplitude)

    @classmethod
    def _per_type_from_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        return {
            **super()._per_type_from_values(values),
            "seed": values["seed"],
            "amplitude": values["amplitude"],
        }

    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        return perturb_phi_z(
            super().build(rho, z),
            z,
            self.wing_z,
            amplitude=self.amplitude,
            seed=self.seed,
        )

    def write_metadata(self, group: h5py.Group) -> None:
        super().write_metadata(group)
        group.attrs["seed"] = self.seed
        group.attrs["amplitude"] = self.amplitude

    def summary(self) -> list[str]:
        return super().summary() + [
            f"seed={self.seed}",
            f"amplitude={self.amplitude:.6e}",
        ]

    @classmethod
    def add_parser_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
        _add_argument_if_missing(
            parser, "--amplitude", type=float, default=argparse.SUPPRESS
        )

    @classmethod
    def make_ui_controls(cls) -> dict[str, Any]:
        import marimo as mo

        return {
            **super().make_ui_controls(),
            "seed": mo.ui.number(start=0, stop=100, step=1, value=0, label="Seed"),
            "amplitude": mo.ui.number(
                start=0, stop=1, step=1e-6, value=1e-3, label="Amplitude"
            ),
        }

    @classmethod
    def ui_layout(cls, controls: Any) -> Any:
        import marimo as mo

        return mo.vstack(
            [mo.md("### Perturbed smooth homogeneous parameters"), controls]
        )

    @classmethod
    def sweep_param_names(cls) -> tuple[str, ...]:
        return super().sweep_param_names() + ("seed", "amplitude")

    @classmethod
    def type_description(cls) -> str:
        return r"$\varphi_{smooth}(\rho,z)[1 + A\eta(z)]$ with a seeded, mass-conserving perturbation"


class SingleModeSmoothHomogeneousPhi(SmoothHomogeneousPhi):
    phi_type = PhiType.SINGLE_MODE_SMOOTH_HOMOGENEOUS
    params_model = SingleModeSmoothHomogeneousPhiParams

    def __init__(self, *, amplitude: float, mode_number: int, **grid: Any) -> None:
        super().__init__(**grid)
        self.amplitude = float(amplitude)
        self.mode_number = int(mode_number)

    @classmethod
    def _per_type_from_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        return {
            **super()._per_type_from_values(values),
            "amplitude": values["amplitude"],
            "mode_number": values["mode_number"],
        }

    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        return perturb_phi_single_cosine_z(
            super().build(rho, z),
            z,
            self.wing_z,
            amplitude=self.amplitude,
            mode_number=self.mode_number,
        )

    def write_metadata(self, group: h5py.Group) -> None:
        super().write_metadata(group)
        group.attrs["amplitude"] = self.amplitude
        group.attrs["mode_number"] = self.mode_number

    def summary(self) -> list[str]:
        return super().summary() + [
            f"amplitude={self.amplitude:.6e}",
            f"mode_number={self.mode_number}",
        ]

    @classmethod
    def add_parser_arguments(cls, parser: argparse.ArgumentParser) -> None:
        _add_argument_if_missing(
            parser, "--amplitude", type=float, default=argparse.SUPPRESS
        )
        parser.add_argument("--mode-number", type=int, default=argparse.SUPPRESS)

    @classmethod
    def make_ui_controls(cls) -> dict[str, Any]:
        import marimo as mo

        return {
            **super().make_ui_controls(),
            "amplitude": mo.ui.number(
                start=0, stop=1, step=1e-6, value=1e-3, label="Amplitude"
            ),
            "mode_number": mo.ui.number(
                start=0, stop=1023, step=1, value=1, label="Mode number"
            ),
        }

    @classmethod
    def ui_layout(cls, controls: Any) -> Any:
        import marimo as mo

        return mo.vstack(
            [mo.md("### Single-mode smooth homogeneous parameters"), controls]
        )

    @classmethod
    def sweep_param_names(cls) -> tuple[str, ...]:
        return super().sweep_param_names() + ("amplitude", "mode_number")

    @classmethod
    def type_description(cls) -> str:
        return r"$\varphi_{smooth}(\rho,z)[1 + A\cos(m\pi x)]$ on the active z domain"


class SingleBinPhi(PhiField):
    phi_type = PhiType.SINGLE_BIN
    params_model = SingleBinPhiParams

    def __init__(self, *, single_bin_idx: int, **grid: Any) -> None:
        super().__init__(**grid)
        self.single_bin_idx = int(single_bin_idx)

    @classmethod
    def _per_type_from_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        return {"single_bin_idx": values["single_bin_idx"]}

    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        return phi_single_bin(rho, z, self.psi_avg, self.single_bin_idx)

    def validate(self) -> list[str]:
        if not (0 <= self.single_bin_idx < self.N):
            return [f"single_bin_idx must be in [0, {self.N - 1}] for N={self.N}."]
        return []

    def write_metadata(self, group: h5py.Group) -> None:
        group.attrs["single_bin_idx"] = int(self.single_bin_idx)

    def summary(self) -> list[str]:
        return [f"single_bin_idx={self.single_bin_idx}"]

    @classmethod
    def add_parser_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--single-bin-idx", type=int, default=argparse.SUPPRESS)

    @classmethod
    def make_ui_controls(cls) -> dict[str, Any]:
        import marimo as mo

        return {
            "single_bin_idx": mo.ui.number(
                start=0,
                stop=1023,
                step=1,
                value=DEFAULT_SINGLE_BIN_IDX,
                label="ρ bin index",
            )
        }

    @classmethod
    def ui_layout(cls, controls: Any) -> Any:
        import marimo as mo

        return mo.vstack([mo.md("### Single-bin parameters"), controls])

    @classmethod
    def sweep_param_names(cls) -> tuple[str, ...]:
        return ("single_bin_idx",)

    @classmethod
    def type_description(cls) -> str:
        return r"$\varphi(\rho_i,z)=\langle\psi\rangle$ for one ρ bin"


class LinearFullRidgePhi(PhiField):
    phi_type = PhiType.LINEAR_FULL_RIDGE
    params_model = LinearFullRidgePhiParams

    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        return phi_linear_full_ridge(rho, z, self.psi_avg, self.rho_center)

    @classmethod
    def type_description(cls) -> str:
        return (
            r"$\varphi(\rho,z)$ on the one-bin diagonal: equilibrium state "
            r"of the LINEAR-FULL gradient"
        )


class SingleModeLinearFullRidgePhi(LinearFullRidgePhi):
    phi_type = PhiType.SINGLE_MODE_LINEAR_FULL_RIDGE
    params_model = SingleModeLinearFullRidgePhiParams

    def __init__(self, *, amplitude: float, mode_number: int, **grid: Any) -> None:
        super().__init__(**grid)
        self.amplitude = float(amplitude)
        self.mode_number = int(mode_number)

    @classmethod
    def _per_type_from_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        return {
            "amplitude": values["amplitude"],
            "mode_number": values["mode_number"],
        }

    def build(self, rho: Array1F, z: Array1F) -> Array2F:
        return perturb_phi_single_cosine_z(
            super().build(rho, z),
            z,
            self.wing_z,
            amplitude=self.amplitude,
            mode_number=self.mode_number,
        )

    def write_metadata(self, group: h5py.Group) -> None:
        group.attrs["amplitude"] = self.amplitude
        group.attrs["mode_number"] = self.mode_number

    def summary(self) -> list[str]:
        return [
            f"amplitude={self.amplitude:.6e}",
            f"mode_number={self.mode_number}",
        ]

    @classmethod
    def add_parser_arguments(cls, parser: argparse.ArgumentParser) -> None:
        _add_argument_if_missing(
            parser, "--amplitude", type=float, default=argparse.SUPPRESS
        )
        _add_argument_if_missing(
            parser, "--mode-number", type=int, default=argparse.SUPPRESS
        )

    @classmethod
    def make_ui_controls(cls) -> dict[str, Any]:
        import marimo as mo

        return {
            "amplitude": mo.ui.number(
                start=0, stop=1, step=1e-6, value=1e-3, label="Amplitude"
            ),
            "mode_number": mo.ui.number(
                start=0, stop=1023, step=1, value=1, label="Mode number"
            ),
        }

    @classmethod
    def ui_layout(cls, controls: Any) -> Any:
        import marimo as mo

        return mo.vstack(
            [mo.md("### Single-mode linear gradient diagonal parameters"), controls]
        )

    @classmethod
    def sweep_param_names(cls) -> tuple[str, ...]:
        return ("amplitude", "mode_number")

    @classmethod
    def type_description(cls) -> str:
        return (
            r"$\varphi_{ridge}(\rho,z)[1 + A\cos(m\pi x)]$ on the active z domain"
        )


PHI_FIELD_TYPES: dict[PhiType, type[PhiField]] = {
    PhiType.GAUSSIAN: GaussianPhi,
    PhiType.GAUSSIAN_BLOB: GaussianBlobPhi,
    PhiType.HOMOGENEOUS: HomogeneousPhi,
    PhiType.SMOOTH_HOMOGENEOUS: SmoothHomogeneousPhi,
    PhiType.PERTURBED_SMOOTH_HOMOGENEOUS: PerturbedSmoothHomogeneousPhi,
    PhiType.SINGLE_MODE_SMOOTH_HOMOGENEOUS: SingleModeSmoothHomogeneousPhi,
    PhiType.SINGLE_BIN: SingleBinPhi,
    PhiType.LINEAR_FULL_RIDGE: LinearFullRidgePhi,
    PhiType.SINGLE_MODE_LINEAR_FULL_RIDGE: SingleModeLinearFullRidgePhi,
}


def phi_field_from_params(params: PhiParamsBase | dict[str, Any]) -> PhiField:
    """Return the :class:`PhiField` subclass matching a ``PhiParams`` payload.

    ``params`` may be a validated union member or a plain dict.
    """
    payload = params.model_dump() if isinstance(params, PhiParamsBase) else params
    validated_params = PHI_PARAMS_ADAPTER.validate_python(payload)
    return PHI_FIELD_TYPES[validated_params.phi_type].from_params(validated_params)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_export_parser(prog: str = "phi export") -> argparse.ArgumentParser:
    """Build the flat export CLI from shared and per-type definitions."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Export a CUDA-compatible initial phi field to HDF5.",
    )
    parser.add_argument("--output", required=True, help="Path to the output HDF5 file.")
    parser.add_argument(
        "--phi-type",
        required=True,
        choices=sorted(t.value for t in PhiType),
        help="Initial phi distribution to use.",
    )
    parser.add_argument(
        "--psi-avg", required=True, type=float, help="Average volume fraction."
    )
    parser.add_argument(
        "--N", type=int, default=DEFAULT_N, help="Grid size in rho and z."
    )
    parser.add_argument(
        "--wing-z", type=int, default=DEFAULT_WING, help="Wing size in z direction."
    )
    parser.add_argument(
        "--wing-r", type=int, default=DEFAULT_WING, help="Wing size in rho direction."
    )
    parser.add_argument("--rho-center", type=float, default=DEFAULT_RHO_CENTER)
    parser.add_argument("--rho-span", type=float, default=DEFAULT_RHO_SPAN)
    parser.add_argument("--dz", type=float, default=DEFAULT_DZ)
    for field_cls in PHI_FIELD_TYPES.values():
        field_cls.add_parser_arguments(parser)
    return parser


def validate_export_namespace(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> PhiField:
    """Validate a CLI payload through the same Pydantic union as sweeps."""
    payload = {key: value for key, value in vars(args).items() if key != "output"}
    try:
        params = PHI_PARAMS_ADAPTER.validate_python(payload)
        field = phi_field_from_params(params)
    except Exception as exc:
        parser.error(str(exc))
    errors = field.validate()
    if errors:
        parser.error("\n".join(errors))
    return field


def run_export_cli(argv: list[str], prog: str = "phi export") -> int:
    parser = build_export_parser(prog)
    args = parser.parse_args(argv)
    field = validate_export_namespace(parser, args)
    result = field.compute()
    output_path = field.write_phi_h5(args.output, result)

    print(f"Exported initial phi to {output_path}")
    print(
        f"Stored shape {result.phi_values.shape} on /phi/values with storage order "
        + "phi[rho_idx, z_idx]"
    )
    summary = [
        f"phi_type={field.phi_type}",
        f"PSI={field.psi_avg:.6e}",
        f"N={field.N}",
        f"wing_z={field.wing_z}",
        f"wing_r={field.wing_r}",
        f"rho_center={field.rho_center:.6e}",
        f"rho_span={field.rho_span:.6e}",
        f"DZ={field.dz:.6e}",
    ]
    summary.extend(field.summary())
    print("Used parameters: " + ", ".join(summary))
    return 0


# --------------------------------------------------------------------------- #
# UI factory (marimo). Imports are lazy so the compute path stays headless.
# --------------------------------------------------------------------------- #


def _common_phi_ui_controls(*, psi_avg: float) -> dict[str, Any]:
    import marimo as mo

    return {
        "psi_avg": mo.ui.number(
            start=0.0,
            stop=1.0,
            step=0.001,
            value=psi_avg,
            label="$\\langle \\psi \\rangle$",
        ),
        "N": mo.ui.dropdown(
            options=[2**x for x in range(6, 12)], value=DEFAULT_N, label="N"
        ),
        "wing_z": mo.ui.number(
            start=0, stop=1000, step=1, value=DEFAULT_WING, label="z wing size"
        ),
        "wing_r": mo.ui.number(
            start=0, stop=1000, step=1, value=DEFAULT_WING, label="ρ wing size"
        ),
    }


def make_phi_ui(*, psi_avg: float = DEFAULT_PSI_AVG):
    """Build nested registered controls for common and per-type phi settings.

    Variant dictionaries intentionally stay registered while hidden, so changing
    the selected type does not discard a user's previous inputs.  Only the
    active dictionary is included in the Pydantic payload.
    """
    import marimo as mo

    variants = {
        phi_type.value: mo.ui.dictionary(field_cls.make_ui_controls())
        for phi_type, field_cls in PHI_FIELD_TYPES.items()
    }
    return mo.ui.dictionary(
        {
            "common": mo.ui.dictionary(_common_phi_ui_controls(psi_avg=psi_avg)),
            "phi_type": mo.ui.dropdown(
                options=[phi_type.label for phi_type in PHI_FIELD_TYPES],
                value=PhiType.GAUSSIAN.label,
                label="Initial phi type",
            ),
            # Marimo dictionaries are UIElements at runtime, but its public
            # type information does not model nested dictionaries as such.
            "variants": mo.ui.dictionary(variants),  # pyright: ignore[reportArgumentType]
        }
    )


def phi_ui_layout(phi_ui: Any) -> Any:
    """Render common controls and only the currently selected type's controls."""
    import marimo as mo

    phi_type = LABEL_MAP[str(phi_ui["phi_type"].value)]
    common = phi_ui["common"]
    controls = phi_ui["variants"][phi_type.value]
    field_cls = PHI_FIELD_TYPES[phi_type]
    return mo.vstack(
        [
            mo.md("### Average volume fraction"),
            common["psi_avg"],
            mo.md("### Grid size"),
            common["N"],
            mo.hstack([common["wing_z"], common["wing_r"]]),
            mo.md("### Distribution type"),
            phi_ui["phi_type"],
            field_cls.ui_layout(controls),
        ],
        gap=0.5,
    )


def phi_field_from_ui(value: dict[str, Any]) -> PhiField:
    """Validate the selected nested UI payload and construct its field."""
    phi_type = LABEL_MAP[str(value["phi_type"])]
    payload = {
        "phi_type": phi_type,
        **value["common"],
        **value["variants"][phi_type.value],
    }
    payload["dz"] = DEFAULT_Z_SYSTEM_SIZE / float(payload["N"])
    params = PHI_PARAMS_ADAPTER.validate_python(payload)
    return phi_field_from_params(params)


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
