"""Pydantic models for validated simulation run payloads.

These models replace the ``dict[str, Any]`` payloads that the sweep layer
previously produced and hand-validated.  Parsing and validation is delegated
to Pydantic v2, and the discriminated union :data:`RunPayload` decides between
Taylor and Convolution runs from the ``variant`` field.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from .types import ClosureType, Gradient, KernelType, PDFType, PhiType, Variant

_DEFAULT_N = 256
_DEFAULT_WING = 30
_DEFAULT_RHO_CENTER = 1100.0
_DEFAULT_RHO_SPAN = 30.0
_DEFAULT_DZ = 0.000267651
_DEFAULT_PSI_AVG = 0.02


class PhiParamsBase(BaseModel):
    """Parameters shared by every phi-generation distribution.

    These mirror the flat ``params`` dict emitted by :class:
    `red_patterns.sweep_jobs.PhiSweep` so existing ``runs.jsonl`` payloads
    still round-trip.  ``phi_type`` is the discriminated-union tag.
    """

    model_config = ConfigDict(extra="forbid")

    phi_type: PhiType
    psi_avg: float
    N: int = _DEFAULT_N
    wing: int = _DEFAULT_WING
    rho_center: float = _DEFAULT_RHO_CENTER
    rho_span: float = _DEFAULT_RHO_SPAN
    dz: float = _DEFAULT_DZ


class GaussianPhiParams(PhiParamsBase):
    phi_type: Literal[PhiType.GAUSSIAN]
    gaussian_mu: float
    gaussian_sigma: float


class GaussianBlobPhiParams(PhiParamsBase):
    phi_type: Literal[PhiType.GAUSSIAN_BLOB]
    gaussian_mu: float
    gaussian_sigma: float
    gaussian_blob_mu_z: float
    gaussian_blob_sigma_z: float


class HomogeneousPhiParams(PhiParamsBase):
    phi_type: Literal[PhiType.HOMOGENEOUS]


class SmoothHomogeneousPhiParams(PhiParamsBase):
    phi_type: Literal[PhiType.SMOOTH_HOMOGENEOUS]
    rho_range: float


class SingleBinPhiParams(PhiParamsBase):
    phi_type: Literal[PhiType.SINGLE_BIN]
    single_bin_idx: int


PhiParams = Annotated[
    GaussianPhiParams
    | GaussianBlobPhiParams
    | HomogeneousPhiParams
    | SmoothHomogeneousPhiParams
    | SingleBinPhiParams,
    Field(discriminator="phi_type"),
]

PHI_PARAMS_ADAPTER: TypeAdapter[PhiParams] = TypeAdapter(PhiParams)


class KernelParamsBase(BaseModel):
    """Common, validated parameters for a generated convolution kernel."""

    model_config = ConfigDict(extra="forbid")

    kernel_type: KernelType
    kernel_n: int
    dz: float
    subdiv: int


class OriginalKernelParams(KernelParamsBase):
    kernel_type: Literal[KernelType.ORIGINAL]
    closure: ClosureType
    pair_distribution: PDFType
    U: float
    sigma: float
    g0: float | None = None
    nn_d: float | None = None
    nn_sigma: float | None = None
    lambda_: float | None = None

    @model_validator(mode="after")
    def validate_distribution(self) -> "OriginalKernelParams":
        if self.kernel_n < 3 or self.kernel_n % 2 == 0 or self.dz <= 0 or self.subdiv <= 0 or self.sigma <= 0:
            raise ValueError("kernel_n must be odd >= 3; dz, subdiv, and sigma must be positive.")
        if self.pair_distribution == PDFType.NEAREST_NEIGHBOR and None in (self.g0, self.nn_d, self.nn_sigma):
            raise ValueError("nearest-neighbor distribution requires g0, nn_d, and nn_sigma.")
        if self.pair_distribution == PDFType.EXPONENTIAL and (self.lambda_ is None or self.U == 0):
            raise ValueError("exponential distribution requires lambda_ and nonzero U.")
        return self


class HNCKernelParams(KernelParamsBase):
    kernel_type: Literal[KernelType.HNC]
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float

    @model_validator(mode="after")
    def validate_hnc(self) -> "HNCKernelParams":
        if self.kernel_n < 3 or self.kernel_n % 2 == 0 or self.dz <= 0 or self.subdiv <= 0:
            raise ValueError("kernel_n must be odd >= 3 and dz/subdiv must be positive.")
        if self.b == 0 or self.alpha <= 0 or self.beta <= 0 or self.gamma <= 0:
            raise ValueError("b must be nonzero and alpha, beta, gamma must be positive.")
        return self


KernelParams = Annotated[
    OriginalKernelParams | HNCKernelParams, Field(discriminator="kernel_type")
]
KERNEL_PARAMS_ADAPTER: TypeAdapter[KernelParams] = TypeAdapter(KernelParams)


class GenerateParams(BaseModel):
    """Nested ``{"mode": "generate", "params": {...}}`` block (kernel / generic)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["generate"] = "generate"
    params: dict[str, Any] = Field(default_factory=dict)


class KernelGenerateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["generate"] = "generate"
    params: KernelParams


class PhiGenerateParams(BaseModel):
    """Nested ``{"mode": "generate", "params": {...}}`` block for phi.

    Unlike :class:`GenerateParams`, ``params`` is validated against the
    :data:`PhiParams` discriminated union on ``phi_type``.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["generate"] = "generate"
    params: PhiParams


class BaseRun(BaseModel):
    """Fields common to every simulation run payload."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = ""
    variant: Variant
    N: int
    T: float
    DT: float
    storeTime: float
    gradient: Gradient
    phi: PhiGenerateParams


class TaylorRun(BaseRun):
    """A run payload for the ``--use-taylor`` variant."""

    variant: Literal[Variant.TAYLOR] = Variant.TAYLOR
    NU: float
    MU: float


class ConvRun(BaseRun):
    """A run payload for the ``--use-convolution`` variant."""

    variant: Literal[Variant.CONVOLUTION] = Variant.CONVOLUTION
    kernel: KernelGenerateParams


RunPayload = Annotated[TaylorRun | ConvRun, Field(discriminator="variant")]

run_payload_adapter: TypeAdapter[RunPayload] = TypeAdapter(RunPayload)
