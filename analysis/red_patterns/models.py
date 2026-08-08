"""Pydantic models for validated simulation run payloads.

These models replace the ``dict[str, Any]`` payloads that the sweep layer
previously produced and hand-validated.  Parsing and validation is delegated
to Pydantic v2, and the discriminated union :data:`RunPayload` decides between
Taylor and Convolution runs from the ``variant`` field.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .types import Gradient, Variant


class GenerateParams(BaseModel):
    """Nested ``{"mode": "generate", "params": {...}}`` block."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["generate"] = "generate"
    params: dict[str, Any] = Field(default_factory=dict)


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
    phi: GenerateParams


class TaylorRun(BaseRun):
    """A run payload for the ``--use-taylor`` variant."""

    variant: Literal[Variant.TAYLOR] = Variant.TAYLOR
    NU: float
    MU: float


class ConvRun(BaseRun):
    """A run payload for the ``--use-convolution`` variant."""

    variant: Literal[Variant.CONVOLUTION] = Variant.CONVOLUTION
    kernel: GenerateParams


RunPayload = Annotated[TaylorRun | ConvRun, Field(discriminator="variant")]

run_payload_adapter: TypeAdapter[RunPayload] = TypeAdapter(RunPayload)
