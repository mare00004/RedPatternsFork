"""Shared domain enums for the red-patterns analysis code.

Consolidates the ``StrEnum`` types previously defined in ``sweep_jobs``
(``Variant``, ``Gradient``, ``PhiMode``/``ClosureMode``/``PairDistributionMode``),
``phi`` (``PhiType``) and ``kernel`` (``PhiType``/``ClosureType``/``PDFType``)
into a single source of truth.  ``PhiMode``, ``ClosureMode`` and
``PairDistributionMode`` are drop-in aliases kept for callers that used the
older sweep names.
"""

from __future__ import annotations

from enum import StrEnum


class Variant(StrEnum):
    TAYLOR = "taylor"
    CONVOLUTION = "convolution"


class Gradient(StrEnum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    ZERO = "zero"


class PhiType(StrEnum):
    GAUSSIAN = "gaussian"
    GAUSSIAN_BLOB = "gaussian_blob"
    HOMOGENEOUS = "homogeneous"
    SMOOTH_HOMOGENEOUS = "smooth_homogeneous"
    PERTURBED_SMOOTH_HOMOGENEOUS = "perturbed_smooth_homogeneous"
    SINGLE_BIN = "single_bin"

    @property
    def label(self) -> str:
        return _PHI_TYPE_LABELS[self]


class ClosureType(StrEnum):
    FORCE = "force"
    POTENTIAL = "potential"

    @property
    def label(self) -> str:
        return _CLOSURE_TYPE_LABELS[self]


class PDFType(StrEnum):
    MEAN_FIELD = "mean-field"
    NEAREST_NEIGHBOR = "nearest-neighbor"
    EXPONENTIAL = "exponential"

    @property
    def label(self) -> str:
        return _PDF_TYPE_LABELS[self]


class KernelType(StrEnum):
    ORIGINAL = "original"
    HNC = "hnc"

    @property
    def label(self) -> str:
        return _KERNEL_TYPE_LABELS[self]


# Backwards-compatible aliases used by the sweep layer.
PhiMode = PhiType
ClosureMode = ClosureType
PairDistributionMode = PDFType


_PHI_TYPE_LABELS = {
    PhiType.GAUSSIAN: "Gaussian",
    PhiType.GAUSSIAN_BLOB: "Gaussian Blob",
    PhiType.HOMOGENEOUS: "Homogeneous",
    PhiType.SMOOTH_HOMOGENEOUS: "Smooth Homogeneous",
    PhiType.PERTURBED_SMOOTH_HOMOGENEOUS: "Perturbed Smooth Homogeneous",
    PhiType.SINGLE_BIN: "Single Bin",
}

_CLOSURE_TYPE_LABELS = {
    ClosureType.FORCE: "Force closure",
    ClosureType.POTENTIAL: "Potential closure",
}

_PDF_TYPE_LABELS = {
    PDFType.MEAN_FIELD: "Mean Field",
    PDFType.NEAREST_NEIGHBOR: "Nearest Neighbor",
    PDFType.EXPONENTIAL: "Exponential",
}

_KERNEL_TYPE_LABELS = {
    KernelType.ORIGINAL: "Original (Lennard-Jones)",
    KernelType.HNC: "HNC effective Morse",
}

__all__ = [
    "Variant",
    "Gradient",
    "PhiType",
    "ClosureType",
    "PDFType",
    "KernelType",
    "PhiMode",
    "ClosureMode",
    "PairDistributionMode",
]
