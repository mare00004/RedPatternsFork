"""Red Patterns analysis library.

Public API is re-exported here so existing imports keep working after the package
split, e.g. ``from red_patterns import RunData, plot_psi, get_rbc_cmap``.

Submodules:
- :mod:`red_patterns.runs`   — run.h5 loading + psi plotting (RunData, plot_psi, ...)
- :mod:`red_patterns.kernel` — kernel generation (KernelConfig, compute_kernel, UI)
- :mod:`red_patterns.phi`    — initial phi generation (PhiConfig, compute_phi, UI)
"""

from __future__ import annotations

from . import kernel, phi, runs, sim
from .runs import (
    Array1F,
    Array2F,
    Array3F,
    ConvVariant,
    ModelParamsData,
    RunConfig,
    RunData,
    RunParamsData,
    TaylVariant,
    cli_args_from_run_h5,
    find_peaks,
    get_rbc_cmap,
    plot_psi,
    plot_psi_arrays,
)

__all__ = [
    # submodules
    "runs",
    "kernel",
    "phi",
    "sim",
    # runs API (kept flat for backwards compatibility)
    "Array1F",
    "Array2F",
    "Array3F",
    "ConvVariant",
    "ModelParamsData",
    "RunConfig",
    "RunData",
    "RunParamsData",
    "TaylVariant",
    "cli_args_from_run_h5",
    "find_peaks",
    "get_rbc_cmap",
    "plot_psi",
    "plot_psi_arrays",
]
