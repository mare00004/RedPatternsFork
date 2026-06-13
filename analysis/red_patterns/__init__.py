"""Red Patterns analysis library.

Public API is exposed lazily so headless utilities can import only the modules
they need without pulling in plotting dependencies from unrelated submodules.
"""

from __future__ import annotations

from importlib import import_module

_SUBMODULES = {"runs", "kernel", "phi", "sim", "sweep_jobs"}

_EXPORTS = {
    "Array1F": ("runs", "Array1F"),
    "Array2F": ("runs", "Array2F"),
    "Array3F": ("runs", "Array3F"),
    "ConvVariant": ("runs", "ConvVariant"),
    "ModelParamsData": ("runs", "ModelParamsData"),
    "RunConfig": ("runs", "RunConfig"),
    "RunData": ("runs", "RunData"),
    "RunParamsData": ("runs", "RunParamsData"),
    "TaylVariant": ("runs", "TaylVariant"),
    "cli_args_from_run_h5": ("runs", "cli_args_from_run_h5"),
    "find_peaks": ("runs", "find_peaks"),
    "get_rbc_cmap": ("runs", "get_rbc_cmap"),
    "plot_psi": ("runs", "plot_psi"),
    "plot_psi_arrays": ("runs", "plot_psi_arrays"),
    "ConvSweep": ("sweep_jobs", "ConvSweep"),
    "Gradient": ("sweep_jobs", "Gradient"),
    "KernelSweep": ("sweep_jobs", "KernelSweep"),
    "PhiSweep": ("sweep_jobs", "PhiSweep"),
    "Range": ("sweep_jobs", "Range"),
    "SimulationSweep": ("sweep_jobs", "SimulationSweep"),
    "TaylSweep": ("sweep_jobs", "TaylSweep"),
    "Variant": ("sweep_jobs", "Variant"),
    "combine_sweeps": ("sweep_jobs", "combine_sweeps"),
    "find_run_by_id": ("sweep_jobs", "find_run_by_id"),
    "load_runs_jsonl": ("sweep_jobs", "load_runs_jsonl"),
    "normalize_runs": ("sweep_jobs", "normalize_runs"),
    "run_ids_from_runs": ("sweep_jobs", "run_ids_from_runs"),
    "runs_to_jsonl": ("sweep_jobs", "runs_to_jsonl"),
    "write_run_id_queue": ("sweep_jobs", "write_run_id_queue"),
    "write_runs_jsonl": ("sweep_jobs", "write_runs_jsonl"),
    "write_sweep_export": ("sweep_jobs", "write_sweep_export"),
}

__all__ = [
    "runs",
    "kernel",
    "phi",
    "sim",
    "sweep_jobs",
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
    "ConvSweep",
    "Gradient",
    "KernelSweep",
    "PhiSweep",
    "Range",
    "SimulationSweep",
    "TaylSweep",
    "Variant",
    "combine_sweeps",
    "find_run_by_id",
    "load_runs_jsonl",
    "normalize_runs",
    "run_ids_from_runs",
    "runs_to_jsonl",
    "write_run_id_queue",
    "write_runs_jsonl",
    "write_sweep_export",
]


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(f".{module_name}", __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
