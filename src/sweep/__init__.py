"""Sweep execution helpers."""

from importlib import import_module

__all__ = ["run_one"]


def __getattr__(name: str):
    if name == "run_one":
        module = import_module(".run_one", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
