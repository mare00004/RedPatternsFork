"""Shared discovery of modern sweep directories for analysis notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import RunPayload
from .sweep_jobs import load_runs_jsonl

class SweepCatalogError(ValueError):
    """A selected directory cannot be used as a modern sweep."""


@dataclass(frozen=True)
class SweepEntry:
    """One configured run and the location where its result is expected."""

    run: RunPayload
    run_h5: Path
    h5_exists: bool

    @property
    def run_id(self) -> str:
        return self.run.run_id


@dataclass(frozen=True)
class SweepCatalog:
    """Parsed ``runs.jsonl`` plus resolved ``results/<run_id>/run.h5`` paths."""

    root: Path
    entries: tuple[SweepEntry, ...]

    @property
    def runs(self) -> tuple[RunPayload, ...]:
        return tuple(entry.run for entry in self.entries)


def load_sweep_catalog(directory: str | Path) -> SweepCatalog:
    """Load a JSONL/results sweep directory without loading HDF5 field data."""

    root = Path(directory)
    runs_path = root / "runs.jsonl"
    if not runs_path.is_file():
        raise SweepCatalogError(f"`{root}` does not contain `runs.jsonl`.")

    try:
        runs = load_runs_jsonl(runs_path)
    except (OSError, ValueError) as exc:
        raise SweepCatalogError(f"Could not read `{runs_path}`: {exc}") from exc

    entries = tuple(
        SweepEntry(
            run=run,
            run_h5=root / "results" / run.run_id / "run.h5",
            h5_exists=(root / "results" / run.run_id / "run.h5").is_file(),
        )
        for run in runs
    )
    return SweepCatalog(root=root, entries=entries)


def sweep_directory_picker(mo: Any, *, initial_path: Path, label: str) -> Any:
    """Create the consistent single-directory picker used by sweep notebooks."""

    return mo.ui.file_browser(
        initial_path=initial_path,
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="directory",
        label=label,
    )


def selected_sweep_catalog(mo: Any, picker: Any) -> tuple[SweepCatalog | None, Any]:
    """Resolve a picker into a catalog and a standard Marimo loading status."""

    selected_path = picker.path(0) if picker.value else None
    if selected_path is None:
        return None, mo.md("Waiting for a sweep directory selection...")
    try:
        catalog = load_sweep_catalog(selected_path)
    except SweepCatalogError as exc:
        return None, mo.callout(str(exc), kind="warn")
    return catalog, mo.md(f"Loaded `{len(catalog.entries)}` configured runs from `{catalog.root}`.")
