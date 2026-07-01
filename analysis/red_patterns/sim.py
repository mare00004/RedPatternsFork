"""Helpers for launching the ``red-patterns`` binary and reading its progress.

Headless (no marimo / matplotlib) so it is usable from scripts. The live
progress-bar loop itself stays in the notebook that drives it; this module only
provides the pure helpers (binary discovery, CLI assembly, progress parsing).

Ported from ``analysis/marimo_run_monitor.py`` so notebooks can import them
instead of re-running named cells.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

DEFAULT_POLL_SEC = 0.5
DEFAULT_STALE_SEC = 30.0


def locate_binary(repo_root: Path) -> Path:
    """Return the first existing ``red-patterns`` build, else the preferred path."""
    candidates = (
        repo_root / "build" / "release" / "red-patterns",
        repo_root / "build" / "dev-debug" / "red-patterns",
        repo_root / "build" / "red-patterns",
        repo_root / "build" / "red_patterns",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def estimate_total_steps(t_final: float, dt: float) -> int:
    if dt <= 0:
        return 1
    return max(1, int(round(t_final / dt)))


def build_cli_args(
    *,
    binary_path: Path,
    mode: str,
    out_dir: Path,
    phi_path: Path,
    kernel_path: Path | None,
    gradient: str,
    t_final: float,
    dt: float,
    save_every: int,
    nu: float,
    mu: float,
) -> list[str]:
    """Assemble the ``red-patterns`` CLI. ``mode`` is ``"Taylor"`` or ``"Convolution"``."""
    args = [
        str(binary_path),
        "--use-taylor" if mode == "Taylor" else "--use-convolution",
        f"--T={t_final}",
        f"--DT={dt}",
        f"--NO={save_every}",
        f"--gradient={gradient}",
        f"--phi-file={phi_path}",
        f"--out-dir={out_dir}",
        "--store=phi",
        "--store=psi",
        "--store=percoll",
    ]
    if mode == "Taylor":
        args.extend([f"--NU={nu}", f"--MU={mu}"])
    elif kernel_path is not None:
        args.append(f"--kernel-file={kernel_path}")
    return args


def read_progress(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _seconds_to_text(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    seconds_f = float(seconds)
    if not math.isfinite(seconds_f) or seconds_f < 0:
        return "unknown"
    total_seconds = int(round(seconds_f))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def progress_summary(
    *,
    snapshot: dict[str, object] | None,
    t_final: float,
    is_waiting: bool,
    is_stale: bool,
    returncode: int | None,
) -> str:
    """Render a markdown summary of a progress snapshot for display."""
    if snapshot is None:
        status = "starting..."
        if returncode is not None:
            status = (
                "finished before progress file was observed"
                if returncode == 0
                else f"failed (returncode={returncode})"
            )
        return status

    step = int(snapshot.get("step", 0))
    total_steps = int(snapshot.get("total_steps", 0))
    elapsed_sec = float(snapshot.get("elapsed_sec", 0.0))
    remaining_sec = float(snapshot.get("remaining_sec", 0.0))
    sim_time_sec = float(snapshot.get("sim_time_sec", 0.0))
    status = str(snapshot.get("status", "unknown"))
    error = str(snapshot.get("error", "")).strip()

    if is_stale and status == "running":
        status = "waiting for next checkpoint"
    elif is_waiting and status == "running":
        status = "starting..."

    lines = [
        f"`status = {status}`",
        f"`step = {step} / {total_steps}`",
        f"`elapsed = {_seconds_to_text(elapsed_sec)}`",
        f"`remaining = {_seconds_to_text(remaining_sec)}`",
        f"`sim time = {sim_time_sec:.6f} / {t_final:.6f} s`",
    ]
    if returncode is not None:
        lines.append(f"`returncode = {returncode}`")
    if error:
        lines.append(f"`error = {error}`")
    return "\n\n".join(lines)
