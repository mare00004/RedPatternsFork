# pyright: reportUnknownVariableType=false
# pyright: reportAny=false
"""Kernel generation library.

This module builds the effective 1-D interaction kernel
$K(x)$ used by the DDFT simulation and exports it to HDF5.

CLI usage (in a marimo notebook's script-mode cell)
---------------------------------------------------

>>> if mo.app_meta().mode == "script" and sys.argv[1:2] == ["export"]:
...     raise SystemExit(run_export_cli(sys.argv[2:], prog="kernel.py export"))
...
Then run: ``uv run analysis/kernel.py export --output kernel.h5 --closure force --pair-distribution nearest-neighbor ...``

Marimo notebook usage
---------------------
>>> ui = make_kernel_ui()                                    # cell 1
>>> kernel_ui_layout(ui)                                     # cell 2 — display
>>> cfg = kernel_config_from_ui(ui.value, "kernel.h5")       # cell 3
>>> result = compute_kernel(cfg)                             # cell 4
>>> write_kernel_h5(cfg.output_path, result, cfg)            # cell 5
>>> plot_kernel(result, cfg)                                 # optional cell

As with :mod:`red_patterns.phi`, the compute path imports
neither marimo nor matplotlib; the UI factory imports them lazily.

For mathematical details see ``analysis/kernel.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from enum import StrEnum

import h5py
import numpy as np
from numpy.typing import NDArray

Array1F = NDArray[np.float64]

# --------------------------------------------------------------------------- #
# Physical constants / labels
# --------------------------------------------------------------------------- #

# Potential
SIGMA = 5.6e-6  # 5.6 micrometers converted to meters
V = 90e-18

# Pair Distribution Function
G0 = 4.0e7
SIGMA_C = 0.5e-6
EQ_DIST = 6.585467201064237091254725819933213415424688719213008880615234375e-06


class PDFType(StrEnum):
    MEAN_FIELD = "mean-field"
    NEAREST_NEIGHBOR = "nearest-neighbor"
    EXPONENTIAL = "exponential"

    @property
    def label(self) -> str:
        return _PDF_DISPLAY[self]


_PDF_DISPLAY = {
    PDFType.MEAN_FIELD: "Mean Field",
    PDFType.NEAREST_NEIGHBOR: "Nearest Neighbor",
    PDFType.EXPONENTIAL: "Exponential",
}


class ClosureType(StrEnum):
    FORCE = "force"
    POTENTIAL = "potential"

    @property
    def label(self) -> str:
        return _CLOSURE_DISPLAY[self]


_CLOSURE_DISPLAY = {
    ClosureType.FORCE: "Force closure",
    ClosureType.POTENTIAL: "Potential closure",
}


def latex_scientific(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)
    return rf"{mantissa:.6f} \cdot 10^{{{exponent}}}"


# --------------------------------------------------------------------------- #
# Config / result dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KernelConfig:
    """All parameters needed to build a kernel stencil and export it."""

    output_path: Path
    closure: ClosureType
    pair_distribution: PDFType
    U: float
    sigma: float
    kernel_n: int
    dz: float
    subdiv: int
    g0: float | None = None
    nn_d: float | None = None
    nn_sigma: float | None = None
    lambda_: float | None = None


@dataclass
class KernelResult:
    """Computed kernel stencil plus an oversampled continuum curve and moments."""

    fine_dz: float
    x_sample: np.ndarray  # exported stencil offsets
    K_sample: np.ndarray  # exported stencil values
    x_dense: np.ndarray  # oversampled curve (for plotting / accurate moments)
    K_dense: np.ndarray
    nu: float
    mu: float


# --------------------------------------------------------------------------- #
# Pair potential / pair distribution functions
# --------------------------------------------------------------------------- #


def lj_potential(r: Array1F, U: np.floating, sigma: np.floating):
    r"""Lennard Jones Potential $u(r) = 4U(\sigma^{12}/r^{12} - \sigma^6/r^6)$."""
    return 4 * U * ((sigma / r) ** 12 - (sigma / r) ** 6)


def lj_derivative(r: Array1F, U: np.floating, sigma: np.floating) -> Array1F:
    r"""Analytical derivative of the Lennard Jones potential."""
    sr6 = (sigma / r) ** 6
    sr12 = sr6**2
    return (4 * U / r) * (-12 * sr12 + 6 * sr6)


def pdf_mean_field(x: Array1F):
    r"""$g(x) = 1$."""
    return np.ones_like(x)


def pdf_nearest_neighbor(
    x: Array1F, g0: np.floating, d: np.floating, sigma: np.floating
):
    r"""$g(x) = g_0 \exp(-(r-d)^2 / (2 \sigma_C^2))$."""
    return g0 * np.exp(-((x - d) ** 2) / (2 * (sigma**2)))


def pdf_exponential(x: Array1F, U: np.floating, sigma: np.floating) -> Array1F:
    r"""$g(x) = \exp(-\sigma\, u(x)/U)$."""
    return np.exp(-sigma * (lj_potential(x, U, sigma) / U))


def guard_pair_distribution(
    fn: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Zero out the pair distribution at (near-)zero separation."""

    def guarded(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.asarray(fn(x), dtype=np.float64)
        out = out.copy()
        out[x < 1e-8] = 0.0
        return out

    return guarded


# --------------------------------------------------------------------------- #
# Kernel closures
# --------------------------------------------------------------------------- #


def compute_force_closure_kernel(
    x,
    u_prime_func,
    g_func,
    sub_res: float = 10000.0,
):
    r"""Computes $K(x) = -x \int_{|x|}^\infty g(R) u'(R)\, dR$."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("`x` must be a 1-D array")
    if sub_res <= 0:
        raise ValueError("`sub_res` must be positive")
    if x.size == 0:
        return np.asarray([], dtype=np.float64)
    if x.size == 1:
        return np.zeros_like(x, dtype=np.float64)

    def _infer_uniform_spacing(sample_x: np.ndarray) -> float:
        spacing = np.diff(sample_x)
        if not np.allclose(spacing, spacing[0]):
            raise ValueError("`x` must be sampled on a uniform grid")
        if spacing[0] <= 0.0:
            raise ValueError("`x` must be strictly increasing")
        return float(spacing[0])

    def _build_force_closure_radial_grid(
        sample_x: np.ndarray, kernel_dz: float, radial_sub_res: float
    ) -> tuple[np.ndarray, float]:
        max_multiple = float(np.max(np.abs(sample_x)) / kernel_dz)
        fine_res = int(radial_sub_res * (max_multiple + 1.0))
        fine_dr = kernel_dz / radial_sub_res
        r = np.arange(fine_res, dtype=np.float64) * fine_dr
        return r, fine_dr

    def _evaluate_force_closure_inputs(
        r: np.ndarray, fine_dr: float
    ) -> tuple[np.ndarray, np.ndarray]:
        r_eval = np.where(r > 0.0, r, fine_dr)
        g_vals = np.asarray(g_func(r_eval), dtype=np.float64)
        u_prime_vals = np.asarray(u_prime_func(r_eval), dtype=np.float64)
        g_vals[0] = 0.0
        u_prime_vals[0] = 0.0
        return g_vals, u_prime_vals

    def _accumulate_force_closure_tail(
        g_vals: np.ndarray, u_prime_vals: np.ndarray, fine_dr: float
    ) -> np.ndarray:
        contributions = fine_dr * u_prime_vals * g_vals
        kernel_fine = np.empty_like(g_vals, dtype=np.float64)
        kernel_fine[0] = 0.0
        # Exclusive left-to-right prefix sum: kernel_fine[k] holds the integral
        # contributions from indices 1..k-1 (contributions[0] is 0 anyway).
        np.cumsum(contributions[:-1], out=kernel_fine[1:])
        return kernel_fine[-1] - kernel_fine

    def _evaluate_force_closure_on_samples(
        sample_x: np.ndarray, tail: np.ndarray, fine_dr: float
    ) -> np.ndarray:
        sample_idx = np.rint(np.abs(sample_x) / fine_dr).astype(np.int64)
        sample_idx = np.clip(sample_idx, 0, tail.size - 1)
        return -(sample_x * tail[sample_idx])

    kernel_dz = _infer_uniform_spacing(x)
    r, fine_dr = _build_force_closure_radial_grid(x, kernel_dz, sub_res)
    g_vals, u_prime_vals = _evaluate_force_closure_inputs(r, fine_dr)
    tail = _accumulate_force_closure_tail(g_vals, u_prime_vals, fine_dr)
    return _evaluate_force_closure_on_samples(x, tail, fine_dr)


def compute_potential_closure_kernel(
    x,
    u_func,
    g_func,
):
    r"""Computes the potential-closure kernel $K(x) = -x\, g(|x|)\, u(|x|)$."""
    x = np.asarray(x, dtype=np.float64)
    abs_x = np.abs(x)
    K = np.zeros_like(x, dtype=np.float64)
    nonzero_mask = abs_x > 0.0
    if np.any(nonzero_mask):
        g_vals = np.asarray(g_func(abs_x[nonzero_mask]), dtype=np.float64)
        u_vals = np.asarray(u_func(abs_x[nonzero_mask]), dtype=np.float64)
        K[nonzero_mask] = -(x[nonzero_mask] * g_vals * u_vals)
    return K


def generate_kernel_stencil(kernel_func, kernel_n: np.integer, kernel_dz: np.floating):
    """Sample a continuous kernel on a discrete odd-length stencil centered at 0."""
    assert kernel_n % 2 != 0, "`kernel_n` needs to be odd!"
    center_idx = (kernel_n - 1) // 2
    x = (np.arange(kernel_n, dtype=np.float64) - center_idx) * kernel_dz
    return x, kernel_func(x)


def calculate_nu_mu(x: Array1F, k: Array1F) -> tuple[float, float]:
    r"""Compute Taylor-expansion moments $\nu$ and $\mu$ from a discrete kernel."""
    x = np.asarray(x, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    if x.ndim != 1 or k.ndim != 1 or x.shape != k.shape:
        raise ValueError("`x` and `K` must be 1-D arrays with the same shape")
    if x.size < 2:
        raise ValueError("Need at least two stencil points to infer the spacing")

    dz = float(x[1] - x[0])
    nu = dz * float(np.sum(x * k))
    mu = (dz * float(np.sum((x**3) * k))) / 6.0

    return nu, mu


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def build_pair_distribution_function(
    pair_distribution: PDFType,
    *,
    U: float,
    sigma: float,
    g0: float | None = None,
    nn_d: float | None = None,
    nn_sigma: float | None = None,
    lambda_: float | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    if pair_distribution == PDFType.MEAN_FIELD:
        return guard_pair_distribution(pdf_mean_field)

    if pair_distribution == PDFType.NEAREST_NEIGHBOR:
        if g0 is None or nn_d is None or nn_sigma is None:
            raise ValueError(
                "Nearest-neighbor pair distribution requires g0, nn_d, and nn_sigma."
            )
        return guard_pair_distribution(
            lambda x: pdf_nearest_neighbor(
                x, np.float64(g0), np.float64(nn_d), np.float64(nn_sigma)
            )
        )

    if lambda_ is None:
        raise ValueError("Exponential pair distribution requires lambda_.")
    if U == 0.0:
        raise ValueError("Exponential pair distribution requires nonzero U.")
    return guard_pair_distribution(
        lambda x: np.exp(-(lambda_ * (lj_potential(x, U, sigma) / U)))
    )


def build_kernel_function(
    closure: ClosureType,
    pair_distribution: PDFType,
    *,
    U: float,
    sigma: float,
    g0: float | None = None,
    nn_d: float | None = None,
    nn_sigma: float | None = None,
    lambda_: float | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    pair_func = build_pair_distribution_function(
        pair_distribution,
        U=U,
        sigma=sigma,
        g0=g0,
        nn_d=nn_d,
        nn_sigma=nn_sigma,
        lambda_=lambda_,
    )

    if closure == ClosureType.FORCE:
        return lambda x: compute_force_closure_kernel(
            x,
            u_prime_func=lambda r: lj_derivative(r, U, sigma),
            g_func=pair_func,
        )

    return lambda x: compute_potential_closure_kernel(
        x,
        u_func=lambda r: lj_potential(r, U, sigma),
        g_func=pair_func,
    )


# Oversampling factor for the continuum curve used in plots / accurate moments.
_DENSE_SCALE = 101


def compute_kernel(cfg: KernelConfig) -> KernelResult:
    """Build the discrete export stencil plus an oversampled continuum curve.

    Mirrors the legacy ``compute_kernel_stencil_data`` + the notebook's dense
    sampling path. ``nu`` / ``mu`` are taken from the oversampled curve.
    """
    if cfg.kernel_n < 3 or cfg.kernel_n % 2 == 0:
        raise ValueError("kernel_n must be an odd integer >= 3.")
    if cfg.subdiv <= 0:
        raise ValueError("subdiv must be positive.")
    if cfg.dz <= 0.0:
        raise ValueError("dz must be positive.")

    fine_dz = cfg.dz / cfg.subdiv
    kernel_func = build_kernel_function(
        cfg.closure,
        cfg.pair_distribution,
        U=cfg.U,
        sigma=cfg.sigma,
        g0=cfg.g0,
        nn_d=cfg.nn_d,
        nn_sigma=cfg.nn_sigma,
        lambda_=cfg.lambda_,
    )
    x_sample, K_sample = generate_kernel_stencil(
        kernel_func=kernel_func,
        kernel_n=cfg.kernel_n,
        kernel_dz=fine_dz,
    )
    x_dense, K_dense = generate_kernel_stencil(
        kernel_func=kernel_func,
        kernel_n=cfg.kernel_n * _DENSE_SCALE,
        kernel_dz=fine_dz / _DENSE_SCALE,
    )
    nu, mu = calculate_nu_mu(x_dense, K_dense)
    return KernelResult(
        fine_dz=fine_dz,
        x_sample=x_sample,
        K_sample=K_sample,
        x_dense=x_dense,
        K_dense=K_dense,
        nu=float(nu),
        mu=float(mu),
    )


def write_kernel_h5(
    output_path: str | Path, result: KernelResult, cfg: KernelConfig
) -> Path:
    """Write the CUDA-compatible convolution stencil to HDF5."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(output_path), "w") as f:
        _ = kernel_group = f.create_group("kernel")
        _ = kernel_group.create_dataset(
            "values", data=np.asarray(result.K_sample, dtype=np.float64)
        )
        _ = kernel_group.create_dataset(
            "x", data=np.asarray(result.x_sample, dtype=np.float64)
        )
        kernel_group.attrs["kernelN"] = int(cfg.kernel_n)
        kernel_group.attrs["spacing"] = float(result.fine_dz)
        kernel_group.attrs["DZ"] = float(cfg.dz)
        kernel_group.attrs["subDiv"] = int(cfg.subdiv)
        kernel_group.attrs["closure"] = cfg.closure.label
        kernel_group.attrs["pair_distribution"] = cfg.pair_distribution.label
        kernel_group.attrs["U"] = float(cfg.U)
        kernel_group.attrs["cuda_compatible"] = 1
        kernel_group.attrs["generated_by"] = "red_patterns/kernel.py"

    return output_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_export_parser(prog: str = "kernel export") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Export a CUDA-compatible convolution kernel to HDF5.",
    )
    _ = parser.add_argument(
        "--output", required=True, help="Path to the output HDF5 file."
    )
    _ = parser.add_argument(
        "--closure",
        required=True,
        choices=sorted(t.value for t in ClosureType),
        help="Kernel closure to use.",
    )
    _ = parser.add_argument(
        "--pair-distribution",
        required=True,
        choices=sorted(t.value for t in PDFType),
        help="Pair distribution model.",
    )
    _ = parser.add_argument(
        "--U", required=True, type=float, help="Interaction energy in Joules."
    )
    _ = parser.add_argument(
        "--sigma", required=True, type=float, help="LJ sigma in meters."
    )
    _ = parser.add_argument(
        "--kernel-n", required=True, type=int, help="Odd stencil size."
    )
    _ = parser.add_argument(
        "--dz", required=True, type=float, help="Coarse grid spacing in meters."
    )
    _ = parser.add_argument(
        "--subdiv", required=True, type=int, help="Subdivisions per coarse cell."
    )
    _ = parser.add_argument("--g0", type=float, help="Nearest-neighbor g0.")
    _ = parser.add_argument(
        "--nn-d", type=float, help="Nearest-neighbor preferred spacing in meters."
    )
    _ = parser.add_argument(
        "--nn-sigma", type=float, help="Nearest-neighbor Gaussian width in meters."
    )
    _ = parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        help="Exponential pair-distribution lambda.",
    )
    return parser


def validate_export_namespace(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> KernelConfig:
    errors: list[str] = []

    if args.kernel_n < 3 or args.kernel_n % 2 == 0:
        errors.append("--kernel-n must be an odd integer >= 3.")
    if args.dz <= 0.0:
        errors.append("--dz must be positive.")
    if args.subdiv <= 0:
        errors.append("--subdiv must be positive.")
    if args.sigma <= 0.0:
        errors.append("--sigma must be positive.")

    if args.pair_distribution == PDFType.NEAREST_NEIGHBOR:
        if args.g0 is None:
            errors.append("--g0 is required with --pair-distribution=nearest-neighbor.")
        if args.nn_d is None:
            errors.append(
                "--nn-d is required with --pair-distribution=nearest-neighbor."
            )
        if args.nn_sigma is None:
            errors.append(
                "--nn-sigma is required with --pair-distribution=nearest-neighbor."
            )
        if args.lambda_ is not None:
            errors.append(
                "--lambda is only valid with --pair-distribution=exponential."
            )
    elif args.pair_distribution == PDFType.EXPONENTIAL:
        if args.lambda_ is None:
            errors.append("--lambda is required with --pair-distribution=exponential.")
        if args.U == 0.0:
            errors.append("--U must be nonzero with --pair-distribution=exponential.")
        if args.g0 is not None or args.nn_d is not None or args.nn_sigma is not None:
            errors.append(
                "--g0, --nn-d, and --nn-sigma are only valid with "
                "--pair-distribution=nearest-neighbor."
            )
    else:
        if args.g0 is not None or args.nn_d is not None or args.nn_sigma is not None:
            errors.append(
                "--g0, --nn-d, and --nn-sigma are only valid with "
                "--pair-distribution=nearest-neighbor."
            )
        if args.lambda_ is not None:
            errors.append(
                "--lambda is only valid with --pair-distribution=exponential."
            )

    if errors:
        parser.error("\n".join(errors))

    return KernelConfig(
        output_path=Path(args.output),
        closure=ClosureType(args.closure),
        pair_distribution=PDFType(args.pair_distribution),
        U=args.U,
        sigma=args.sigma,
        kernel_n=args.kernel_n,
        dz=args.dz,
        subdiv=args.subdiv,
        g0=args.g0,
        nn_d=args.nn_d,
        nn_sigma=args.nn_sigma,
        lambda_=args.lambda_,
    )


def run_export_cli(argv: list[str], prog: str = "kernel export") -> int:
    parser = build_export_parser(prog)
    args = parser.parse_args(argv)
    cfg = validate_export_namespace(parser, args)
    result = compute_kernel(cfg)
    output_path = write_kernel_h5(cfg.output_path, result, cfg)

    print(f"Exported CUDA-compatible convolution kernel to {output_path}")
    print(
        f"Stored {cfg.kernel_n} samples with spacing {result.fine_dz:.6e} m "
        "on /kernel/values"
    )
    print(
        "Used parameters: "
        f"closure={cfg.closure}, "
        f"pair_distribution={cfg.pair_distribution}, "
        f"DZ={cfg.dz:.6e}, "
        f"subdiv={cfg.subdiv}, "
        f"U={cfg.U:.6e} J"
    )
    return 0


# --------------------------------------------------------------------------- #
# UI factory (marimo). Imports are lazy so the compute path stays headless.
# --------------------------------------------------------------------------- #


def make_kernel_ui(
    *,
    U: float = 100.0,
    sigma: float = 5.6,
    closure: str = ClosureType.FORCE.label,
    pair_distribution: str = PDFType.NEAREST_NEIGHBOR.label,
    g0: float = 4.0,
    nn_d: float = 6.585467,
    nn_sigma: float = 0.5,
    lambda_: float = 1.0,
    kernel_n: int = 31,
    dz: float = 256.0 * 1.0455122765372783e-6,
    subdiv: int = 256,
):
    """Build every kernel control as a single ``mo.ui.dictionary``.

    Returning one registered UI element is what makes cross-cell ``.value`` reads
    reliable in marimo: display it (or :func:`kernel_ui_layout`) in one cell and
    read ``kernel_ui.value`` from any other cell with normal reactivity. Units in
    the UI are display units; :func:`kernel_config_from_ui` applies the scalings.
    """
    import marimo as mo

    return mo.ui.dictionary(
        {
            "U": mo.ui.number(
                start=0.0, stop=1000, step=0.05, value=U, label="$U \\; [10^{-18} J]$"
            ),
            "sigma": mo.ui.number(
                start=0.0,
                stop=10,
                step=0.05,
                value=sigma,
                label="$\\sigma \\; [10^{-6} m]$",
            ),
            "closure": mo.ui.tabs(
                {
                    ClosureType.POTENTIAL.label: mo.md(
                        r"$$K(x) = -x\,g(|x|)\,u(|x|)$$"
                    ),
                    ClosureType.FORCE.label: mo.md(
                        r"$$K(x) = x\int_{|x|}^{\infty} g(R)\,f(R)\, d R,"
                        r"\qquad f(R) = -u'(R)$$"
                    ),
                },
                value=closure,
            ),
            "pdf_type": mo.ui.tabs(
                {
                    PDFType.MEAN_FIELD.label: mo.md(r"$g(x) = 1$"),
                    PDFType.NEAREST_NEIGHBOR.label: mo.md(
                        r"$g(x) = g_0 \exp(-(r-d)^2 / (2\sigma_C^2))$"
                    ),
                    PDFType.EXPONENTIAL.label: mo.md(
                        r"$g(x) = \exp(-\lambda\, u(x)/U)$"
                    ),
                },
                value=pair_distribution,
            ),
            "g0": mo.ui.number(
                start=0.0, stop=10.0, step=0.1, value=g0, label="$g_0 \\; [10^{7}]$"
            ),
            "nn_d": mo.ui.number(
                start=0.0,
                stop=10.0,
                step=0.00000001,
                value=nn_d,
                label="$d \\; [10^{-6} m]$",
            ),
            "nn_sigma": mo.ui.number(
                start=0.0,
                stop=1.0,
                step=0.01,
                value=nn_sigma,
                label="$\\sigma_C \\; [10^{-6} m]$",
            ),
            "lambda": mo.ui.number(
                start=0.0, stop=10.0, step=0.5, value=lambda_, label="$\\lambda$"
            ),
            "kernel_n": mo.ui.number(
                start=3, stop=10001, step=2, value=kernel_n, label="kernelN"
            ),
            "dz": mo.ui.number(
                start=1e-12, step=1e-8, value=dz, label="(coarse) $\\Delta z$"
            ),
            "subdiv": mo.ui.number(start=1, step=1, value=subdiv, label="subDiv"),
        }
    )


def kernel_ui_layout(kernel_ui):
    """Compose the dictionary's child widgets into a readable control panel.

    The children stay bound to ``kernel_ui`` so ``kernel_ui.value`` keeps
    aggregating their values regardless of how they are laid out here.
    """
    import marimo as mo

    pdf_params = {
        PDFType.MEAN_FIELD.label: mo.md("Mean field has no free parameters."),
        PDFType.NEAREST_NEIGHBOR.label: mo.vstack(
            [kernel_ui["g0"], kernel_ui["nn_d"], kernel_ui["nn_sigma"]]
        ),
        PDFType.EXPONENTIAL.label: kernel_ui["lambda"],
    }
    active_pdf = kernel_ui["pdf_type"].value
    return mo.vstack(
        [
            mo.md("### Pair potential"),
            kernel_ui["U"],
            kernel_ui["sigma"],
            mo.md("### Closure"),
            kernel_ui["closure"],
            mo.md("### Pair distribution"),
            kernel_ui["pdf_type"],
            pdf_params.get(active_pdf, mo.md("")),
            mo.md("### Stencil grid"),
            kernel_ui["kernel_n"],
            kernel_ui["dz"],
            kernel_ui["subdiv"],
        ],
        gap=0.5,
    )


_PDF_LABEL_MAP = {t.label: t for t in PDFType}
_CLOSURE_LABEL_MAP = {t.label: t for t in ClosureType}


def kernel_config_from_ui(
    value: dict, output_path: str | Path = "kernel.h5"
) -> KernelConfig:
    """Map a ``make_kernel_ui()`` value dict to a :class:`KernelConfig`.

    Applies the display-unit -> SI scalings. All pair-distribution parameters are
    passed through; :func:`build_pair_distribution_function` ignores the inactive
    ones.
    """
    return KernelConfig(
        output_path=Path(output_path),
        closure=_CLOSURE_LABEL_MAP[str(value["closure"])],
        pair_distribution=_PDF_LABEL_MAP[str(value["pdf_type"])],
        U=float(value["U"]) * 1e-18,
        sigma=float(value["sigma"]) * 1e-6,
        kernel_n=int(value["kernel_n"]),
        dz=float(value["dz"]),
        subdiv=int(value["subdiv"]),
        g0=float(value["g0"]) * 1e7,
        nn_d=float(value["nn_d"]) * 1e-6,
        nn_sigma=float(value["nn_sigma"]) * 1e-6,
        lambda_=float(value["lambda"]),
    )


def plot_kernel(result: KernelResult, cfg: KernelConfig | None = None):
    """Plot the dense kernel curve with the exported stencil overlaid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        result.x_dense * 1e6,
        result.K_dense,
        color="blue",
        linewidth=2,
        label="dense sampled curve",
    )
    ax.scatter(
        result.x_sample * 1e6,
        result.K_sample,
        s=18,
        color="black",
        label="exported stencil",
        zorder=3,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel(r"Offset $x$ ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"Kernel $K(x)$", fontsize=12)
    if cfg is not None:
        ax.set_title(
            f"{cfg.closure.label} ({cfg.pair_distribution.label})",
            fontsize=14,
        )
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()
    return fig


def plot_pair_distribution(cfg: KernelConfig):
    """Plot the pair distribution function $g(r)$ for a config."""
    import matplotlib.pyplot as plt

    g_func = build_pair_distribution_function(
        cfg.pair_distribution,
        U=cfg.U,
        sigma=cfg.sigma,
        g0=cfg.g0,
        nn_d=cfg.nn_d,
        nn_sigma=cfg.nn_sigma,
        lambda_=cfg.lambda_,
    )
    r_plot = np.linspace(1e-9, 5e-5, 400)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r_plot * 1e6, g_func(r_plot), color="blue", linewidth=2)
    ax.set_xlabel(r"Distance $r$ ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"Pair distribution $g(r)$", fontsize=12)
    ax.set_title("Pair distribution function", fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.7)
    return fig
