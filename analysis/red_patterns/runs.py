from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias
import numpy as np
from numpy.typing import NDArray
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

Array1F = NDArray[np.float64]
Array2F = NDArray[np.float64]
Array3F = NDArray[np.float64]

AttrScalar: TypeAlias = str | int | float | bool


# Parameters from the MATLAB colorModel.chBfit
_CH_B_FIT = np.array(
    [
        [145.7586, 134.7227, 130.9048],
        [-130.4334, -124.1164, -101.2598],
        [0.4656, -0.4106, -0.4582],
        [0.7014, 0.3524, 0.4840],
        [2.4949, 2.2571, 5.9188],
    ],
    dtype=np.float64,
)


def _color_model(b: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Python equivalent of MATLAB color model fit.

    Evaluates the rational/power function using complex arrays to avoid NaNs for
    negative bases, returning the real component.
    """

    x_c = x.astype(complex)
    term1 = (x_c - b[2]) / b[3]
    denom = (1 + term1 ** b[4]) ** (1 / b[4])
    result = b[0] + b[1] * term1 / denom
    return np.real(result)


def _get_custom_colormap(psi_min: float, psi_max: float) -> mcolors.ListedColormap:
    """Generate the MATLAB-fit colormap used for psi plots."""

    psi_vals = np.linspace(float(psi_min), float(psi_max), 2**16, dtype=np.float64)

    # Apply log10 transformation safely. Match the MATLAB logic: when psi==0,
    # treat it as 0.001 (in psi/2.22 units).
    with np.errstate(divide="ignore", invalid="ignore"):
        log10_psi = np.log10(psi_vals / 2.22)
    log10_psi[np.isinf(log10_psi)] = np.log10(0.001)
    log10_psi[np.isnan(log10_psi)] = np.log10(0.001)

    R = _color_model(_CH_B_FIT[:, 0], log10_psi)
    G = _color_model(_CH_B_FIT[:, 1], log10_psi)
    B = _color_model(_CH_B_FIT[:, 2], log10_psi)

    rgb = np.column_stack((R, G, B))
    rgb = np.clip(rgb, 0, 255) / 255.0

    return mcolors.ListedColormap(rgb)


_RBC_CMAP = _get_custom_colormap(0.01, 100.0)


def get_rbc_cmap() -> mcolors.ListedColormap:
    """Return the fixed RBC colormap calibrated on the physical 0.01%-100% range."""

    return _RBC_CMAP


def _decode(v: object) -> AttrScalar:
    """Convert an HDF5 attribute value into a plain Python scalar.

    We only expect scalar attributes in `run.h5` (numbers/strings/bools).
    """
    if isinstance(v, bytes):
        # Fixed-length string attributes often come back as raw bytes (possibly NUL-padded).
        return v.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(v, np.bytes_):
        # Same as above, but stored in a numpy scalar container.
        return v.tobytes().decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(v, np.generic):
        # Convert numpy scalar (e.g. np.int32/np.float64) to a plain Python scalar.
        item = v.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace").rstrip("\x00")
        if isinstance(item, (str, int, float, bool)):
            return item
        # Best-effort: keep return type scalar-only.
        return str(item)
    if isinstance(v, (str, int, float, bool)):
        return v
    # Best-effort: keep return type scalar-only.
    return str(v)


def _attrs_to_dict(
    obj: h5py.File | h5py.Group | h5py.Dataset | h5py.Datatype,
) -> dict[str, AttrScalar]:
    """Return decoded HDF5 attributes as a plain Python dict."""

    out: dict[str, AttrScalar] = {}
    for k, v in obj.attrs.items():
        out[str(k)] = _decode(v)
    return out


def _get_dataset(group: h5py.Group, key: str) -> h5py.Dataset:
    """Guarantees that ``<group>/<key>`` is a Dataset. Otherwise raises type error."""
    obj = group[key]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(
            f"Expected HDF5 dataset at {key!r}, got {type(obj).__name__} instead"
        )
    return obj


# The dataclasses below intentionally mirror the C configuration layout in
# `include/sim_types.h` (SimConfig -> RunParams + ModelParams).
# In particular, ModelParams is a tagged union: `modelType` selects which
# variant payload is meaningful (Conv vs Tayl).


@dataclass
class RunParamsData:
    """Python mirror of `RunParams` from `include/sim_types.h`."""

    N: int
    NT: int
    T: float
    DT: float
    DZ: float
    fineDZ: float
    sysL: float
    NO: int


@dataclass
class ConvVariant:
    kernelN: int
    subDiv: int
    M: int


@dataclass
class TaylVariant:
    NU: float
    MU: float


@dataclass
class ModelParamsData:
    """Python mirror of `ModelParams` from `include/sim_types.h`.

    `modelType` acts as the tag selecting the active `variant` payload.
    """

    modelType: Literal["CONV", "TAYL"]
    gradientType: Literal["LINEAR", "SIGMOID"]
    U: float
    PSI: float
    alpha: float
    beta: float
    variant: ConvVariant | TaylVariant  # Tagged-union payload selected by `modelType`.


@dataclass
class RunConfig:
    """Simulation configuration as stored in `run.h5`."""

    run: RunParamsData
    model: ModelParamsData
    git_commit: str | None
    git_describe: str | None
    created: str | None


@dataclass
class RunData:
    path: Path
    config: RunConfig
    time: Array1F
    rho: Array1F
    z: Array1F
    phi: Array3F | None
    psi: Array2F | None

    @property
    def n_saved(self) -> int:
        """Returns the total number of time steps."""
        return int(self.time.shape[0])  # pyright: ignore[reportAny]

    @property
    def final_time(self) -> float | None:
        return None if self.n_saved == 0 else float(self.time[-1])  # pyright: ignore[reportAny]

    def load_phi(self) -> Array3F:
        if self.phi is None:
            with h5py.File(os.fspath(self.path), "r") as f:
                ds = _get_dataset(f, "fields/phi")
                self.phi = np.asarray(ds[...], dtype=np.float32)
        assert self.phi is not None
        return self.phi

    def load_psi(self) -> Array2F:
        if self.psi is None:
            with h5py.File(os.fspath(self.path), "r") as f:
                ds = _get_dataset(f, "fields/psi")
                self.psi = np.asarray(ds[...], dtype=np.float32)
        assert self.psi is not None
        return self.psi

    def phi_frame(self, i: int) -> Array2F:
        if self.phi is not None:
            return self.phi[i]  # pyright: ignore[reportAny]
        with h5py.File(os.fspath(self.path), "r") as f:
            ds = _get_dataset(f, "fields/phi")
            return np.asarray(ds[i], dtype=np.float32)

    def psi_frame(self, i: int) -> Array1F:
        if self.psi is not None:
            return self.psi[i]  # pyright: ignore[reportAny]
        with h5py.File(os.fspath(self.path), "r") as f:
            ds = _get_dataset(f, "fields/psi")
            return np.asarray(ds[i], dtype=np.float32)

    @classmethod
    def from_h5(cls, path: str | Path, load_fields: bool = False) -> "RunData":
        with h5py.File(os.fspath(path), "r") as f:
            root = _attrs_to_dict(f)
            run_attrs = _attrs_to_dict(f["config/run"])
            model_attrs = _attrs_to_dict(f["config/model"])

            if "config/model/variant/conv" in f:
                vattrs = _attrs_to_dict(f["config/model/variant/conv"])
                variant = ConvVariant(
                    kernelN=int(vattrs["kernelN"]),
                    subDiv=int(vattrs["subDiv"]),
                    M=int(vattrs["M"]),
                )
                model_type = "CONV"
            elif "config/model/variant/tayl" in f:
                vattrs = _attrs_to_dict(f["config/model/variant/tayl"])
                variant = TaylVariant(
                    NU=float(vattrs["NU"]),
                    MU=float(vattrs["MU"]),
                )
                model_type = "TAYL"
            else:
                raise ValueError(f"No variant group found in {path}")

            run = RunParamsData(
                N=int(run_attrs["N"]),
                NT=int(run_attrs["NT"]),
                T=float(run_attrs["T"]),
                DT=float(run_attrs["DT"]),
                DZ=float(run_attrs["DZ"]),
                fineDZ=float(run_attrs["fineDZ"]),
                sysL=float(run_attrs["sysL"]),
                NO=int(run_attrs["NO"]),
            )

            model = ModelParamsData(
                modelType=model_type,
                gradientType=(
                    "LINEAR"
                    if model_attrs["gradientType"] == "LINEAR"
                    else "SIGMOID"
                    if model_attrs["gradientType"] == "SIGMOID"
                    else (_ for _ in ()).throw(
                        ValueError(
                            f"Unknown gradientType attr: {model_attrs['gradientType']!r}"
                        )
                    )
                ),
                U=float(model_attrs["U"]),
                PSI=float(model_attrs["PSI"]),
                alpha=float(model_attrs["alpha"]),
                beta=float(model_attrs["beta"]),
                variant=variant,
            )

            config = RunConfig(
                run=run,
                model=model,
                git_commit=str(root["git_commit"]),
                git_describe=str(root["git_describe"]),
                created=str(root["created"]),
            )

            time = np.asarray(_get_dataset(f, "time")[...], dtype=np.float32)
            rho = np.asarray(_get_dataset(f, "coords/rho")[...], dtype=np.float32)
            z = np.asarray(_get_dataset(f, "coords/z")[...], dtype=np.float32)

            phi = (
                np.asarray(_get_dataset(f, "fields/phi")[...], dtype=np.float32)
                if load_fields
                else None
            )
            psi = (
                np.asarray(_get_dataset(f, "fields/psi")[...], dtype=np.float32)
                if load_fields
                else None
            )

        return cls(
            path=Path(path),
            config=config,
            time=time,
            rho=rho,
            z=z,
            phi=phi,
            psi=psi,
        )


def plot_psi(
    run: RunData,
    *,
    vmin: float = 0.0,
    vmax: float = 100.0,
    max_t_pixels: int = 2400,
    max_z_pixels: int = 1200,
    interpolation: str = "nearest",
    cmap: str | mcolors.Colormap | None = None,
    title: str | None = None,
) -> plt.Figure:
    r"""Plot :math:`\psi(t, z)` for a simulation run.

    The plot is a 2D heatmap with time on the x-axis and the z-coordinate on the
    y-axis. For large runs, the function down-samples the data to keep the image
    size manageable.

    Notes:
    - If ``run.psi`` is not loaded, this function reads ``/fields/psi`` lazily
      from ``run.path``.
    - Down-sampling is performed by simple striding (no filtering).

    Args:
        run: Loaded run metadata and (optionally) fields.
        vmin: Lower color limit.
        vmax: Upper color limit.
        max_t_pixels: Maximum number of samples plotted along time.
        max_z_pixels: Maximum number of samples plotted along z.
        interpolation: Matplotlib interpolation mode passed to ``imshow``.
        cmap: Optional Matplotlib colormap name.
        title: Optional plot title. Defaults to the run directory name.

    Returns:
        The created Matplotlib figure.
    """

    # Ensure 1D float32 axes.
    t = np.asarray(run.time, dtype=np.float32)
    z = np.asarray(run.z, dtype=np.float32)

    if t.ndim != 1 or z.ndim != 1 or t.shape[0] == 0 or z.shape[0] == 0:
        raise ValueError("Invalid time or z axes.")

    nt, nz = t.shape[0], z.shape[0]

    if max_t_pixels <= 0 or max_z_pixels <= 0:
        raise ValueError("max_t_pixels and max_z_pixels must be positive.")

    # Pick stride factors so the plotted resolution stays within the requested
    # bounds. This keeps plotting responsive for large runs.
    t_step = max(1, int(np.ceil(nt / max_t_pixels)))
    z_step = max(1, int(np.ceil(nz / max_z_pixels)))

    t_idx = slice(None, None, t_step)
    z_idx = slice(None, None, z_step)
    t_plot, z_plot = t[t_idx], z[z_idx]

    if run.psi is not None:
        # Prefer the in-memory field if already loaded.
        psi = np.asarray(run.psi[t_idx, z_idx], dtype=np.float32)
    else:
        with h5py.File(os.fspath(run.path), "r") as f:
            psi_ds = _get_dataset(f, "fields/psi")
            psi = np.asarray(psi_ds[t_idx, z_idx], dtype=np.float32)

    if psi.ndim != 2:
        raise ValueError(f"Expected psi to be 2D (time x z), got shape {psi.shape}.")
    if psi.shape[0] != t_plot.shape[0] or psi.shape[1] != z_plot.shape[0]:
        raise ValueError(
            "psi shape does not match axes after down-sampling: "
            f"psi={psi.shape}, t={t_plot.shape}, z={z_plot.shape}"
        )

    plot_title = run.path.parent.name if title is None else title
    return plot_psi_arrays(
        psi,
        t_plot,
        z_plot,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
        cmap=cmap,
        title=plot_title,
    )


def plot_psi_arrays(
    psi: NDArray[np.floating],
    t: NDArray[np.floating],
    z: NDArray[np.floating],
    *,
    vmin: float = 0.0,
    vmax: float = 100.0,
    interpolation: str = "nearest",
    cmap: str | mcolors.Colormap | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    constrained_layout: bool = True,
    origin: Literal["upper", "lower"] | None = "lower",
    aspect: float | Literal["equal", "auto"] | None = "auto",
    add_colorbar: bool = True,
    cbar_label: str = r"$\psi(t,z) \; [\%]$",
    **imshow_kwargs: object,
):
    r"""Plot $\psi(t, z)$ from arrays.

    Args:
        psi: 2D array with shape (time, z).
        t: 1D time axis (same length as psi.shape[0]).
        z: 1D z axis (same length as psi.shape[1]).
        vmin: Lower color limit.
        vmax: Upper color limit.
        interpolation: Matplotlib interpolation mode passed to ``imshow``.
        cmap: Optional Matplotlib colormap name/object.
        title: Optional plot title.
        ax: Optional Matplotlib axes to draw into. If not provided, a new figure
            and axes are created.
        constrained_layout: If creating a new figure, whether to enable
            constrained layout.
        origin: ``imshow`` origin.
        aspect: ``imshow`` aspect.
        add_colorbar: Whether to add a colorbar.
        cbar_label: Colorbar label.
        imshow_kwargs: Additional keyword args forwarded to ``imshow``.

    Returns:
        The created Matplotlib figure.
    """

    t = np.asarray(t, dtype=np.float32)
    z = 100 * np.asarray(z, dtype=np.float32)  # Convert z from m to cm
    psi = 100 * np.asarray(psi, dtype=np.float32)  # Convert psi to %

    if t.ndim != 1 or z.ndim != 1 or t.shape[0] == 0 or z.shape[0] == 0:
        raise ValueError("Invalid time or z axes.")
    if psi.ndim != 2:
        raise ValueError(f"Expected psi to be 2D (time x z), got shape {psi.shape}.")
    if psi.shape[0] != t.shape[0] or psi.shape[1] != z.shape[0]:
        raise ValueError(
            f"psi shape does not match axes: psi={psi.shape}, t={t.shape}, z={z.shape}"
        )

    # imshow expects array shape (ny, nx). Our psi is (time, z), so transpose it.
    C = psi.T

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=constrained_layout)
    else:
        fig = ax.figure

    # Map array indices to physical axes. This assumes t and z are monotonic.
    extent = (float(t[0]), float(t[-1]), float(z[0]), float(z[-1]))

    im = ax.imshow(
        C,
        origin=origin,
        aspect=aspect,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=extent,
        **imshow_kwargs,
    )

    ax.set_xlabel(r"$t \; [s]$")
    ax.set_ylabel(r"$z \; [cm]$")
    if title is not None:
        ax.set_title(title)

    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

    return fig


def find_peaks(
    z: Array1F,
    psi: Array1F,
) -> tuple[Array1F, Array1F, float, float]:
    r"""Find the peaks of $\psi(z)$.

    Args:
        z: 1D array with shape (z,).
        psi: 1D array with shape (z,).

    Returns:
        Tuple of (z-coordinates of peaks, psi at each peak, average spacing,
        standard deviation of the spacing). Edge peaks are trimmed: the first
        peak is dropped only when its z < 1, the last only when its z > 6.
    """
    from scipy.signal import find_peaks as _scipy_find_peaks

    peak_indices, _ = _scipy_find_peaks(psi, prominence=0.1)

    peak_indices = peak_indices[
        1 if z[peak_indices[0]] < 1 else 0 : -1 if z[peak_indices[-1]] > 6 else None
    ]

    peak_z = z[peak_indices]
    peak_psi = psi[peak_indices]

    distances = np.diff(peak_z)
    average_spacing = np.mean(distances)
    standard_deviation = np.std(distances)

    return peak_z, peak_psi, average_spacing, standard_deviation


def cli_args_from_run_h5(
    run_h5: str | Path,
    *,
    include_prog: bool = False,
    out_dir: str | Path | None = None,
) -> list[str]:
    """Recreate (best-effort) CLI arguments for ``red-patterns`` from ``run.h5``.

    This reads ``/config`` attributes written by the simulator. Some CLI options
    are not persisted in ``run.h5`` (notably ``--out-dir`` and any legacy args),
    so this function can only reconstruct what is stored.

    Args:
        run_h5: Path to a ``run.h5`` file.
        include_prog: If True, include the program name ``red-patterns`` as the
            first element.
        out_dir: Optional value for ``--out-dir`` to include in the output.

    Returns:
        List of CLI arguments, suitable for ``subprocess`` (no shell quoting).
    """

    def f64(x: AttrScalar) -> str:
        return format(float(x), ".17g")

    args: list[str] = ["red-patterns"] if include_prog else []

    with h5py.File(os.fspath(run_h5), "r") as f:
        run_attrs = _attrs_to_dict(f["config/run"])
        model_attrs = _attrs_to_dict(f["config/model"])

        model_type = str(model_attrs.get("modelType"))
        if model_type == "CONV":
            args.append("--use-convolution")
        elif model_type == "TAYL":
            args.append("--use-taylor")
        else:
            raise ValueError(f"Unknown modelType in {run_h5}: {model_type!r}")

        args.extend(
            [
                f"--T={f64(run_attrs['T'])}",
                f"--DT={f64(run_attrs['DT'])}",
                f"--NO={int(run_attrs['NO'])}",
                f"--U={f64(model_attrs['U'])}",
                f"--PSI={f64(model_attrs['PSI'])}",
            ]
        )

        grad = str(model_attrs.get("gradientType"))
        if grad == "LINEAR":
            args.append("--gradient=linear")
        elif grad == "SIGMOID":
            args.append("--gradient=sigmoid")
        else:
            raise ValueError(f"Unknown gradientType in {run_h5}: {grad!r}")

        # Optional: source file references if present.
        phi_source = str(model_attrs.get("phiSource", "internal"))
        if phi_source and phi_source != "internal":
            args.append(f"--phi-file={phi_source}")

        if model_type == "CONV":
            vattrs = _attrs_to_dict(f["config/model/variant/conv"])
            kernel_source = str(vattrs.get("kernelSource", "internal"))
            if kernel_source and kernel_source != "internal":
                args.append(f"--kernel-file={kernel_source}")
        else:
            vattrs = _attrs_to_dict(f["config/model/variant/tayl"])
            args.append(f"--NU={f64(vattrs['NU'])}")
            args.append(f"--MU={f64(vattrs['MU'])}")

    if out_dir is not None:
        args.append(f"--out-dir={os.fspath(out_dir)}")

    return args
