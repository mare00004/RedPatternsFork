from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias
import numpy as np
from numpy.typing import NDArray
import h5py
import matplotlib.pyplot as plt
import os

Array1F = NDArray[np.float32]
Array2F = NDArray[np.float32]
Array3F = NDArray[np.float32]

AttrScalar: TypeAlias = str | int | float | bool


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
    vmax: float = 1.0,
    max_t_pixels: int = 2400,
    max_z_pixels: int = 1200,
    interpolation: str = "nearest",
    cmap: str | None = None,
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

    # imshow expects array shape (ny, nx). Our psi is (time, z), so transpose it.
    C = psi.T

    fig, ax = plt.subplots(constrained_layout=True)

    # Map array indices to physical axes. This assumes t and z are monotonic.
    extent = (float(t_plot[0]), float(t_plot[-1]), float(z_plot[0]), float(z_plot[-1]))

    im = ax.imshow(
        C,
        origin="lower",
        aspect="auto",
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=extent,
    )

    ax.set_xlabel("t")
    ax.set_ylabel("z")
    ax.set_title(run.path.parent.name if title is None else title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\psi(t,z)$")

    return fig
