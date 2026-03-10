from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union
import numpy as np
from numpy.typing import NDArray
import h5py
import matplotlib.pyplot as plt

Array1F = NDArray[np.float32]
Array2F = NDArray[np.float32]
Array3F = NDArray[np.float32]


def _decode(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(v, np.bytes_):
        return v.tobytes().decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(v, np.generic):
        return v.item()
    return v


def _attrs_to_dict(obj) -> dict:
    return {k: _decode(v) for k, v in obj.attrs.items()}


@dataclass
class RunParamsData:
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
    modelType: Literal["CONV", "TAYL"]
    gradient: str
    U: float
    PSI: float
    alpha: float
    beta: float
    gamma: float
    delta: float
    kappa: float
    variant: Union[ConvVariant, TaylVariant]


@dataclass
class RunConfig:
    run: RunParamsData
    model: ModelParamsData
    schema_version: Optional[int] = None
    git_commit: Optional[str] = None
    git_state: Optional[str] = None
    created_utc: Optional[str] = None


@dataclass
class RunData:
    path: Path
    config: RunConfig
    time: Array1F
    rho: Array1F
    z: Array1F
    phi: Optional[Array3F] = None
    psi: Optional[Array2F] = None

    @property
    def n_saved(self) -> int:
        return int(self.time.shape[0])

    @property
    def final_time(self) -> Optional[float]:
        return None if self.n_saved == 0 else float(self.time[-1])

    def load_phi(self) -> Array3F:
        if self.phi is None:
            with h5py.File(self.path, "r") as f:
                self.phi = f["fields/phi"][:]
        return self.phi

    def load_psi(self) -> Array2F:
        if self.psi is None:
            with h5py.File(self.path, "r") as f:
                self.psi = f["fields/psi"][:]
        return self.psi

    def phi_frame(self, i: int) -> Array2F:
        if self.phi is not None:
            return self.phi[i]
        with h5py.File(self.path, "r") as f:
            return f["fields/phi"][i]

    def psi_frame(self, i: int) -> Array1F:
        if self.psi is not None:
            return self.psi[i]
        with h5py.File(self.path, "r") as f:
            return f["fields/psi"][i]

    @classmethod
    def from_h5(cls, path: Union[str, Path], load_fields: bool = False) -> "RunData":
        path = Path(path)

        with h5py.File(path, "r") as f:
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
                gradient=str(model_attrs["gradient"]),
                U=float(model_attrs["U"]),
                PSI=float(model_attrs["PSI"]),
                alpha=float(model_attrs["alpha"]),
                beta=float(model_attrs["beta"]),
                gamma=float(model_attrs["gamma"]),
                delta=float(model_attrs["delta"]),
                kappa=float(model_attrs["kappa"]),
                variant=variant,
            )

            config = RunConfig(
                run=run,
                model=model,
                schema_version=root.get("schema_version"),
                git_commit=root.get("git_commit"),
                git_state=root.get("git_state"),
                created_utc=root.get("created_utc"),
            )

            time = f["time"][:]
            rho = f["coords/rho"][:]
            z = f["coords/z"][:]

            phi = f["fields/phi"][:] if load_fields else None
            psi = f["fields/psi"][:] if load_fields else None

        return cls(
            path=path,
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
) -> plt.Figure:
    """Plot psi(t, z) from a RunData object."""
    t = np.asarray(run.time, dtype=np.float32)
    z = np.asarray(run.z, dtype=np.float32)

    if t.ndim != 1 or z.ndim != 1 or t.shape[0] == 0 or z.shape[0] == 0:
        raise ValueError("Invalid time or z axes.")

    nt, nz = t.shape[0], z.shape[0]
    t_step = max(1, int(np.ceil(nt / max_t_pixels)))
    z_step = max(1, int(np.ceil(nz / max_z_pixels)))

    t_idx = slice(None, None, t_step)
    z_idx = slice(None, None, z_step)
    t_plot, z_plot = t[t_idx], z[z_idx]

    if run.psi is not None:
        psi = np.asarray(run.psi[t_idx, z_idx], dtype=np.float32)
    else:
        with h5py.File(run.path, "r") as f:
            psi_ds = f["fields/psi"]
            psi = np.asarray(psi_ds[t_idx, z_idx], dtype=np.float32)

    C = psi.T

    fig, ax = plt.subplots(constrained_layout=True)
    extent = (float(t_plot[0]), float(t_plot[-1]), float(z_plot[0]), float(z_plot[-1]))

    im = ax.imshow(
        C,
        origin="lower",
        aspect="auto",
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )

    ax.set_xlabel("t")
    ax.set_ylabel("z")
    ax.set_title(run.path.parent.name)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\psi(t,z)$")

    return fig
