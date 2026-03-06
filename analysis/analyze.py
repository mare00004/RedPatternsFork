# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "h5py==3.15.1",
#     "marimo>=0.19.4",
#     "matplotlib==3.10.8",
#     "numpy==2.4.1",
#     "pandas==3.0.0",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    from __future__ import annotations

    from dataclasses import dataclass, asdict
    from pathlib import Path
    import re
    import shlex
    from typing import Optional, Union
    import numpy as np
    import pandas as pd
    import h5py
    import matplotlib.pyplot as plt
    import marimo as mo
    return (
        Optional,
        Path,
        Union,
        asdict,
        dataclass,
        h5py,
        mo,
        np,
        pd,
        plt,
        re,
        shlex,
    )


@app.cell
def _(Optional, Path, Union, asdict, dataclass, re, shlex):
    def _parse_scalar(s: str) -> Union[int, float, str]:
        s = s.strip()
        # int?
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        # float? (incl. scientific)
        try:
            return float(s)
        except ValueError:
            return s


    def parse_red_patterns_cmdline(line: str) -> dict:
        """
        Turns:
          /bin/red-patterns --use-taylor --T=1600 --DT=5.0e-04 ...
        into:
          {"model": "taylor", "T": 1600.0, "DT": 5e-4, ...}
        """
        tokens = shlex.split(line.strip())
        if not tokens:
            return {}

        # first token is the binary path
        args: dict = {"binary": tokens[0]}

        flags = set()
        for tok in tokens[1:]:
            if not tok.startswith("--"):
                continue
            tok = tok[2:]
            if "=" in tok:
                k, v = tok.split("=", 1)
                args[k] = _parse_scalar(v)
            else:
                flags.add(tok)

        if "use-taylor" in flags:
            args["model"] = "taylor"
        elif "use-convolution" in flags:
            args["model"] = "convolution"
        else:
            args["model"] = "unknown"

        # keep flags too (optional)
        args["flags"] = sorted(flags)
        return args


    @dataclass(frozen=True)
    class Simulation:
        run_dir: Path
        run_h5: Path
        cmdline: str
        model: str

        # common params
        T: float
        DT: float
        NO: int
        U: float
        PSI: float
        gamma: float
        delta: float
        kappa: float
        gradient: str

        # helpful ids from folder name ClusterID.JobID
        cluster_id: Optional[int] = None
        job_id: Optional[int] = None

        def to_row(self) -> dict:
            d = asdict(self)
            d["run_dir"] = str(self.run_dir)
            d["run_h5"] = str(self.run_h5)
            return d


    @dataclass(frozen=True)
    class TaylorSimulation(Simulation):
        NU: float = 0.0
        MU: float = 0.0


    @dataclass(frozen=True)
    class ConvolutionSimulation(Simulation):
        pass


    def simulation_from_cmd(
        run_dir: Path, run_h5: Path, cmdline: str
    ) -> Simulation:
        args = parse_red_patterns_cmdline(cmdline)
        model = args.get("model", "unknown")

        # parse ClusterID.JobID from folder name if possible
        cluster_id = job_id = None
        if "." in run_dir.name:
            a, b = run_dir.name.split(".", 1)
            try:
                cluster_id = int(a)
                job_id = int(b)
            except ValueError:
                pass

        common = dict(
            run_dir=run_dir,
            run_h5=run_h5,
            cmdline=cmdline,
            model=model,
            T=float(args["T"]),
            DT=float(args["DT"]),
            NO=int(args["NO"]),
            U=float(args["U"]),
            PSI=float(args["PSI"]),
            gamma=float(args["gamma"]),
            delta=float(args["delta"]),
            kappa=float(args["kappa"]),
            gradient=str(args["gradient"]),
            cluster_id=cluster_id,
            job_id=job_id,
        )

        if model == "taylor":
            return TaylorSimulation(
                **common,
                NU=float(args["NU"]),
                MU=float(args["MU"]),
            )
        if model == "convolution":
            return ConvolutionSimulation(**common)

        # fallback: still return something useful (you can also raise)
        return Simulation(**common)
    return Simulation, simulation_from_cmd


@app.cell
def _(Path, Simulation, pd, simulation_from_cmd):
    def scan_runs(base_dir: Path) -> tuple[list[Simulation], pd.DataFrame]:
        sims: list[Simulation] = []

        for d in sorted(base_dir.iterdir()):
            if not d.is_dir():
                continue
            if "." not in d.name:  # only ClusterID.JobID by your description
                continue

            h5 = d / "run.h5"
            cmdfile = d / "run_manifest.txt"
            if (not h5.exists()) or (not cmdfile.exists()):
                continue

            cmdline = (
                cmdfile.read_text(encoding="utf-8", errors="replace")
                .splitlines()[0]
                .strip()
            )
            sims.append(simulation_from_cmd(d, h5, cmdline))

        df = pd.DataFrame([s.to_row() for s in sims])
        if len(df):
            # nice ordering
            sort_cols = [
                c
                for c in ["cluster_id", "job_id", "model", "gradient"]
                if c in df.columns
            ]
            if sort_cols:
                df = df.sort_values(sort_cols, kind="stable").reset_index(
                    drop=True
                )
        return sims, df
    return (scan_runs,)


@app.cell
def _(np):
    # Start + [0, ..., N-1] * step
    z_vals = 0.0 + (np.arange(256) * 2.676511e-04)
    return (z_vals,)


@app.cell
def _(Path, h5py, np, plt, z_vals):
    def plot_run(
        run_h5: Path,
        *,
        vmin: float = -0.2,
        vmax: float = 1.5,
        max_t_pixels: int = 2400,  # downsample time if larger
        max_z_pixels: int = 1200,  # downsample z if larger
        interpolation: str = "nearest",
    ) -> plt.Figure:
        """
        Plot psi(t,z) from run.h5 as an image with correct axes scaling.

        Assumes:
          - /psi has shape (nt, nz) or (time, z)
          - z_vals is a 1D numpy array of length nz (global/outer scope)
        """
        with h5py.File(run_h5, "r") as f:
            psi_ds = f["/psi"]
            t_ds = f["/time"]

            nt, nz = psi_ds.shape

            # sanity check against global z_vals
            if len(z_vals) != nz:
                raise ValueError(
                    f"z_vals has length {len(z_vals)} but /psi has nz={nz}"
                )

            # choose strides to cap displayed resolution (keeps plotting fast)
            t_step = max(1, nt // max_t_pixels)
            z_step = max(1, nz // max_z_pixels)

            t_idx = slice(None, None, t_step)
            z_idx = slice(None, None, z_step)

            # read only what we will actually display
            psi = psi_ds[t_idx, z_idx]  # (nt', nz')
            t = np.asarray(t_ds[t_idx])  # (nt',)
            z = np.asarray(z_vals[z_idx])  # (nz',)

        # imshow expects (rows, cols) = (z, t)
        C = np.asarray(psi, dtype=np.float32).T

        fig, ax = plt.subplots(constrained_layout=True)

        # Map pixels to your physical axes
        extent = (float(t[0]), float(t[-1]), float(z[0]), float(z[-1]))

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
        ax.set_title(run_h5.parent.name)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$\psi(t,z)$")

        return fig
    return (plot_run,)


@app.cell
def _(mo, pd):
    get_df, set_df = mo.state(pd.DataFrame())
    return get_df, set_df


@app.cell
def _(mo):
    base_dir_ui = mo.ui.file_browser(
        selection_mode="directory",
        multiple=False,
        label="Choose data directory!",
    )

    base_dir_ui
    return (base_dir_ui,)


@app.cell
def _(base_dir_ui, mo, scan_runs, set_df):
    selected_dir = base_dir_ui.path(0)  # Path | None

    sims, df_new = scan_runs(selected_dir)
    set_df(
        df_new
    )  # updates state -> reruns dependent cells :contentReference[oaicite:3]{index=3}

    mo.md(f"Scanned `{selected_dir}`: {len(df_new)} runs")
    return


@app.function
def format_mmss(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


@app.cell
def _(get_df, mo):
    df_display = get_df().reset_index(drop=True)

    table = mo.ui.table(
        data=df_display[
            [
                "model",
                "gradient",
                "T",
                "DT",
                "U",
                "PSI",
                "gamma",
                "delta",
                "kappa",
                "NU",
                "MU",
                "run_h5",
                "cluster_id",
                "job_id"
            ]
        ],
        format_mapping={
            "T": format_mmss,
            "DT": "{:.1e}",
            "U": "{:.3e}",
            "gamma": "{:.1e}",
            "delta": "{:.1e}",
            "kappa": "{:.1e}",
            "NU": "{:.3e}",
            "MU": "{:.3e}",
        },
        selection="multi",
        pagination=True,
        show_column_summaries=None,
    )
    table
    return (table,)


@app.cell
def _(Path, mo, plot_run, table):
    selection = table.value
    count = len(selection)

    if count == 0:
        result = mo.md("**Select a run to plot!**")
    elif count == 1:
        result = plot_run(Path(table.value.iloc[0]["run_h5"]), vmin=0.0, vmax=0.5)
    elif count == 2:
        result = mo.hstack(
            [
                plot_run(Path(table.value.iloc[0]["run_h5"]), vmin=0.0, vmax=0.5),
                plot_run(Path(table.value.iloc[1]["run_h5"]), vmin=0.0, vmax=0.5),
            ],
            align="center",
            gap=1
        )
    else:
        result = mo.md("**Too many runs selected!**")

    result
    return


if __name__ == "__main__":
    app.run()
