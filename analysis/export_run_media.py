# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "h5py>=3.16.0",
#     "marimo>=0.23.6",
#     "matplotlib>=3.10.9",
#     "numpy>=2.4.5",
#     "pillow>=10",
# ]
# ///

"""Export plots and animations from a Red Patterns ``run.h5`` file.

Launch with ``uv run marimo edit analysis/export_run_media.py``.  Select a
``run.h5`` file, inspect its psi heatmap, then submit the export form to write:

* ``<name>_psi_zt.png`` -- the psi(z, t) heatmap;
* ``<name>_phi_flux_psi_z.<format>`` -- phi(rho, z) with face-flux arrows,
  psi(z), and psi(z, t) with a synchronized time cursor.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="wide")

with app.setup:
    from pathlib import Path
    import sys

    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    NOTEBOOK_FILE = Path(__file__).resolve()
    ANALYSIS_DIR = NOTEBOOK_FILE.parent
    if str(ANALYSIS_DIR) not in sys.path:
        sys.path.insert(0, str(ANALYSIS_DIR))

    from red_patterns import RunData, get_rbc_cmap, plot_psi


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Export run media

    Select a simulation's `run.h5`, preview its $\psi(z,t)$ heatmap, and export
    a PNG plus one synchronized animation. The $\varphi(\rho,z)$ panel contains
    **only face-flux arrows** (no face-velocity arrows); the heatmap below it
    shows the active time with a vertical cursor.
    """)
    return


@app.cell
def _():
    run_picker = mo.ui.file_browser(
        initial_path=Path.cwd(),
        filetypes=[".h5"],
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="file",
        label="Simulation run.h5",
        restrict_navigation=False,
    )
    run_picker
    return (run_picker,)


@app.cell
def _(run_picker):
    mo.stop(not run_picker.value, mo.md("Select a `run.h5` file to continue."))
    selected_run_path = Path(run_picker.path()).resolve()
    selected_run = RunData.from_h5(selected_run_path, load_fields=False)
    psi_heatmap = plot_psi(
        selected_run,
        vmin=0.0,
        vmax=100.0,
        cmap=get_rbc_cmap(),
        title=selected_run_path.parent.name,
    )
    return psi_heatmap, selected_run, selected_run_path


@app.cell
def _(psi_heatmap, selected_run, selected_run_path):
    fields_summary = (
        f"Selected `{selected_run_path}`  \\n+"
        f"Saved frames: `{selected_run.n_saved}` · final time: "
        f"`{selected_run.final_time:.6g} s`"
    )
    mo.vstack([mo.md(fields_summary), mo.as_html(psi_heatmap)])
    return


@app.cell
def _(selected_run_path):
    export_dir = mo.ui.file_browser(
        initial_path=selected_run_path.parent,
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="directory",
        label="Export directory",
        restrict_navigation=False,
    )
    export_name = mo.ui.text(
        value=selected_run_path.parent.name or "run", label="File-name prefix"
    )
    export_format = mo.ui.dropdown(
        options=["gif", "mp4"], value="gif", label="Animation format"
    )
    frame_count = mo.ui.number(
        start=2, stop=300, step=1, value=100, label="Animation frames"
    )
    fps = mo.ui.number(
        start=1, stop=60, step=1, value=15, label="Frames per second (lower = slower)"
    )
    export_form = (
        mo.md(
            """
            ## Export files

            `{{prefix}}_psi_zt.png` and `{{prefix}}_phi_flux_psi_z.{{format}}` will
            be written. The animation places the phi/flux and psi panels next
            to each other, above a full-width psi heatmap with a time cursor.
            GIF is the portable default; MP4 requires an `ffmpeg` executable
            on your PATH.

            {export_dir}

            {export_name}

            {export_format}

            {frame_count}

            {fps}
            """
        )
        .batch(
            export_dir=export_dir,
            export_name=export_name,
            export_format=export_format,
            frame_count=frame_count,
            fps=fps,
        )
        .form(submit_button_label="Export plot and animations", clear_on_submit=False)
    )
    export_form
    return (export_form,)


@app.function
def load_animation_data(run_path):
    """Load the fields required for a consistent pair of animations."""
    with h5py.File(run_path, "r") as h5:
        required = ("fields/phi", "fields/psi", "fields/face_flux", "coords/z_face")
        missing = [key for key in required if key not in h5]
        if missing:
            raise ValueError(
                "The selected run does not contain the required datasets: "
                + ", ".join(missing)
                + ". Re-run it with --store-fields phi,psi,face-flux."
            )
        return (
            np.asarray(h5["fields/phi"], dtype=np.float32),
            np.asarray(h5["fields/psi"], dtype=np.float32),
            np.asarray(h5["fields/face_flux"], dtype=np.float32),
            np.asarray(h5["coords/z_face"], dtype=np.float64),
        )


@app.function
def save_animations(
    *,
    output_dir,
    prefix,
    animation_format,
    frames_per_second,
    requested_frames,
    run,
    phi,
    psi,
    face_flux,
    z_face,
):
    """Write the heatmap and a side-by-side animation; return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = str(animation_format)
    frame_indices = np.unique(
        np.linspace(0, run.n_saved - 1, min(int(requested_frames), run.n_saved), dtype=int)
    )
    if frame_indices.size < 2:
        raise ValueError("At least two saved frames are required for an animation.")

    if extension == "mp4":
        if not FFMpegWriter.isAvailable():
            raise RuntimeError("MP4 export needs ffmpeg on PATH. Select GIF or install ffmpeg.")
        writer = FFMpegWriter(fps=int(frames_per_second))
    else:
        writer = PillowWriter(fps=int(frames_per_second))

    heatmap_path = output_dir / f"{prefix}_psi_zt.png"
    heatmap = plt.figure(constrained_layout=True)
    heatmap_ax = heatmap.add_subplot()
    image = heatmap_ax.imshow(
        100.0 * psi.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=get_rbc_cmap(),
        vmin=0.0,
        vmax=100.0,
        extent=(float(run.time[0]), float(run.time[-1]), float(run.z[0] * 100), float(run.z[-1] * 100)),
    )
    heatmap_ax.set(xlabel=r"$t$ [s]", ylabel=r"$z$ [cm]", title=r"$\psi(z,t)$")
    heatmap.colorbar(image, ax=heatmap_ax, label=r"$\psi$ [%]")
    heatmap.savefig(heatmap_path, dpi=200)
    plt.close(heatmap)

    z_cm = np.asarray(run.z, dtype=np.float64) * 100.0
    rho_cm = np.asarray(run.rho, dtype=np.float64) * 100.0
    z_face_cm = np.asarray(z_face, dtype=np.float64) * 100.0
    phi_min, phi_max = float(np.nanmin(phi)), float(np.nanmax(phi))
    if phi_min == phi_max:
        phi_max = phi_min + 1.0
    rho_count, face_count = face_flux.shape[1:]
    rho_indices = np.unique(np.linspace(0, rho_count - 1, min(12, rho_count), dtype=int))
    face_indices = np.unique(np.linspace(0, face_count - 1, min(18, face_count), dtype=int))
    flux_z, flux_rho = np.meshgrid(z_face_cm[face_indices], rho_cm[rho_indices])
    sample_spacing = (
        float(np.median(np.diff(z_face_cm[face_indices])))
        if face_indices.size > 1
        else 1.0
    )
    arrow_length = 0.65 * sample_spacing

    animation_path = output_dir / f"{prefix}_phi_flux_psi_z.{extension}"
    animation_figure = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    animation_grid = animation_figure.add_gridspec(2, 2, height_ratios=(1.55, 1))
    phi_ax = animation_figure.add_subplot(animation_grid[0, 0])
    psi_ax = animation_figure.add_subplot(animation_grid[0, 1])
    heatmap_ax = animation_figure.add_subplot(animation_grid[1, :])
    phi_image = phi_ax.imshow(
        phi[frame_indices[0]],
        origin="lower",
        aspect="auto",
        extent=(z_cm[0], z_cm[-1], rho_cm[0], rho_cm[-1]),
        vmin=phi_min,
        vmax=phi_max,
        cmap="magma",
    )
    animation_figure.colorbar(phi_image, ax=phi_ax, label=r"$\varphi$")
    phi_ax.set(xlabel=r"$z$ [cm]", ylabel=r"$\rho$ [cm]")
    quiver = phi_ax.quiver(flux_z, flux_rho, np.zeros_like(flux_z), np.zeros_like(flux_z), color="white", width=0.003, scale_units="xy", scale=1)
    phi_ax.set_xlim(z_cm[0], z_cm[-1])
    phi_ax.set_ylim(rho_cm[0], rho_cm[-1])
    phi_ax.margins(x=0, y=0)
    (psi_line,) = psi_ax.plot(
        z_cm, 100.0 * psi[frame_indices[0]], color="#2563eb", linewidth=1.5
    )
    psi_min, psi_max = float(np.nanmin(psi) * 100), float(np.nanmax(psi) * 100)
    if psi_min == psi_max:
        psi_max = psi_min + 1.0
    psi_ax.set(
        xlabel=r"$z$ [cm]",
        ylabel=r"$\psi$ [%]",
        xlim=(z_cm[0], z_cm[-1]),
        ylim=(psi_min, psi_max),
    )
    psi_ax.grid(alpha=0.3)
    animation_heatmap = heatmap_ax.imshow(
        100.0 * psi.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=get_rbc_cmap(),
        vmin=0.0,
        vmax=100.0,
        extent=(float(run.time[0]), float(run.time[-1]), z_cm[0], z_cm[-1]),
    )
    animation_figure.colorbar(
        animation_heatmap, ax=heatmap_ax, label=r"$\psi$ [%]", pad=0.02
    )
    heatmap_ax.set(xlabel=r"$t$ [s]", ylabel=r"$z$ [cm]", title=r"$\psi(z,t)$")
    time_cursor = heatmap_ax.axvline(
        float(run.time[frame_indices[0]]), color="white", linewidth=1.5
    )

    def update_animation(frame_index):
        phi_image.set_data(phi[frame_index])
        sampled_flux = face_flux[frame_index][np.ix_(rho_indices, face_indices)]
        max_abs = float(np.max(np.abs(sampled_flux)))
        arrows = np.zeros_like(sampled_flux) if max_abs == 0 else arrow_length * sampled_flux / max_abs
        quiver.set_UVC(arrows, np.zeros_like(arrows))
        phi_ax.set_title(rf"$\varphi(\rho,z)$ with face flux at $t={run.time[frame_index]:.4g}$ s")
        psi_line.set_ydata(100.0 * psi[frame_index])
        psi_ax.set_title(rf"$\psi(z)$ at $t={run.time[frame_index]:.4g}$ s")
        time_cursor.set_xdata([run.time[frame_index], run.time[frame_index]])
        return phi_image, quiver, psi_line, time_cursor

    animation = FuncAnimation(
        animation_figure, update_animation, frames=frame_indices, blit=False
    )
    animation.save(animation_path, writer=writer, dpi=150)
    plt.close(animation_figure)
    return heatmap_path, animation_path


@app.cell
def _(export_form, selected_run, selected_run_path):
    if export_form.value is None:
        export_result = mo.md("Submit the form to write the selected run's files.")
    else:
        values = export_form.value
        directory_entries = values.get("export_dir") or []
        prefix = str(values.get("export_name", "")).strip()
        if not directory_entries:
            export_result = mo.md("Please select an export directory.")
        elif not prefix:
            export_result = mo.md("Please enter a file-name prefix.")
        else:
            phi, psi, face_flux, z_face = load_animation_data(selected_run_path)
            heatmap_path, animation_path = save_animations(
                output_dir=Path(directory_entries[0].path),
                prefix=prefix,
                animation_format=values["export_format"],
                frames_per_second=int(values["fps"]),
                requested_frames=int(values["frame_count"]),
                run=selected_run,
                phi=phi,
                psi=psi,
                face_flux=face_flux,
                z_face=z_face,
            )
            export_result = mo.md(
                f"Saved:\n\n- `{heatmap_path}`\n- `{animation_path}`"
            )
    export_result
    return


if __name__ == "__main__":
    app.run()
