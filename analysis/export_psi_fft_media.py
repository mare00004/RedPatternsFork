"""Export a ``psi(z, t)`` heatmap together with its spatial FFT spectrum.

Launch with ``uv run marimo edit analysis/export_psi_fft_media.py``. Select a
simulation ``run.h5`` and submit the form to write a combined PNG plus a
synchronized GIF or MP4 of the psi profile and its FFT mode-amplitude traces.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="wide")


with app.setup:
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    from red_patterns import RunData, get_rbc_cmap


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Export psi heatmap and FFT

    Select a simulation's `run.h5` and export one PNG with $\psi(z,t)$ above
    the mode-amplitude traces over time of
    $\delta\psi(z,t) = \psi(z,t) - \overline{\psi(t)}_z$. The DC component is
    omitted from the FFT panel.
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
    mo.stop(
        selected_run.n_saved < 2,
        mo.md("The selected run needs at least two saved timesteps."),
    )
    psi = np.asarray(selected_run.load_psi(), dtype=np.float64)
    time = np.asarray(selected_run.time, dtype=np.float64)
    z_cm = 100.0 * np.asarray(selected_run.z, dtype=np.float64)
    return psi, selected_run, selected_run_path, time, z_cm


@app.function
def fft_amplitude(psi, z_cm):
    """Return non-DC spatial rFFT amplitudes and frequencies in cm^-1."""
    if psi.shape[1] < 2:
        raise ValueError("At least two z points are required for the FFT.")
    dz_cm = float(z_cm[1] - z_cm[0])
    if dz_cm <= 0.0:
        raise ValueError("The z coordinate must be strictly increasing.")
    delta_psi = psi - psi.mean(axis=1, keepdims=True)
    coefficients = np.fft.rfft(delta_psi, axis=1)
    frequencies = np.fft.rfftfreq(psi.shape[1], d=dz_cm)
    return np.abs(coefficients[:, 1:]), frequencies[1:]


@app.function
def make_psi_fft_figure(*, psi, time, z_cm, log_fft):
    """Create the paired psi(z,t) and spatial-FFT heatmaps."""
    amplitudes, frequencies = fft_amplitude(psi, z_cm)
    if frequencies.size == 0:
        raise ValueError("The selected run does not contain a non-DC FFT mode.")
    mode_numbers = np.arange(1, amplitudes.shape[1] + 1)

    figure, (psi_ax, fft_ax) = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        sharex=True,
        constrained_layout=True,
        height_ratios=(1, 1),
    )
    psi_image = psi_ax.imshow(
        100.0 * psi.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=get_rbc_cmap(),
        vmin=0.0,
        vmax=100.0,
        extent=(time[0], time[-1], z_cm[0], z_cm[-1]),
    )
    figure.colorbar(psi_image, ax=psi_ax, label=r"$\psi$ [%]")
    psi_ax.set(ylabel=r"$z$ [cm]", title=r"$\psi(z,t)$")

    mode_colormap = plt.get_cmap("viridis")
    mode_scale = max(1, int(mode_numbers[-1] - mode_numbers[0]))
    for column, mode in enumerate(mode_numbers):
        fft_ax.plot(
            time,
            amplitudes[:, column],
            color=mode_colormap((mode - mode_numbers[0]) / mode_scale),
            linewidth=1.0,
        )
    mode_colors = plt.cm.ScalarMappable(cmap=mode_colormap)
    mode_colors.set_clim(float(mode_numbers[0]), float(max(2, mode_numbers[-1])))
    figure.colorbar(mode_colors, ax=fft_ax, label="Mode number n")
    if log_fft and np.any(amplitudes > 0.0):
        fft_ax.set_yscale("log")
    fft_ax.set(
        xlabel=r"$t$ [s]",
        ylabel=r"$A_n(t) = |\delta\hat{\psi}_n(t)|$",
        title="FFT amplitude of every mode over time",
    )
    return figure


@app.function
def save_psi_fft_animation(
    *,
    output_path,
    psi,
    time,
    z_cm,
    log_fft,
    frames_per_second,
    requested_frames,
):
    """Write a synchronized psi-profile, FFT-profile, and heatmap animation."""
    amplitudes, frequencies = fft_amplitude(psi, z_cm)
    if frequencies.size == 0:
        raise ValueError("The selected run does not contain a non-DC FFT mode.")
    mode_numbers = np.arange(1, amplitudes.shape[1] + 1)

    frame_indices = np.unique(
        np.linspace(0, psi.shape[0] - 1, min(int(requested_frames), psi.shape[0]), dtype=int)
    )
    if frame_indices.size < 2:
        raise ValueError("At least two saved frames are required for an animation.")
    if output_path.suffix == ".mp4":
        if not FFMpegWriter.isAvailable():
            raise RuntimeError("MP4 export needs ffmpeg on PATH. Select GIF or install ffmpeg.")
        writer = FFMpegWriter(fps=int(frames_per_second))
    else:
        writer = PillowWriter(fps=int(frames_per_second))

    figure = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.35, 1))
    profile_ax = figure.add_subplot(grid[0, 0])
    spectrum_ax = figure.add_subplot(grid[0, 1])
    heatmap_ax = figure.add_subplot(grid[1, :])

    first_frame = int(frame_indices[0])
    (psi_line,) = profile_ax.plot(z_cm, 100.0 * psi[first_frame], color="#2563eb")
    psi_min, psi_max = 100.0 * float(np.nanmin(psi)), 100.0 * float(np.nanmax(psi))
    if psi_min == psi_max:
        psi_max = psi_min + 1.0
    profile_ax.set(
        xlabel=r"$z$ [cm]",
        ylabel=r"$\psi$ [%]",
        xlim=(z_cm[0], z_cm[-1]),
        ylim=(psi_min, psi_max),
    )
    profile_ax.grid(alpha=0.3)

    mode_colormap = plt.get_cmap("viridis")
    mode_scale = max(1, int(mode_numbers[-1] - mode_numbers[0]))
    for column, mode in enumerate(mode_numbers):
        spectrum_ax.plot(
            time,
            amplitudes[:, column],
            color=mode_colormap((mode - mode_numbers[0]) / mode_scale),
            linewidth=1.0,
        )
    mode_colors = plt.cm.ScalarMappable(cmap=mode_colormap)
    mode_colors.set_clim(float(mode_numbers[0]), float(max(2, mode_numbers[-1])))
    figure.colorbar(mode_colors, ax=spectrum_ax, label="Mode number n")
    if log_fft and np.any(amplitudes > 0.0):
        spectrum_ax.set_yscale("log")
    spectrum_ax.set(
        xlabel=r"$t$ [s]",
        ylabel=r"$A_n(t) = |\delta\hat{\psi}_n(t)|$",
        title="FFT amplitude of every mode over time",
    )
    fft_time_cursor = spectrum_ax.axvline(time[first_frame], color="white", linewidth=1.5)

    heatmap = heatmap_ax.imshow(
        100.0 * psi.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=get_rbc_cmap(),
        vmin=0.0,
        vmax=100.0,
        extent=(time[0], time[-1], z_cm[0], z_cm[-1]),
    )
    figure.colorbar(heatmap, ax=heatmap_ax, label=r"$\psi$ [%]", pad=0.02)
    heatmap_ax.set(xlabel=r"$t$ [s]", ylabel=r"$z$ [cm]", title=r"$\psi(z,t)$")
    time_cursor = heatmap_ax.axvline(time[first_frame], color="white", linewidth=1.5)

    def update_animation(frame_index):
        psi_line.set_ydata(100.0 * psi[frame_index])
        profile_ax.set_title(rf"$\psi(z)$ at $t={time[frame_index]:.4g}$ s")
        fft_time_cursor.set_xdata([time[frame_index], time[frame_index]])
        time_cursor.set_xdata([time[frame_index], time[frame_index]])
        return psi_line, fft_time_cursor, time_cursor

    animation = FuncAnimation(figure, update_animation, frames=frame_indices, blit=False)
    animation.save(output_path, writer=writer, dpi=150)
    plt.close(figure)


@app.cell
def _():
    preview_log_fft = mo.ui.switch(value=True, label="Use logarithmic FFT color scale")
    return (preview_log_fft,)


@app.cell
def _(preview_log_fft, psi, time, z_cm):
    preview_figure = make_psi_fft_figure(
        psi=psi, time=time, z_cm=z_cm, log_fft=preview_log_fft.value
    )
    preview = mo.vstack([preview_log_fft, mo.as_html(preview_figure)])
    preview
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
    frame_count = mo.ui.number(start=2, stop=300, step=1, value=100, label="Animation frames")
    fps = mo.ui.number(
        start=1, stop=60, step=1, value=15, label="Frames per second (lower = slower)"
    )
    export_form = (
        mo.md(
            """
            ## Export files

            `{{prefix}}_psi_zt_fft.png` will contain the psi heatmap and its
            FFT amplitude traces over time. `{{prefix}}_psi_fft.{{format}}` will
            animate the current psi profile above FFT traces and the psi heatmap,
            with synchronized time cursors. The Fourier transform removes each
            frame's spatial mean and excludes mode zero.

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
        .form(submit_button_label="Export PNG and animation", clear_on_submit=False)
    )
    export_form
    return (export_form,)


@app.cell
def _(export_form, preview_log_fft, psi, time, z_cm):
    if export_form.value is None:
        export_result = mo.md("Submit the form to write the combined PNG and animation.")
    else:
        values = export_form.value
        directory_entries = values.get("export_dir") or []
        prefix = str(values.get("export_name", "")).strip()
        if not directory_entries:
            export_result = mo.md("Please select an export directory.")
        elif not prefix:
            export_result = mo.md("Please enter a file-name prefix.")
        else:
            output_dir = Path(directory_entries[0].path)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{prefix}_psi_zt_fft.png"
            export_figure = make_psi_fft_figure(
                psi=psi,
                time=time,
                z_cm=z_cm,
                log_fft=preview_log_fft.value,
            )
            export_figure.savefig(output_path, dpi=200)
            plt.close(export_figure)
            animation_path = output_dir / f"{prefix}_psi_fft.{values['export_format']}"
            save_psi_fft_animation(
                output_path=animation_path,
                psi=psi,
                time=time,
                z_cm=z_cm,
                log_fft=preview_log_fft.value,
                frames_per_second=int(values["fps"]),
                requested_frames=int(values["frame_count"]),
            )
            export_result = mo.md(f"Saved:\n\n- `{output_path}`\n- `{animation_path}`")
    export_result
    return


if __name__ == "__main__":
    app.run()
