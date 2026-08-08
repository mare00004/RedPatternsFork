# /// script
# dependencies = [
#     "h5py==3.16.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "pydantic==2.13.4",
#     "wigglystuff==0.3.3",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="wide")

with app.setup:
    import shlex
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    import marimo as mo
    from wigglystuff import ProgressBar

    NOTEBOOK_FILE = (
        Path(__file__).resolve()
        if "__file__" in globals()
        else (Path.cwd() / "analysis" / "test_runs_jsonl.py").resolve()
    )
    ANALYSIS_DIR = NOTEBOOK_FILE.parent
    REPO_ROOT = ANALYSIS_DIR.parent
    for _path in (str(ANALYSIS_DIR), str(REPO_ROOT)):
        if _path not in sys.path:
            sys.path.insert(0, _path)

    from red_patterns.models import ConvRun, TaylorRun, run_payload_adapter
    from red_patterns.phi import PhiConfig, compute_phi, plot_phi, write_phi_h5
    from red_patterns.sim import (
        DEFAULT_POLL_SEC,
        DEFAULT_STALE_SEC,
        estimate_total_steps,
        locate_binary,
        progress_summary,
        read_progress,
    )
    from red_patterns.runs import RunData, get_rbc_cmap, plot_psi
    from sweep import run_one


    _sample_jsonl = REPO_ROOT / "data" / "runs.jsonl"
    DEFAULT_JSONL_LINE = (
        _sample_jsonl.read_text(encoding="utf-8").splitlines()[0]
        if _sample_jsonl.exists()
        else '{"run_id":"r000001","variant":"taylor","N":256,"T":1000.0,'
        '"DT":0.001,"storeTime":1.0,"gradient":"linear",'
        '"phi":{"mode":"generate","params":{"psi_avg":0.02,"N":256,"wing":32,'
        '"rho_center":1100.0,"rho_span":30.0,"dz":0.000267651,'
        '"phi_type":"homogeneous"}},"NU":-2.832638e-32,'
        '"MU":-4.4684449999999995e-39}'
    )


    def fmt_value(value: object) -> str:
        """Render a payload scalar value for the summary table."""
        if isinstance(value, (int, float)):
            return f"{value:g}"
        return str(value)


    def payload_table(run: TaylorRun | ConvRun) -> str:
        """Render one parsed ``RunPayload`` as a markdown table."""
        rows = [
            ("run_id", run.run_id),
            ("variant", run.variant.value),
            ("N", fmt_value(run.N)),
            ("T", fmt_value(run.T)),
            ("DT", fmt_value(run.DT)),
            ("storeTime", fmt_value(run.storeTime)),
            ("gradient", run.gradient.value),
        ]
        for key, value in run.phi.params.model_dump().items():
            if key != "phi_type":
                rows.append((f"phi.{key}", fmt_value(value)))
        if isinstance(run, TaylorRun):
            rows.append(("NU", f"{run.NU:.6e}"))
            rows.append(("MU", f"{run.MU:.6e}"))
        else:
            for key, value in run.kernel.params.items():
                rows.append((f"kernel.{key}", fmt_value(value)))
        table = "| Key | Value |\n|-----|-------|\n"
        table += "\n".join(f"| `{key}` | `{value}` |" for key, value in rows)
        return table


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # runs.jsonl → local simulation test

    A manual smoke test for the `runs.jsonl` → `red-patterns` pipeline. Paste a
    single line from a `runs.jsonl`, preview the parsed payload, then run the
    simulation locally through the exact same helpers the cluster wrapper
    (`sweep/run_one.py`) uses — with a live progress bar, exactly like
    `workbench.py`.
    """)
    return


@app.cell
def cell_binary():
    _default_binary = locate_binary(REPO_ROOT)
    ui_binary = mo.ui.file_browser(
        initial_path=_default_binary.parent,
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="file",
        label="Simulation binary",
    )
    _built = Path(_default_binary).exists()
    mo.vstack(
        [
            mo.md(
                f"Default binary: `{_default_binary}`  \n"
                + (
                    "**Found on disk.**"
                    if _built
                    else "**Not built yet** — run "
                    "`cmake --preset release && cmake --build build/release`."
                )
            ),
            ui_binary,
        ]
    )
    return (ui_binary,)


@app.cell
def cell_input():
    ui_line = mo.ui.text_area(
        value=DEFAULT_JSONL_LINE,
        rows=3,
        label="Paste a single runs.jsonl line",
        placeholder='{"run_id": "r000001", ...}',
    )
    ui_run_button = mo.ui.run_button(label="Parse and run simulation")
    mo.vstack([ui_line, ui_run_button], gap=0.5)
    return ui_line, ui_run_button


@app.cell
def cell_parse(ui_line):
    _line = ui_line.value.strip()
    if not _line:
        run_payload = None
        parsed_md = mo.md(
            "Paste a `runs.jsonl` line above — this cell parses it live as you type."
        )
    else:
        try:
            run_payload = run_payload_adapter.validate_json(_line)
            parsed_md = mo.md(f"### Parsed payload\n\n{payload_table(run_payload)}")
        except (ValueError, TypeError) as exc:
            run_payload = None
            parsed_md = mo.callout(f"Invalid runs.jsonl line: {exc}", kind="danger")
    parsed_md
    return (run_payload,)


@app.cell
def cell_run(run_payload, ui_binary, ui_run_button):
    if not ui_run_button.value:
        run_h5 = None
        phi_result = None
        run_md = mo.md(
            "Configure the paste + binary, then click **Parse and run simulation**."
        )
    elif run_payload is None:
        run_h5 = None
        phi_result = None
        run_md = mo.md("Fix the JSONL line before running.")
    elif not ui_binary.value:
        run_h5 = None
        phi_result = None
        run_md = mo.md("Select a simulation binary first.")
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="jsonl_test_"))
        phi_path = work_dir / "phi.h5"
        kernel_path = (
            work_dir / "kernel.h5" if isinstance(run_payload, ConvRun) else None
        )

        phi_config = PhiConfig.from_params(run_payload.phi.params, phi_path)
        phi_result = compute_phi(phi_config)
        write_phi_h5(phi_path, phi_result, phi_config)

        if isinstance(run_payload, ConvRun):
            from red_patterns.kernel import compute_kernel, write_kernel_h5

            kernel_config = run_one.kernel_config_from_params(
                run_payload.kernel.params, kernel_path
            )
            kernel_result = compute_kernel(kernel_config)
            write_kernel_h5(kernel_path, kernel_result, kernel_config)

        binary = Path(ui_binary.value[0].path)
        cli = run_one.cli_args_from_payload(
            run=run_payload,
            binary_path=binary,
            out_dir=work_dir,
            phi_path=phi_path,
            kernel_path=kernel_path,
        )
        (work_dir / "command.txt").write_text(shlex.join(cli), encoding="utf-8")

        progress = mo.ui.anywidget(
            ProgressBar(
                value=0,
                max_value=estimate_total_steps(float(run_payload.T), float(run_payload.DT)),
                color="#22c55e",
                show_text=False,
                width="100%",
                height=24,
            )
        )
        progress_path = work_dir / "progress.json"
        snapshot = None
        returncode = None
        proc = subprocess.Popen(cli, cwd=REPO_ROOT)
        last_seen = time.monotonic()
        while True:
            maybe = read_progress(progress_path)
            if maybe is not None:
                snapshot = maybe
                last_seen = time.monotonic()
                total = max(1, int(snapshot.get("total_steps", 1)))
                progress.max_value = total
                progress.value = min(total, max(0, int(snapshot.get("step", 0))))
            returncode = proc.poll()
            age = time.monotonic() - last_seen
            status = None if snapshot is None else str(snapshot.get("status", ""))
            mo.output.replace(
                mo.vstack(
                    [
                        progress,
                        mo.md(
                            progress_summary(
                                snapshot=snapshot,
                                t_final=float(run_payload.T),
                                is_waiting=snapshot is None,
                                is_stale=age > DEFAULT_STALE_SEC,
                                returncode=returncode,
                            )
                        ),
                    ],
                    gap=1,
                )
            )
            if returncode is not None or status in {"finished", "failed"}:
                break
            time.sleep(DEFAULT_POLL_SEC)
        proc.wait()

        run_h5 = work_dir / "run.h5"
        run_md = mo.vstack(
            [
                progress,
                mo.md(f"`run_h5 = {run_h5}`"),
                mo.md(f"`returncode = {returncode}`"),
                mo.md(f"`command = {shlex.join(cli)}`"),
            ],
            gap=1,
        )
    run_md
    return phi_result, run_h5


@app.cell
def cell_inspect(run_h5):
    mo.stop(
        run_h5 is None or not run_h5.exists(),
        mo.md("Run a simulation to inspect the resulting `run.h5`."),
    )
    inspect_run = RunData.from_h5(run_h5, load_fields=False)
    inspect_md = mo.md(
        f"**Active file:** `{run_h5}`  \n"
        f"`n_saved = {inspect_run.n_saved}`, `final_time = {inspect_run.final_time}`"
    )
    return (inspect_run,)


@app.cell
def cell_psi_heatmap(inspect_run, run_h5):
    psi_fig = plot_psi(
        inspect_run,
        vmin=0.0,
        vmax=100.0,
        cmap=get_rbc_cmap(),
        title=run_h5.parent.name,
    )
    psi_fig
    return


@app.cell
def cell_psi_profile(inspect_run):
    _z_cm = np.asarray(inspect_run.z, dtype=np.float64) * 100.0
    _psi_pct = np.asarray(inspect_run.load_psi(), dtype=np.float64)[-1] * 100.0
    _fig, _ax = plt.subplots(constrained_layout=True)
    _ax.plot(_z_cm, _psi_pct)
    _ax.set_xlabel("$z$ [cm]")
    _ax.set_ylabel(r"$\psi$ [%]")
    _ax.set_title(r"$\psi(z)$ at the latest time step")
    mo.ui.matplotlib(_fig.gca())
    return


@app.cell
def cell_phi_plot(phi_result):
    mo.stop(
        phi_result is None,
        mo.md("Run a simulation to see the generated initial phi field."),
    )
    phi_fig = plot_phi(phi_result)
    phi_fig
    return


if __name__ == "__main__":
    app.run()
