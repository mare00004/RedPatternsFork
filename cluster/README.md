# HTCondor

This directory contains the submit files and helper scripts for running
`red-patterns` sweeps on the UdS HTCondor cluster.

## Sweep Inputs

The source of truth for a sweep is now `runs.jsonl`.

- Each line is one JSON object.
- Every object must contain a unique `run_id`.
- HTCondor queues plain `run_id`s derived from that file, not raw CLI strings.

Generate a sweep directory with [analysis/gen-params.py](../analysis/gen-params.py).
The notebook writes both `runs.jsonl` and `run_ids.txt`.

## Submit Flow

Use the helper script:

```bash
./submit-sweep.sh path/to/sweep-dir [COMMIT_HASH]
```

This script:

1. Checks that `runs.jsonl` and `run_ids.txt` already exist in the sweep directory.
2. Creates `logs/`, `results/`, and per-`run_id` result directories.
3. Calls `condor_submit` with the run metadata.

You can still call `condor_submit` directly. The relevant submit macros are:

- `COMMIT_HASH`: Docker image tag / commit hash to run.
- `RUN_DIR`: Sweep directory containing `runs.jsonl`, `run_ids.txt`, `logs/`, and `results/`.
- `RUNS_JSONL`: Path to the canonical `runs.jsonl`.
- `RUN_IDS_FILE`: Path to the derived one-run-id-per-line queue file.
- `RUN_TAG`: Batch label used in HTCondor.

## Job Execution

Each queued job executes:

```bash
launch.sh <run_id>
```

Inside the container this resolves to:

```bash
python3 /opt/red-patterns/sweep/run_one.py --runs-jsonl runs.jsonl --run-id <run_id>
```

`run_one.py` is responsible for:

- looking up the matching JSON object by `run_id`
- generating `phi.h5` for every run
- generating `kernel.h5` for convolution runs only
- translating the payload into the final simulation CLI
- running the simulation in the per-job HTCondor scratch working directory
- writing `command.txt`, `run_spec.json`, and `run.h5`
- treating generated `phi.h5` and `kernel.h5` as scratch-only inputs

HTCondor transfers `runs.jsonl` and `launch.sh` into that scratch directory, then
remaps the returned artifacts into `results/<run_id>/` on the submit node.

- `results/<run_id>/command.txt`
- `results/<run_id>/run_spec.json`
- `results/<run_id>/run.h5`

## Monitoring

- `condor_q -nobatch` shows all jobs.
- `condor_q -batch-name <RUN_TAG>` filters by sweep.
- `condor_tail -f <cluster-id>.<proc-id>` streams one job's stdout.

## Notebook Server

For interactive analysis on the cluster, the `marimo.submit`,
`run-marimo.sh`, and `start-notebook-server.sh` helpers still work as before.
