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
./submit-sweep.sh path/to/sweep-dir IMAGE_TAG
```

This script:

1. Checks that `runs.jsonl` and `run_ids.txt` already exist in the sweep directory.
2. Creates `logs/`, `results/`, and per-`run_id` result directories.
3. Calls `condor_submit` with the run metadata and requested Docker image tag.

Build, push, and verify the exact image tag before submitting it:

```bash
TAG="$(git rev-parse --short HEAD)"
./build-docker.sh "$TAG"
docker run --rm "mare00004/cuda-hdf5-dev:$TAG" \
  /opt/red-patterns/.venv/bin/python -c 'import h5py, numpy, pydantic; print(h5py.__version__)'
./submit-sweep.sh path/to/sweep-dir "$TAG"
```

You can still call `condor_submit` directly. The relevant submit macros are:

- `COMMIT_HASH`: required Docker image tag / commit hash to run.
- `RUN_DIR`: Sweep directory containing `runs.jsonl`, `run_ids.txt`, `logs/`, and `results/`.
- `RUNS_JSONL`: Path to the canonical `runs.jsonl`.
- `RUN_IDS_FILE`: Path to the derived one-run-id-per-line queue file.
- `RUN_TAG`: Batch label used in HTCondor.

When submitting directly, always provide `COMMIT_HASH=<IMAGE_TAG>`; the
placeholder default is deliberately not a usable Docker tag.

## Job Execution

Each queued job executes:

```bash
launch.sh <run_id>
```

Inside the container this resolves to:

```bash
/opt/red-patterns/.venv/bin/python /opt/red-patterns/sweep/run_one.py --runs-jsonl runs.jsonl --run-id <run_id>
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
- `results/<run_id>/runtime-diagnostics.txt`

`runtime-diagnostics.txt` records the requested tag, the revision embedded at
image build time, and the virtualenv locations and versions of the Python sweep
dependencies. If it shows a revision different from the requested tag even
though the local `docker run` verification succeeds, include this artifact and
the job's execute host when reporting the worker image-cache issue to the
cluster administrators.

## Monitoring

- `condor_q -nobatch` shows all jobs.
- `condor_q -batch-name <RUN_TAG>` filters by sweep.
- `condor_tail -f <cluster-id>.<proc-id>` streams one job's stdout.

## Notebook Server

For interactive analysis on the cluster, the `marimo.submit`,
`run-marimo.sh`, and `start-notebook-server.sh` helpers still work as before.
