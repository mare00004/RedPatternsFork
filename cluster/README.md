# HTCondor

This directory contains HTCondor Submit files as well as wrapper scripts for running the [RedPatternsFork](https://github.com/mare00004/RedPatternsFork/) simulation on the UdS HTCondor Cluster and analyzing the data produced.

## Running the Simulation [🔗](https://wiki.cs.uni-saarland.de/en/HPC/faq)

By default the submit file `sweep.submit` runs the latest working docker container of the simulation for each of the parameters specified in `parameters.txt`. You can change the default behaviour of the submit file by overwriting the following parameters:

- `COMMIT_HASH`: The commit hash of the simulation used for building the docker container. You can find the available docker containers on [DockerHub](https://hub.docker.com/r/mare00004/cuda-hdf5-dev).
- `RUN_DIR`: The directory where the results of the simulation should be stored. By default this will be the same directory where the `sweep.submit` file lives.
- `PARAMS_FILE`: The file from which to read the parameters. This file should consist of the CLI Arguments for a single simulation per line.
- `RUN_TAG`: A tag to easily find your simulations on the cluster.

The general command looks as follows:

```bash
condor_submit COMMIT_HASH=... RUN_DIR=... PARAMS_FILE=... RUN_TAG=... sweep.submit
```

The sweep.submit file internally calls the `run-sim.sh` script, which is a wrapper around the main simulation binary. After the sweep is done you fill find a `logs/` and a `results/` director inside `RUN_DIR` with one subdirectory for each run, identified by its Process- and Job-ID.

> [!info] Tips
>
> - You can check the status of (all) your jobs with `condor_q -nobatch`. If you know the `RUN_TAG` then you can use `condor_q -batch-name <RUN_TAG>`.

## Analyzing the Results

You can analyze the data yourself by copying the `run.h5` file for your selected job and then doing whatever you want with it. But you can also use the provided [marimo](https://marimo.io/) notebooks in the `analysis/` directory. These notebooks can simply be executed as you would any other marimo notebook for data that you have copied to your local machine, or have otherweise gathered access to, but this repository also provies an HTCondor submit file that allows you to start a **notebook server** on an execution node of the cluster. To do that you have to:

1. Copy the `marimo.submit` and `run-marimo.sh` files as well as the `start-notebook-server.sh` script to the submit node of the HTCondor cluster.
2. Start the notebook server, by running the `start-notebook-server.sh` script. Then wait a couple seconds.
3. Forward the port `3718` from the cluster to your local machine. This can be done from the command line with `ssh -N -L 3718:127.0.0.1:3718 <cluster>`.
4. Open the notebook server in a browser of your choice at the domain `localhost:3718`. The password is `password` (which should be safe to expose since you need the ssh key to login to the cluster anyway).
