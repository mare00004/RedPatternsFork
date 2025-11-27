# Red Patterns

A Fork of the original [RedPatterns](https://github.com/FelixMaurer/RedPatterns) repository.

## Installation

You can build the simulation from source using [CMake](https://cmake.org/) or download a [docker container](https://www.docker.com/resources/what-container/) with the built binary and all the needed dependencies at [this](https://hub.docker.com/repository/docker/mare00004/cuda-hdf5-dev/general) link.

### Building from Source

You need to have the following packages installed

 - [CMake](https://cliutils.gitlab.io/modern-cmake/chapters/intro/installing.html) Version >=3.15 installed
 - [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
 - [HDF5](https://www.hdfgroup.org/download-hdf5/)

and then run the following code from within the project directory

```bash
mkdir build
cd build

cmake ..
cmake --build .
```

This will build a binary called `red-patterns` for Ampere type NVIDIA GPUs assuming you have all the necessary dependencies installed on your `PATH` ([see](https://en.wikipedia.org/wiki/PATH_(variable))). If you do not have CUDA (or nvcc) and HDF5 on your `PATH` , then you need to tell CMake where they live

```bash
mkdir build
cd build

cmake .. \
    -DCMAKE_CUDA_COMPILER=<path-to-nvcc> \
    -DHDF5_ROOT=<path-to-nvcc>

cmake --build .
```

The paths should look something like this:

 - CUDA: `*/cuda-<version>/bin/nvcc`
 - HDF5: `*/HDF5/build/HDF5_Group/HDF5/<version>`

And if you want to compile for a different GPU architecture you need to add the flag `-DCMAKE_CUDA_ARCHITECTURES=<arch>`. The general build command looks like this:

```bash
mkdir build
cd build

cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=<arch> \
    -DCMAKE_CUDA_COMPILER=<path-to-nvcc> \
    -DHDF5_ROOT=<path-to-nvcc>

cmake --build .
```

## Running

In order to run the simulation you will need to have a CUDA capable GPU on your system.

You can change the following parameters for the simulation via the command line:

 - $N = ...$ ...
 - ...

Any parameter you don't change will take on one of the following default parameters:

 - ...

Theoretically you could run the simulation without any parameters with `./red-patterns` (from within `./build`). Or you could only selectively overwrite the paramters that you actually want to change.

To get a list of all the options you can change run `./red-patterns --help`. This will print:

```
Explanation TODO

Usage:
         red-patterns [COMMON...]
         red-patterns  -c|--use-convolution [COMMON...]
         red-patterns  -t|--use-taylor [--NU=<double>] [--MU=<double>] [COMMON...]

COMMON:
        --T=<double>              total simulation time in seconds
        --DT=<double>             time increment in seconds
        --NO=<int>                time steps between saves
        --gradient=linear|sigmoid Pressure gradient
        --U=<double>              RBC effective interaction energy in Joule
        --PSI=<double>            RBC average volume fraction
        -g, --gamma=<double>      gamma
        -d, --delta=<double>      delta
        -k, --kappa=<double>      kappa
        -o, --out-dir=<file>      directory where simulation data is stored
CONVOLUTION:
        -c, --use-convolution     use convolution integral
TAYLOR:
        -t, --use-taylor          use taylor approximation
        --NU=<double>             interaction nu
        --MU=<double>             interaction mu
```

## Analyzing the Output

 - What is HDF5
 - How can you inspect the data
 - Link to notebook

## Workflows

### For UdS Members

#### Connecting to Cluster and submitting Jobs

If you are a member of UdS and have access to the SIC GPU Cluster, then you can use the following workflow

 1. You can access the HTCondor submission node only from within the university network. Normally you need to be on campus or use CiscoAnyConnect as VPN, but you can do the same from the commandline with `sudo openconnect asa1.uni-saarland.de --user <user> --authgroup "UdS"`
 2. Then you can connect to the submission node via SSH with `ssh <user>@conduit.hpc.uni-saarland.de`. You will need your passphrase for your ssh-key and your user password.
 3. After that you can create a new directory and copy the `sweep.submit` file from the RedPatterns Repository to the cluster. To copy a single file to the cluster (without a gui) you can use `scp`, while connect to your **local** machine: `scp <path-to-file> <user>@<remote>:<path-on-remote>`.
 4. Then you have to create a `params.txt` file on the server in the same directory as the `sweep.submit` file. An example `paramters.txt` file can be found in the repository. For each job you specify all the CLI arguments on a single line.
 5. To submit all the jobs run `condor_submit sweep.submit`. This prints your `<cluster-id>`.
 6. To check if the jobs are running correctly use `condor_q <cluster-id>` or `condor_q -nobatch <cluster-id>.<process-id>` to check the status of a specific job.
 7. If you want to see the `stdout` of the a job run `condor_tail -f <cluster-id>.<process-id>`.

At the end of a jobs execution. The logs and simulation data will be copied to your home directory. (WHERE???)

#### Copying the data back

Use:

```bash
rsync 
```

### For anyone

#### Analyzing the Data

## Explanation

...
