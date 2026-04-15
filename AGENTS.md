# Agents.md

Guidance for AI agents working with the Red Patterns codebase.

## Project Overview

Red Patterns is a CUDA-based simulation for modeling red blood cell patterns. It's a fork of the original [RedPatterns](https://github.com/FelixMaurer/RedPatterns) repository.

**Tech Stack:**

- **CUDA** - GPU-accelerated computation
- **C/C++** - Host code and CUDA kernels
- **CMake** - Build system (version >= 3.15)
- **HDF5** - Simulation data storage
- **Python** - Post-processing and analysis (h5py, numpy, matplotlib, marimo)

**Output:** Simulation produces `run.h5` HDF5 files containing all simulation data and configuration for reproducibility.

## Build Instructions

### Prerequisites

- CMake >= 3.15
- CUDA Toolkit (nvcc compiler)
- HDF5 library

### Environment Setup (Recommended)

Use mamba/conda with the provided environment file:

```bash
mamba env create -f environment.yml
mamba activate cuda-dev
```

This provides CUDA 12.8, GCC 14, CMake, and HDF5 1.14.

### Build Commands

**Using CMake presets:**

```bash
# Release build (optimized)
cmake --preset release
cmake --build build/release

# Debug build (with CUDA debug symbols)
cmake --preset dev-debug
cmake --build build/dev-debug
```

**Manual CMake (if presets unavailable):**

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build .
```

**Custom CUDA architecture:**

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=120  # For RTX 50-series
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86  # For RTX 30-series
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80  # For A100
```

### Build Output

- Binary: `build/release/red-patterns` or `build/dev-debug/red-patterns`
- Compile commands: `build/*/compile_commands.json` (for clangd)

## Running Simulations

### CLI Usage

```bash
./build/release/red-patterns [OPTIONS]
```

**Required mode selection:**

- `-c, --use-convolution` - Use convolution integral method
- `-t, --use-taylor` - Use Taylor approximation method

**Common parameters:**

- `--T=<double>` - Total simulation time (seconds)
- `--DT=<double>` - Time increment (seconds)
- `--NO=<int>` - Time steps between saves
- `--gradient=linear|sigmoid` - Pressure gradient type
- `--U=<double>` - RBC effective interaction energy (Joule)
- `--PSI=<double>` - RBC average volume fraction
- `-g, --gamma=<double>` - Gamma parameter
- `-d, --delta=<double>` - Delta parameter
- `-k, --kappa=<double>` - Kappa parameter
- `-o, --out-dir=<file>` - Output directory for simulation data

**Taylor-specific:**

- `--NU=<double>` - Interaction nu
- `--MU=<double>` - Interaction mu

**Help:**

```bash
./build/release/red-patterns --help
```

### Makefile Targets

```bash
make run-tayl  # Taylor method with sigmoid gradient
make run-conv  # Convolution method with sigmoid gradient
make test      # Quick test run (T=1.0, minimal steps)
```

## Code Style & Conventions

### Formatting

The project uses `.clang-format` with these key settings:

- Base: LLVM style
- Indent: 4 spaces (no tabs)
- `BreakBeforeBraces: Attach`
- `BinPackArguments: false`
- No column limit

**Format code before committing:**

```bash
clang-format -i src/*.cu src/*.c include/*.h include/*.cuh
```

### Compiler Flags

- C: `-Wall -Wextra`
- CUDA: `-Xcompiler=-Wall,-Wextra`
- Debug builds: `-g -G -O0` (host + device debug symbols)

### File Naming Conventions

| Extension | Purpose                                 |
| --------- | --------------------------------------- |
| `.cu`     | CUDA source files (kernels + host code) |
| `.cuh`    | CUDA header files                       |
| `.c`      | Pure C source files                     |
| `.h`      | Pure C header files                     |

### Key Data Structures

Defined in `include/sim_types.h`:

```c
typedef struct {
    ModelType modelType;      // CONV or TAYL
    GradientType gradientType; // LINEAR or SIGMOID
    double U, PSI, alpha, beta, gamma, delta, kappa;
    union { ConvParams; TaylParams; } variant;
} ModelParams;

typedef struct {
    int N, NT, NO;
    double T, DT, DZ, fineDZ, sysL;
    char outDir[256];
} RunParams;

typedef struct {
    RunParams run;
    ModelParams model;
} SimConfig;
```

### Documentation

The project uses Doxygen-style comments. The `.clangd` configuration enables `-Wdocumentation` warnings.

## Project Architecture

### Source Files (`src/`)

| File             | Purpose                                     |
| ---------------- | ------------------------------------------- |
| `main.cu`        | Entry point, orchestrates simulation        |
| `cli.c`          | Command-line argument parsing (argtable3)   |
| `config.c`       | Configuration initialization and validation |
| `gpu_state.cu`   | GPU memory management and state             |
| `cuda_kernel.cu` | CUDA kernels for simulation                 |
| `simulations.cu` | High-level simulation logic                 |
| `hdf5_file.c`    | HDF5 file I/O for results                   |

### Header Files (`include/`)

| File              | Purpose                                 |
| ----------------- | --------------------------------------- |
| `sim_types.h`     | Core type definitions (SimConfig, etc.) |
| `parameters.h`    | Default parameter values                |
| `config.h`        | Configuration functions                 |
| `cli.h`           | CLI parsing interface                   |
| `cuda_kernel.cuh` | CUDA kernel declarations                |
| `cuda_utils.cuh`  | CUDA utility functions                  |
| `gpu_state.cuh`   | GPU state management                    |
| `gradient.cuh`    | Gradient computation (linear/sigmoid)   |
| `simulations.cuh` | Simulation interface                    |
| `hdf5_file.h`     | HDF5 I/O interface                      |

### Data Flow

```
CLI args → SimConfig → GPU State → CUDA Kernels → HDF5 Output
                                              ↓
                                    Python Analysis (analysis/)
```

### HDF5 Output Structure

The `run.h5` file contains:

- Root attributes: `git_commit`, `git_state`, `schema_version`, `created_utc`
- `/config/run` - Run parameters
- `/config/model` - Model parameters
- `/config/model/variant/conv` or `/config/model/variant/tayl` - Variant params
- `/time` - Time array
- `/coords/rho`, `/coords/z` - Spatial coordinates
- `/fields/phi` - Phi field data (3D: time × rho × z)
- `/fields/psi` - Psi field data (2D: time × z)

## Testing

**Note:** This project does not have a formal test framework.

The `make test` target runs a quick simulation for validation:

```bash
make test  # T=1.0, NO=1, minimal computation
```

For development testing, use short simulation runs:

```bash
./build/release/red-patterns --use-taylor --T=1.0 --DT=5e-04 --NO=1 \
    --gradient=sigmoid --U=1.1115e-16 --PSI=0.02 \
    --gamma=1.8e-10 --delta=1e-11 --kappa=0.0 \
    --NU=-1.6049962938777745e-29 --MU=-7.052525226362305e-36 \
    --out-dir=./data/test
```

## Analysis Tools

### Python Scripts (`analysis/`)

| File                    | Purpose                                                        |
| ----------------------- | -------------------------------------------------------------- |
| `red_patterns.py`       | Core data structures for loading HDF5 (`RunData`, `RunConfig`) |
| `analyze.py`            | Analysis workflows                                             |
| `analyze_single_run.py` | Single run analysis                                            |
| `convolution.py`        | Convolution-specific analysis                                  |
| `gradient.py`           | Gradient computation utilities                                 |
| `gen-params.py`         | Parameter sweep generation                                     |
| `notebook.py`           | Marimo notebook utilities                                      |

### Required Python Packages

```python
h5py      # HDF5 file reading
numpy     # Numerical computation
matplotlib # Plotting
```

### Loading Simulation Data

```python
from pathlib import Path
from analysis.red_patterns import RunData

run = RunData.from_h5(Path("data/tayl_sigmoid/run.h5"), load_fields=True)

# Access data
print(run.config.model.modelType)  # "CONV" or "TAYL"
print(run.time.shape)              # (n_saved,)
print(run.phi.shape)               # (n_saved, N, N)
print(run.psi.shape)               # (n_saved, N)
```

### Plotting

```python
from analysis.red_patterns import plot_psi

fig = plot_psi(run, vmin=0.0, vmax=1.0)
fig.savefig("psi_plot.png")
```

## Cluster/HTCondor Workflow (UdS Members)

### Connecting to Cluster

1. **VPN (if off-campus):**

   ```bash
   sudo openconnect asa1.uni-saarland.de --user <user> --authgroup "UdS"
   ```

2. **SSH to submission node:**
   ```bash
   ssh <user>@conduit.hpc.uni-saarland.de
   ```

### Submitting Jobs

1. **Copy files to cluster:**

   ```bash
   scp cluster/sweep.submit cluster/run-sim.sh <user>@conduit.hpc.uni-saarland.de:~/
   scp params.txt <user>@conduit.hpc.uni-saarland.de:~/
   ```

   Or mount remotely:

   ```bash
   sshfs <user>@conduit.hpc.uni-saarland.de:~/ ~/remote_cluster
   fusermount3 -u ~/remote_cluster  # Unmount when done
   ```

2. **Create `params.txt`:** One CLI argument set per line

3. **Submit jobs:**

   ```bash
   condor_submit sweep.submit
   # Or with custom parameters:
   condor_submit COMMIT_HASH=abc123 RUN_DIR=./results PARAMS_FILE=params.txt sweep.submit
   ```

4. **Monitor jobs:**

   ```bash
   condor_q <cluster-id>              # Check all jobs in cluster
   condor_q -nobatch <cluster-id>.<process-id>  # Specific job
   condor_q -batch-name <RUN_TAG>    # Filter by tag
   ```

5. **View job output:**
   ```bash
   condor_tail -f <cluster-id>.<process-id>
   ```

### Analyzing on Cluster

Marimo notebooks can run on cluster execution nodes:

1. Copy `marimo.submit`, `run-marimo.sh`, `start-notebook-server.sh` to cluster
2. Run `start-notebook-server.sh`
3. Forward port locally: `ssh -N -L 3718:127.0.0.1:3718 <cluster>`
4. Open `localhost:3718` in browser (password: `password`)

### Copying Data Back

```bash
rsync -avz <user>@conduit.hpc.uni-saarland.de:~/results/ ./local_results/
```

## Common Tasks

### Adding a New Simulation Parameter

1. Add field to `RunParams` or `ModelParams` in `include/sim_types.h`
2. Add default value in `include/parameters.h`
3. Add CLI argument in `src/cli.c`
4. Update `src/config.c` to wire the parameter
5. Update HDF5 writing in `src/hdf5_file.c`
6. Update Python `analysis/red_patterns.py` dataclasses

### Adding a New CUDA Kernel

1. Declare kernel in `include/cuda_kernel.cuh`
2. Implement in `src/cuda_kernel.cu`
3. Call from `src/simulations.cu`
4. Ensure `CUDA_SEPARABLE_COMPILATION` is ON (already set in CMakeLists.txt)

### Modifying Analysis Scripts

1. Update `analysis/red_patterns.py` for data structure changes
2. Run `mypy analysis/` for type checking (optional)
3. Test with: `python -c "from analysis.red_patterns import RunData; ..."`

## Troubleshooting

### CUDA Architecture Issues

**Error:** `no kernel image is available for execution on the device`

**Solution:** Rebuild with correct architecture:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native  # Auto-detect
# Or specify explicitly:
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86      # RTX 30-series
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80      # A100
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90      # H100
```

### HDF5 Linking Issues

**Error:** `Could not find HDF5`

**Solution:**

```bash
# Using mamba environment:
mamba activate cuda-dev

# Or specify manually:
cmake .. -DHDF5_ROOT=/path/to/hdf5
```

### CUDA Compiler Not Found

**Error:** `Could not find nvcc`

**Solution:**

```bash
# Using mamba:
mamba activate cuda-dev

# Or specify manually:
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Clangd Errors with CUDA

The `.clangd` config removes problematic CUDA flags. If issues persist:

1. Ensure `compile_commands.json` exists in `build/`
2. Restart clangd/LSP server
3. Check `.clangd` has correct GPU arch for your system

### Out of Memory

**Error:** CUDA out of memory

**Solutions:**

- Reduce grid size (`N` parameter)
- Reduce time steps between saves (`NO`)
- Use smaller `DT` with fewer total steps

### Build Directory Confusion

If CMake behaves strangely:

```bash
rm -rf build/
mkdir build && cd build
cmake ..
```

## Docker

Build container:

```bash
./build-docker.sh  # Builds and pushes to DockerHub
# Or manually:
docker build -t <name>:<tag> .
```

Use unique tags (e.g., git hash) to force HTCondor to pull new versions:

```bash
docker tag <name>:latest <name>:$(git rev-parse --short HEAD)
docker push <name>:$(git rev-parse --short HEAD)
```

## Important Files

| File                | Purpose                            |
| ------------------- | ---------------------------------- |
| `CMakeLists.txt`    | Build configuration                |
| `CMakePresets.json` | CMake presets (release, dev-debug) |
| `Makefile`          | Quick run targets                  |
| `.clang-format`     | Code formatting rules              |
| `.clangd`           | clangd/LSP configuration           |
| `environment.yml`   | mamba/conda environment            |
| `params.txt`        | Example parameter file             |
| `sweep.submit`      | HTCondor submit file               |
| `run-sim.sh`        | Simulation wrapper script          |
