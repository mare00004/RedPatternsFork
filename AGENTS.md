# AGENTS.md

High-signal, repo-specific notes for OpenCode agents working on RedPatternsFork (CUDA + HDF5 simulation).

## Build (Source Of Truth: `CMakePresets.json`, `CMakeLists.txt`)

- Recommended toolchain is the conda env in `environment.yml` (CUDA 12.8, GCC 14, HDF5 1.14):
  `mamba env create -f environment.yml` then `mamba activate cuda-dev`.
- Preferred build uses presets (binary goes into `build/<preset>/red-patterns`):
  `cmake --preset release && cmake --build build/release`
  `cmake --preset dev-debug && cmake --build build/dev-debug`
- GPU arch gotcha:
  `CMakeLists.txt` defaults `CMAKE_CUDA_ARCHITECTURES` to `native`, but presets pin it to `86`.

## Run (Source Of Truth: `README.md`, `Makefile`, `src/cli.c`)

- You must pick a model mode: `--use-convolution` or `--use-taylor` (running without one is not the normal path).
- Quick validation run (no test framework): `make test`.
- `Makefile` currently calls `./build/red-patterns ...` (not `build/release/...`). If you use presets, run the built binary directly from `build/<preset>/red-patterns` or adjust the Makefile path.
- Primary output artifact is `<out-dir>/run.h5`.

## Formatting / IDE

- Format C/CUDA with repo `.clang-format`:
  `clang-format -i src/*.{c,cu} include/*.{h,cuh}`
- `.clangd` removes some CUDA flags for clangd and hardcodes `--cuda-gpu-arch=sm_120` for `.cu` files; this is for editor diagnostics and does not need to match the build preset arch.

## Where Things Live (Source Of Truth: `CMakeLists.txt`)

- Main executable target `red-patterns` is built from: `src/main.cu`, `src/{cli.c,config.c,hdf5_file.c}`, `src/{gpu_state.cu,simulations.cu,cuda_kernel.cu}`.
- CLI/config wiring is in `src/cli.c` and `src/config.c`; changes to runtime parameters usually also require updates in HDF5 writing (`src/hdf5_file.c`) and analysis loaders (`analysis/`).

## Cluster (UdS HTCondor)

- Cluster job submit + wrappers live in `cluster/` (see `cluster/README.md`).
- `cluster/sweep.submit` is parameterized by `COMMIT_HASH`, `RUN_DIR`, `PARAMS_FILE`, `RUN_TAG`:
  `condor_submit COMMIT_HASH=... RUN_DIR=... PARAMS_FILE=... RUN_TAG=... sweep.submit`
