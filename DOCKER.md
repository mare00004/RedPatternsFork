# Docker

This repo can be built into a CUDA-enabled Docker image that contains the `red-patterns` executable.

Important files:

- `Dockerfile`: defines build + runtime image.
- `.dockerignore`: keeps the build context small and avoids including `build/` (which can confuse CMake caching).
- `build-docker.sh`: convenience script that builds (and in many setups, pushes) the image.

## Build A New Image

Pick a Docker Hub repository name and a unique tag, e.g. the current git hash, so HTCondor pulls the updated image instead of reusing a cached one:

```bash
TAG=$(git rev-parse --short HEAD)
IMAGE=<dockerhub-user-or-org>/<repo-name>
```

Build locally:

```bash
sudo docker build -t "$IMAGE:$TAG" .
```

Or use the helper script (see `build-docker.sh` for defaults/flags):

```bash
./build-docker.sh
```

## Push To Docker Hub

1. Login to Docker Hub (use a Personal Access Token instead of your password):

```bash
sudo docker login -u <user>
```

2. Push the tag you built:

```bash
sudo docker push "$IMAGE:$TAG"
```

## Pull And Run The Simulation

On the machine that should run the simulation:

```bash
sudo docker pull "$IMAGE:$TAG"
```

### GPU Prerequisites

To use the GPU inside the container you need NVIDIA drivers on the host and the NVIDIA Container Toolkit (so `docker run --gpus all ...` works).

### Run Interactively (Shell Inside The Container)

Mount a host directory for outputs (and optionally inputs), then open a shell:

```bash
OUT_DIR=$PWD/out
mkdir -p "$OUT_DIR"

sudo docker run --rm -it \
  --gpus all \
  -v "$OUT_DIR:/out" \
  "$IMAGE:$TAG" \
  bash
```

Inside the container, run the binary and write outputs to `/out`:

```bash
red-patterns --help

# Example (adjust flags/paths to your workflow)
red-patterns --use-convolution --out-dir /out
```

### Run Non-Interactively (One-Shot Command)

Run the simulation directly (no shell), still mounting an output directory:

```bash
OUT_DIR=$PWD/out
mkdir -p "$OUT_DIR"

sudo docker run --rm \
  --gpus all \
  -v "$OUT_DIR:/out" \
  "$IMAGE:$TAG" \
  red-patterns --use-convolution --out-dir=/out
```

If you have a parameter file on the host, mount it and pass the in-container path:

```bash
KERNEL=$PWD/kernel.h5
OUT_DIR=$PWD/out
mkdir -p "$OUT_DIR"

sudo docker run --rm \
  --gpus all \
  -v "$OUT_DIR:/out" \
  -v "$PARAMS:/kernel.h5:ro" \
  "$IMAGE:$TAG" \
  red-patterns --use-convolution --params /params.json --out-dir /out
```
