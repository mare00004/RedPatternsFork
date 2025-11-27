#!/usr/bin/env bash

# Use git hash or first cli argument as tag
TAG="${1:-$(git rev-parse --short HEAD)}"
IMAGE="mare00004/cuda-hdf5-dev:${TAG}"

sudo docker build -t "${IMAGE}" .
sudo docker push "${IMAGE}"

echo "${TAG}"
