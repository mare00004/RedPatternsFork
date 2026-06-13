# syntax=docker/dockerfile:1.7

# Choose base image with specific CUDA and Ubuntu version
# Not every combination of CUDA_VER and Ubuntu exists!
ARG CUDA_VER=12.8.1
ARG HDF5_VERSION=1.14.5

# =========================
# Build stage
# =========================
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG HDF5_VERSION

# SHA256 for https://support.hdfgroup.org/releases/hdf5/v1_14/v1_14_5/downloads/hdf5-1.14.5.tar.gz
ARG HDF5_TGZ_SHA256=ec2e13c52e60f9a01491bb3158cb3778c985697131fc6a342262d32a26e58e44

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get -o Acquire::Check-Date=false -o Acquire::Check-Valid-Until=false update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    ca-certificates \
    zlib1g-dev \
    git \
  && rm -rf /var/lib/apt/lists/*

ENV HDF5_ROOT=/opt/hdf5

# Build HDF5 from source to match Ubuntu 24.04 (instead of using the Ubuntu 22.04 binary bundle)
RUN mkdir -p /tmp/hdf5-src \
 && cd /tmp/hdf5-src \
 && wget -q \
      "https://support.hdfgroup.org/releases/hdf5/v1_14/v1_14_5/downloads/hdf5-${HDF5_VERSION}.tar.gz" \
      -O "hdf5-${HDF5_VERSION}.tar.gz" \
 && echo "${HDF5_TGZ_SHA256}  hdf5-${HDF5_VERSION}.tar.gz" | sha256sum -c - \
 && tar -xzf "hdf5-${HDF5_VERSION}.tar.gz" \
 && cd "hdf5-${HDF5_VERSION}" \
 && ./configure --prefix="${HDF5_ROOT}" --enable-shared --disable-static \
 && make -j"$(nproc)" \
 && make install \
 && rm -rf /tmp/hdf5-src

ENV PATH=${HDF5_ROOT}/bin:$PATH
ENV LD_LIBRARY_PATH=${HDF5_ROOT}/lib:$LD_LIBRARY_PATH

# ------------------------------------------------------------------
# Build your project with CMake
# ------------------------------------------------------------------
WORKDIR /workspace

# Copy only what's needed to compile (better Docker cache hit rate)
COPY CMakeLists.txt CMakePresets.json ./
COPY cmake ./cmake
COPY include ./include
COPY src ./src
COPY third_party ./third_party

ARG CUDA_ARCH=80;86

RUN --mount=type=cache,target=/workspace/build \
    cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DHDF5_ROOT=${HDF5_ROOT} \
      "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}" \
  && cmake --build build --config Release -j"$(nproc)" \
  && cmake --install build --prefix /opt/red-patterns \
  && test -x /opt/red-patterns/bin/red-patterns

# =========================
# Runtime stage
# =========================
FROM nvidia/cuda:${CUDA_VER}-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

# Minimal Python runtime for sweep orchestration inside the container.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get -o Acquire::Check-Date=false -o Acquire::Check-Valid-Until=false update \
 && apt-get install -y --no-install-recommends \
    python3 \
    python3-h5py \
    python3-numpy \
 && rm -rf /var/lib/apt/lists/*

# Copy the executable
COPY --from=build /opt/red-patterns/bin/red-patterns /bin/red-patterns

# Copy HDF5 install tree from the build stage into the runtime image
COPY --from=build /opt/hdf5 /opt/hdf5

# Copy the Python sweep runtime used by HTCondor jobs.
COPY analysis/red_patterns /opt/red-patterns/analysis/red_patterns
COPY sweep /opt/red-patterns/sweep

ENV LD_LIBRARY_PATH=/opt/hdf5/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/opt/red-patterns/analysis:$PYTHONPATH

CMD ["/bin/bash"]
