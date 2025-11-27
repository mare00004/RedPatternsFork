# Choose base image with specific CUDA and Ubuntu version
# Not every combination of CUDA_VER and UBUNTU_VER exists!
ARG CUDA_VER=12.8.1
ARG UBUNTU_VER=24.04
ARG HDF5_VERSION=1.14.5

# =========================
# Build stage
# =========================
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu${UBUNTU_VER} AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG HDF5_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV HDF5_BASE=/opt/hdf5

RUN mkdir -p ${HDF5_BASE} \
 && cd ${HDF5_BASE} \
 && wget -q \
      "https://sourceforge.net/projects/hdf5.mirror/files/hdf5_${HDF5_VERSION}/hdf5-${HDF5_VERSION}-ubuntu-2204_gcc.tar.gz/download" \
      -O hdf5-${HDF5_VERSION}-ubuntu-2204_gcc.tar.gz \
 && gunzip hdf5-${HDF5_VERSION}-ubuntu-2204_gcc.tar.gz \
 && tar xvf hdf5-${HDF5_VERSION}-ubuntu-2204_gcc.tar \
 && cd hdf5 \
 && gunzip HDF5-${HDF5_VERSION}-Linux.tar.gz \
 && tar xvf HDF5-${HDF5_VERSION}-Linux.tar

# Final HDF5 prefix inside that binary package:
#   /opt/hdf5/hdf5/HDF5-1.14.5-Linux/HDF_Group/HDF5/1.14.5
ENV HDF5_ROOT=${HDF5_BASE}/hdf5/HDF5-${HDF5_VERSION}-Linux/HDF_Group/HDF5/${HDF5_VERSION}
ENV PATH=${HDF5_ROOT}/bin:$PATH
ENV LD_LIBRARY_PATH=${HDF5_ROOT}/lib:$LD_LIBRARY_PATH

# ------------------------------------------------------------------
# Build your project with CMake
# ------------------------------------------------------------------
WORKDIR /workspace

# Dockerfile is in the repo, so just copy the source in
COPY . .

ARG CUDA_ARCH=80;86

RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DHDF5_ROOT=${HDF5_ROOT} \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
 && cmake --build build --config Release -j"$(nproc)"

# =========================
# Runtime stage
# =========================
FROM nvidia/cuda:${CUDA_VER}-runtime-ubuntu${UBUNTU_VER}

ARG HDF5_VERSION=1.14.5

# Copy the executable
COPY --from=build /workspace/build/red-patterns /bin/red-patterns

# Copy HDF5 shared libs from the build stage into the runtime image
COPY --from=build \
  /opt/hdf5/hdf5/HDF5-${HDF5_VERSION}-Linux/HDF_Group/HDF5/${HDF5_VERSION}/lib/*.so* \
  /lib/

ENV LD_LIBRARY_PATH=/lib:$LD_LIBRARY_PATH

CMD ["/bin/bash"]

