rm -rf build/
mkdir build
cd build
/usr/bin/cmake .. -DCMAKE_C_COMPILER=clang-14 \
      -DCMAKE_CXX_COMPILER=clang++-14 \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-14 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.0/bin/nvcc \
    -DHDF5_ROOT=/home/max/Uni/HDF5/build/HDF_Group/HDF5/2.0.1 \
    -DCMAKE_CUDA_ARCHITECTURE=86

/usr/bin/cmake --build .
