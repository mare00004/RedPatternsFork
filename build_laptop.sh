gcc -Ithird_party/argtable3 -c third_party/argtable3/argtable3.c -o build/argtable.o
h5cc -c -O2 -Iinclude -I/home/max/Uni/HDF5/build/HDF_Group/HDF5/2.0.1/include src/hdf5_file.c -o build/hdf5_file.o
/usr/local/cuda-12.0/bin/nvcc -O2 -arch=sm_86 -Iinclude -Ithird_party/argtable3 -I/home/max/Uni/HDF5/build/HDF_Group/HDF5/2.0.1/include -c src/main.cu -o build/simulation.o
/usr/local/cuda-12.0/bin/nvcc -O2 -arch=sm_86 \
  -Iinclude -Ithird_party/argtable3 \
  -I/home/max/Uni/HDF5/build/HDF_Group/HDF5/2.0.1/include \
  build/argtable.o build/hdf5_file.o build/simulation.o -o build/red-patterns \
  -L/home/max/Uni/HDF5/build/HDF_Group/HDF5/2.0.1/lib -lhdf5 -lhdf5_hl \
  -Xlinker -rpath,/home/max/Uni/HDF5/build/HDF_Group/HDF5/2.0.1/lib \
  -lz -ldl -lpthread -lm

