/usr/local/cuda-12.0/bin/nvcc -Xptxas -O2 -arch=compute_86 -L/usr/local/cuda-12.0/lib64 -Iinclude -Ithird_party/argtable3 third_party/argtable3/argtable3.c src/main.cu -o build/red-patterns
