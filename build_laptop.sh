/usr/local/cuda-12.0/bin/nvcc -Xptxas -O2 -arch=compute_86 -L/usr/local/cuda-12.0/lib64 -Iinclude src/main.cu -o build/red_patterns
