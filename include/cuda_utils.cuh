#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H
#include <cassert>
#include <stdio.h>

/* check cuda device */
inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

#endif
