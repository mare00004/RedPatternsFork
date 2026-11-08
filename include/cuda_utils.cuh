#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

/* check cuda device */
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Runtime Error: ") + cudaGetErrorString(result));
    }
    return result;
}

#endif
