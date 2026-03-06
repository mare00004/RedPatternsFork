#include "cli.h"
#include "config.h"
#include "cuda_utils.cuh"
#include "gpu_state.cuh"
#include "sim_types.h"
#include "simulations.cuh"
#include <stdio.h>

int main(int argc, char *argv[]) {
    SimConfig cfg;

    printf("Setting default parameters...\n");
    setDefaults(&cfg);

    printf("Parsing CLI...\n");
    if (parseArguments(argc, argv, &cfg)) {
        return EXIT_FAILURE;
    }

    printf("Validating parameters...\n");
    if (deriveAndValidateOrDie(&cfg)) {
        return EXIT_FAILURE;
    }

    printConfig(&cfg);

    // detect cuda device
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0));
    int cudaDevice;
    checkCuda(cudaChooseDevice(&cudaDevice, &prop));
    printf("\nDevice Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    cudaError_t err = cudaMemcpyToSymbol(d_cfg, &cfg, sizeof(SimConfig));
    if (err != cudaSuccess) {
        fprintf(stderr, "CRITICAL: Failed to copy config to GPU: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    printf("Starting Simulation...\n");
    runSim(cfg);
    return EXIT_SUCCESS;
}
