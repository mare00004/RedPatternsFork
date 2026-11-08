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

    try {
        // detect cuda device
        cudaDeviceProp prop;
        checkCuda(cudaGetDeviceProperties(&prop, 0));
        int cudaDevice;
        checkCuda(cudaChooseDevice(&cudaDevice, &prop));
        checkCuda(cudaSetDevice(cudaDevice));
        printf("\nDevice Name: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

        printf("Starting Simulation...\n");
        return runSim(cfg) == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception &error) {
        fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }
}
