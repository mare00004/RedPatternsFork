#include "cli.cuh"
#include "config.h"
#include "cuda_utils.cuh"
#include "parameters.cuh"
#include "simulations.cuh"
#include <stdio.h>

// main function
int main(int argc, char *argv[]) {
    SimConfig cfg;
    setDefaults(&cfg);
    if (parseArguments(argc, argv, &cfg)) {
        return EXIT_FAILURE;
    }
    printConfig(&cfg);
    // Overwrite constants (like in old readParameters)
    U = cfg.model.U;
    PSI = cfg.model.PSI;
    IT = cfg.run.DT;
    T = cfg.run.T;
    NO = cfg.run.NO;
    h_gamma = cfg.model.gamma;
    h_delta = cfg.model.delta;
    h_kappa = cfg.model.kappa;
    NT = ceil(T / IT);
    h_beta *= U;

    // detect cuda device
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0));
    int cudaDevice;
    checkCuda(cudaChooseDevice(&cudaDevice, &prop));
    printf("\nDevice Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    // run simulation
    runSim(&cfg);
    return EXIT_SUCCESS;
}
