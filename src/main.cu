#include "cli_arguments_parser.cuh"
#include "cuda_utils.cuh"
#include "simulations.cuh"
#include <stdio.h>

// main function
int main(int argc, char *argv[]) {
    // detect cuda device
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0));
    int cudaDevice;
    checkCuda(cudaChooseDevice(&cudaDevice, &prop));
    printf("\nDevice Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);
    // read parameter arguments from command line
    readParameters(argc, argv);
    // run simulation
    runSim();
    return 0;
}
