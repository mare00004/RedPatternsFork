#include "parameters.cuh"
#include <cmath>
#include <string>

/* taking arguments */
void readParameters(int argc, char *argv[]) {
    int argIdx = 1;
    // U
    if (argc > argIdx)
        U = std::stod(argv[argIdx]);
    argIdx++;
    // PSI
    if (argc > argIdx)
        PSI = std::stod(argv[argIdx]);
    argIdx++;
    // IT
    if (argc > argIdx)
        IT = std::stod(argv[argIdx]);
    argIdx++;
    // T
    if (argc > argIdx)
        T = std::stod(argv[argIdx]);
    argIdx++;
    // NO
    if (argc > argIdx)
        NO = std::stod(argv[argIdx]);
    argIdx++;
    // gamma
    if (argc > argIdx)
        h_gamma = std::stod(argv[argIdx]);
    argIdx++;
    // delta
    if (argc > argIdx)
        h_delta = std::stod(argv[argIdx]);
    argIdx++;
    // kappa
    if (argc > argIdx)
        h_kappa = std::stod(argv[argIdx]);
    argIdx++;
    // re-evalutate parameters
    NT = ceil(T / IT);
}
