#include "gpu_state.cuh"
#include "parameters.h"
#include "sim_types.h"

// Percoll density gradient
#define gradL 0.06                            // [m] tube length
#define zShift ((d_cfg.run.sysL - gradL) / 2) // gradient spatial center

/**********
 * LINEAR *
 **********/

/* Linear Percoll gradient kernel */
#define PL 8.0 // Spread of gradient (in units of density)
__global__ void CuKernelGradLinear(double *percoll, double t) {
    // get indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double x = d_cfg.run.DZ * double(i);
    percoll[i] = (x - zShift - gradL / 2) / (gradL / 2) * PL / 2;
}

__global__ void CuKernelWingLinear(double *percoll, double *gradWing, double t) {
    // get indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N = d_cfg.run.N;
    // compute gradient wing
    double r1, r2, r3;
    double x1, x2;
    r3 = (percoll[wingL] - percoll[wingL - 1]);
    r2 = percoll[wingL];
    r1 = r2 - 10;
    x1 = 20;
    x2 = wingL;
    if (percoll[int(x1)] < r1)
        r1 = percoll[int(x1)];

    double a, b, c; // parameters of parabola
    a = (r1 - r2 + r3 * (x2 - x1)) / ((x1 - x2) * (x1 - x2));
    b = r3 - 2 * a * x2;
    c = r2 - r3 * x2 + x2 * x2 * a;
    gradWing[i] = 0.0;
    if (i <= wingL)
        gradWing[i] = a * i * i + b * i + c;
    if (i >= N - 1 - wingL)
        gradWing[i] = -(a * (N - 1 - i) * (N - 1 - i) + b * (N - 1 - i) + c);
    if (i >= N - 1 - 13)
        gradWing[i] = gradWing[N - 1 - 13];
    if (i <= 13)
        gradWing[i] = gradWing[13];
}

/*********************************************************
 * SIGMOID                                               *
 *  - Equation (3) from supplementary material of paper. *
 *********************************************************/

/* sigmoid parameters */
#define b2 3.1773e-4
#define b3 (0.06 / 2)
#define b4 0.0338
#define b5 1.1012e-3
#define b6 0.6
#define b7 1.5205
/* Sigmoidal Percoll gradient kernel */
__global__ void CuKernelGradSigmoid(double *percoll, double t) {
    // get indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N = d_cfg.run.N;
    double x = d_cfg.run.DZ * double(i);
    if (i > double(N - 1) / 2) {
        double debug = +b2 * pow(t, b7) * (x - zShift - b3) / b4 / pow(1 - pow((x - zShift - b3) / b4, b5 * t + b6), 1 / (b5 * t + b6));
        percoll[i] = debug;
    }
    if (i < double(N - 1) / 2)
        percoll[i] = -b2 * pow(t, b7) * (-x + zShift + b3) / b4 / pow(1 - pow((-x + zShift + b3) / b4, b5 * t + b6), 1 / (b5 * t + b6));
}

// TEST:
/* sigmoid parameters */
// #define delta_1 3.1773e-4
// #define z_0 (d_cfg.run.sysL / 2.0)
// #define lambda 0.0338
// #define mu_1 1.1012e-3
// #define mu_2 0.6
// #define delta_2 1.5205
//
// __global__ void CuKernelGradSigmoid(double *percoll, double t) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int N = d_cfg.run.N;
//
//     // double z = d_cfg.run.DZ * double(i);
//     double z = (d_cfg.run.sysL / ((double)N - 1.0)) * double(i);
//     double mu = mu_1 * t + mu_2;
//
//     double chi = (z - z_0) / lambda;
//     double abs_chi = fabs(chi);
//
//     /*
//      * In the experiment the tube length was exactely 6cm. That means the measured lambda value was large enough to ensure that abs_chi < 1.
//      * In the simulation however the system length was slightly increased to around 6.8cm, which means that lambda is no longer large enough
//      * to ensure that abs_chi is between 0 and 1, which would leed to NaN values in the denominator.
//      * SOLUTION: We clamp abs_chi to be smaller than 1
//      */
//     if (abs_chi >= 2.0) {
//         abs_chi = 1.0 - 1e-9;
//         // percoll[i] = 0.0;
//         // return;
//     }
//
//     double denom = pow(1.0 - pow(abs_chi, mu), 1.0 / mu);
//     percoll[i] = delta_1 * pow(t, delta_2) * (chi / denom);
// }

__global__ void CuKernelWingSigmoid(double *percoll, double *gradWing, double t) {
    // get indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N = d_cfg.run.N;
    // compute gradient wing
    double r1, r2, r3;
    double x1, x2;
    r3 = (percoll[wingL] - percoll[wingL - 1]);
    r2 = percoll[wingL];
    r1 = r2 - 50;
    x1 = 12;
    x2 = wingL;
    if (percoll[int(x1)] < r1)
        r1 = percoll[int(x1)];

    double a, b, c; // parameters of parabola
    a = (r1 - r2 + r3 * (x2 - x1)) / ((x1 - x2) * (x1 - x2));
    b = r3 - 2 * a * x2;
    c = r2 - r3 * x2 + x2 * x2 * a;

    gradWing[i] = 0.0;
    if (i <= wingL)
        gradWing[i] = a * i * i + b * i + c;
    if (i >= N - 1 - wingL)
        gradWing[i] = -(a * (N - 1 - i) * (N - 1 - i) + b * (N - 1 - i) + c);
}
