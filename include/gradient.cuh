#include "parameters.cuh"

// Percoll density gradient
#define gradL 0.06                  // [m] tube length
#define zShift ((sysL - gradL) / 2) // gradient spatial center

/**********
 * LINEAR *
 **********/

/* Linear Percoll gradient kernel */
#define PL 8.0
__global__ void CuKernelGradLinear(double *percoll, double t) {
    // get indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double x = IZ * double(i);
    percoll[i] = (x - zShift - gradL / 2) / (gradL / 2) * PL / 2;
}
__global__ void CuKernelWingLinear(double *percoll, double *gradWing, double t) {
    // get indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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
#define delta_1 3.1773e-4
#define z_0 (sysL / 2.0)
#define lambda 0.0338
#define mu_1 1.1012e-3
#define mu_2 0.6
#define delta_2 1.5205

/* Sigmoidal Percoll gradient kernel */
__global__ void CuKernelGradSigmoid(double *percoll, double t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double z = IZ * double(i);
    double chi = (z - z_0) / lambda;
    double mu = mu_1 * t + mu_2;
    if (i > double(N - 1) / 2) {
        // Equivalent to chi > 0
        percoll[i] = delta_1 * pow(t, delta_2) * (chi / pow(1 - pow(chi, mu), 1 / mu));
    } else if (i < double(N - 1) / 2) {
        // Equivalent to chi < 0
        percoll[i] = delta_1 * pow(t, delta_2) * (chi / pow(1 + pow(chi, mu), 1 / mu));
    }
}
__global__ void CuKernelWingSigmoid(double *percoll, double *gradWing, double t) {
    // get indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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
