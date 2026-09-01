#include "gpu_state.cuh"
#include "parameters.h"
#include "sim_types.h"

// Percoll density gradient
#define gradL 0.06                            // [m] tube length
#define zShift ((d_cfg.run.sysL - gradL) / 2) // gradient spatial center

/**********
 * LINEAR *
 **********/

// constexpr double P = 18.0;
// constexpr double P = 30.1;
constexpr double offset = 0.002;
// FIX: that shouldn't be hardcoded
constexpr double R_min = -30.0;
constexpr double R_max = 30.0;

__device__ __forceinline__ double p_func(double x) {
    // FIX: No magic numbers
    const double DR = 30.0 / ((double)d_cfg.run.N);
    const double L = d_cfg.run.L;
    const double DZ = d_cfg.run.DZ;
    const double P = (DR / DZ) * L;
    return (P / L) * (x - (d_cfg.run.L / 2.0));
}

__device__ __forceinline__ double l_func(double x) {
    // FIX: No magic numbers
    const double DR = 30.0 / ((double)d_cfg.run.N);
    const double L = d_cfg.run.L;
    const double DZ = d_cfg.run.DZ;
    const double P = (DR / DZ) * L;
    const double wingL = d_cfg.run.wingL;
    const double t = offset - wingL;
    const double a = ((P / 2.0) + R_min - ((P / L) * t)) / (t * t);
    const double dx = x - wingL;
    return a * (dx * dx) + (P / L) * dx - (P / 2.0);
}

__device__ __forceinline__ double piecewise_func(double x) {
    const double wingL = d_cfg.run.wingL;
    const double sysL = d_cfg.run.sysL;

    if (x <= offset) {
        return R_min;
    } else if ((offset < x) && (x <= wingL)) {
        return l_func(x);
    } else if ((wingL < x) && (x < (sysL - wingL))) {
        return p_func(x - wingL);
    } else if ((sysL - wingL <= x) && (x < (sysL - offset))) {
        return -l_func(sysL - x);
    } else {
        return R_max;
    }
}

__global__ void CuKernelGradLinear(double *p) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_cfg.run.N) {
        return;
    }

    const double dz = d_cfg.run.DZ;
    const double z = ((double)idx + 0.5) * dz;
    // Store the physical Percoll density p = P0 + Q. The flipped z axis runs
    // upward, so the physical density decreases with z.
    p[idx] = P0 - piecewise_func(z);
}

/*****************
 * LINEAR (FULL) *
 *****************/

__global__ void CuKernelGradLinearFull(double *p) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_cfg.run.N) {
        return;
    }

    const double dz = d_cfg.run.DZ;
    const double z = ((double)idx + 0.5) * dz;
    p[idx] = P0 - p_func(z - d_cfg.run.wingL);
}

/*********
 * ZERO   *
 **********/

__global__ void CuKernelGradZero(double *p) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_cfg.run.N) {
        return;
    }

    p[idx] = P0;
}

/*********************************************************
 * SIGMOID                                               *
 *  - Equation (3) from supplementary material of paper. *
 *********************************************************/

/* sigmoid parameters */
#define delta_1 3.1773e-4
#define z_0 (d_cfg.run.sysL / 2.0)
#define lambda 0.0338
#define mu_1 1.1012e-3
#define mu_2 0.6
#define delta_2 1.5205

__device__ __forceinline__ double sigmoid_value_at_index(int i, int N, double t) {
    // double z = d_cfg.run.DZ * double(i);
    double z = (d_cfg.run.sysL / ((double)N - 1.0)) * double(i);
    double mu = mu_1 * t + mu_2;

    double chi = (z - z_0) / lambda;
    double abs_chi = fabs(chi);

    /*
     * In the experiment the tube length was exactely 6cm. That means the measured lambda value was large enough to ensure that abs_chi < 1.
     * In the simulation however the system length was slightly increased to around 6.8cm, which means that lambda is no longer large enough
     * to ensure that abs_chi is between 0 and 1, which would leed to NaN values in the denominator.
     * SOLUTION: We clamp abs_chi to be smaller than 1
     */
    if (abs_chi >= 1.0) {
        abs_chi = 1.0 - 1e-9;
    }

    double denom = pow(1.0 - pow(abs_chi, mu), 1.0 / mu);
    return delta_1 * pow(t, delta_2) * (chi / denom);
}

__global__ void CuKernelGradSigmoid(double *p, double t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N = d_cfg.run.N;
    if (i >= N) {
        return;
    }

    p[i] = sigmoid_value_at_index(i, N, t);

    const double dz = d_cfg.run.sysL / ((double)N - 1.0);
    int wingL = int((d_cfg.run.wingL / dz) + 0.5);
    if (wingL < 1) {
        wingL = 1;
    }
    if (wingL > N - 2) {
        wingL = N - 2;
    }

    const int x1_idx = (wingL > 12) ? 12 : wingL - 1;
    const double x1 = double(x1_idx);
    const double x2 = double(wingL);
    const double r3 = sigmoid_value_at_index(wingL, N, t) - sigmoid_value_at_index(wingL - 1, N, t);
    const double r2 = sigmoid_value_at_index(wingL, N, t);
    double r1 = r2 - 50.0;
    const double x1_value = sigmoid_value_at_index(x1_idx, N, t);
    if (x1_value < r1) {
        r1 = x1_value;
    }

    double a, b, c; // parameters of parabola
    a = (r1 - r2 + r3 * (x2 - x1)) / ((x1 - x2) * (x1 - x2));
    b = r3 - 2 * a * x2;
    c = r2 - r3 * x2 + x2 * x2 * a;

    if (i <= wingL) {
        p[i] = a * i * i + b * i + c;
    }
    if (i >= N - 1 - wingL) {
        const double mirrored = double(N - 1 - i);
        p[i] = -(a * mirrored * mirrored + b * mirrored + c);
    }

    // Convert the profile expressed in the old code convention into the true
    // physical Percoll density p = P0 + Q.
    p[i] = P0 - p[i];
}
