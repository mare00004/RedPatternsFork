#include "cmath"
#include "cuda_kernel.cuh"
#include "gpu_state.cuh"
#include "parameters.h"
#include "sim_types.h"

__global__ void CuKernelTayl(double *psi, double *I, double nu, double mu) {
    // get indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // compute convolution integral
    if ((i >= 8) && (i <= d_cfg.run.N - 1 - 8)) {
        I[i] =
            nu *
                (-psi[i + 2] + 8 * psi[i + 1] + psi[i - 2] - 8 * psi[i - 1]) /
                (12 * d_cfg.run.DZ) +
            mu *
                (psi[i + 2] - 2 * psi[i + 1] + 2 * psi[i - 1] - psi[i - 2]) /
                (2 * pow(d_cfg.run.DZ, 3));
    }

    __syncthreads();
}

/* phi density integration kernel */
__global__ void CuKernelInte(double *phi, double *psi) {
    // get indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_cfg.run.N) {
        return;
    }
    double sum = 0.0;

    // discrete sum integration
    for (int k = 0; k < d_cfg.run.N; k++) {
        sum += phi[k * d_cfg.run.N + i]; // TODO multiply by Delta rho?
    }

    psi[i] = sum;
}

// TODO replace 256 with N
__global__ void CuKernelSplineCoeffs(
    const double *__restrict__ y,
    double *__restrict__ b,
    double *__restrict__ c,
    double *__restrict__ d,
    const int N) {

    extern __shared__ double shared[]; // Layout [mu | ze]
    double *mu = shared;
    double *ze = shared + N;

    /**
     * This is a sequential implementation of the Thomas-Algorithm.
     * The only reason this isn't done on the CPU is to avoid copying the values from the GPU.
     * Only one thread executes the whole algorithm.
     **/
    if (threadIdx.x == 0) {
        mu[0] = 0.0;
        ze[0] = 0.0;

        // Forward sweep
        for (int i = 1; i <= N - 2; i++) {
            const double alpha = 3.0 * (y[i + 1] - y[i]) / 1.0 - 3.0 * (y[i] - y[i - 1]) / 1.0;
            const double denom = 4.0 - mu[i - 1];
            mu[i] = 1.0 / denom;
            ze[i] = (alpha - ze[i - 1]) / denom;
        }

        // Natural boundary conditions
        mu[N - 1] = 0.0;
        ze[N - 1] = 0.0;
        c[N - 1] = 0.0;

        // Backwards substitution
        for (int i = N - 2; i >= 0; i--) {
            c[i] = ze[i] - mu[i] * c[i + 1];
            b[i] = (y[i + 1] - y[i]) - (c[i + 1] + 2.0 * c[i]) / 3.0;
            d[i] = (c[i + 1] - c[i]) / 3.0;
        }

        // Natural boundary condition
        c[0] = 0.0;
    }
}

__global__ void CuKernelSplineEval(
    const double *__restrict__ y,
    const double *__restrict__ b,
    const double *__restrict__ c,
    const double *__restrict__ d,
    double *__restrict__ y_intp,
    int N,
    int M,
    int subDiv) {

    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= M) {
        return;
    }

    if (k == M - 1) {
        y_intp[k] = y[N - 1];
        return;
    }

    int j = k / subDiv;
    int r = k - j * subDiv;
    double dx = double(r) / double(subDiv);

    y_intp[k] = y[j] + (b[j] + (c[j] + d[j] * dx) * dx) * dx;
}

/* downsampling kernel */
__global__ void CuKernelSplineDownSample(double *IIntp, double *I, int subDiv) {
    // get indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_cfg.run.N) {
        return;
    }
    int j = i * subDiv;
    I[i] = IIntp[j];
}

/* convolution kernel */
__global__ void CuKernelConv(double *psi, double *I, double *convKernel, int M, int kernelN, int subDiv) {
    // get indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M) {
        return;
    }

    // compute convolution integral
    double sum = 0.0;
    int d = (kernelN - 1) / 2;

    for (int k = 0; k < kernelN; k++) {
        if ((i + (k - d) >= 0) && (i + (k - d) < M)) {
            sum += psi[i + (k - d)] * convKernel[k] * (d_cfg.run.fineDZ);
        }
    }

    I[i] = sum;
}

/*
 * Precompute degenerate diffusion power factors. For each i in [0, N-1],
 * psiPow0[i] = pow(1.0 - psi[i], mDeg) and psiPow1[i] = pow(psi[i], mDeg).
 * This kernel runs with a 1D grid over N elements.
 */
__global__ void CuKernelDegDiffPow(double *psi, double *psiPow0, double *psiPow1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_cfg.run.N) {
        double val = psi[i];
        psiPow0[i] = pow(1.0 - val, mDeg);
        psiPow1[i] = pow(val, mDeg);
    }
}

/* main time iteration */
__global__ void CuKernelIter(
    double *phi,
    double *J,
    double *dJ,
    double *percoll,
    double *R,
    double *I,
    double *psi,
    double *psiPow0,
    double *psiPow1,
    double t,
    double *gradWing) {

    int N = d_cfg.run.N;
    // get indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int gi = i + j * N; // global index

    double DZ = d_cfg.run.DZ;

    // compute physical flux
    double rpTerm;
    if ((i > wingL) && (i < N - 1 - wingL)) {

        rpTerm = R[j] + percoll[i] - P0;
    }
    if ((i <= wingL) || (i >= N - 1 - wingL)) {
        rpTerm = R[j] + gradWing[i] - P0;
    }

    J[gi] = (d_cfg.model.alpha * rpTerm + d_cfg.model.beta * I[i]) * phi[gi];

    // compute flux derivative
    if ((i >= 4) & (i <= N - 1 - 4)) {
        // physical flux first derivative
        dJ[gi] = (+0.5 / DZ * (J[gi + 1] - J[gi - 1]));
        // degenerate diffusion second derivative for phi (no precomputation needed)
        double ddPhi0_i = jDegDiffPhi0(gi);
        double ddPhi0_p1 = jDegDiffPhi0(gi + 1);
        double ddPhi0_m1 = jDegDiffPhi0(gi - 1);
        dJ[gi] -= (-2.0 * ddPhi0_i + (ddPhi0_p1 + ddPhi0_m1)) / (DZ * DZ) * d_cfg.model.gamma;
        // degenerate diffusion second derivative for psi using precomputed power arrays
        double deg0 = psiPow0[i] * fabs(phi[gi]);
        double deg0_p1 = psiPow0[i + 1] * fabs(phi[gi + 1]);
        double deg0_m1 = psiPow0[i - 1] * fabs(phi[gi - 1]);
        dJ[gi] -= (-2.0 * deg0 + (deg0_p1 + deg0_m1)) / (DZ * DZ) * d_cfg.model.delta;
        double deg1 = -psiPow1[i] * fabs(phi[gi]);
        double deg1_p1 = -psiPow1[i + 1] * fabs(phi[gi + 1]);
        double deg1_m1 = -psiPow1[i - 1] * fabs(phi[gi - 1]);
        dJ[gi] += (-2.0 * deg1 + (deg1_p1 + deg1_m1)) / (DZ * DZ) * d_cfg.model.kappa;
    }

    // compute euler step
    phi[gi] = phi[gi] + d_cfg.run.DT * dJ[gi];
}
