#include "cmath"
#include "cuda_kernel.cuh"
#include "gpu_state.cuh"
#include "parameters.h"
#include "sim_types.h"

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
        sum += phi[k * d_cfg.run.N + i];
    }

    psi[i] = sum;
}

__device__ __forceinline__ int clampCellIndex(int idx, int N) {
    return max(0, min(idx, N - 1));
}

__global__ void CuKernelTaylFace(
    const double *__restrict__ psi,
    double *__restrict__ I_face,
    double c1,
    double c3) {
    /*
     * Computes Taylor/local approximation directly on FVM faces.
     *
     * psi[i]      lives at cell center z_i = (i + 1/2) dz
     * I_face[f]  lives at face        z_f = f dz
     *
     * f = 0, ..., N
     *
     * For interior face f, the neighboring cell centers are:
     *
     *   f - 2 : z_f - 3/2 dz
     *   f - 1 : z_f - 1/2 dz
     *   f     : z_f + 1/2 dz
     *   f + 1 : z_f + 3/2 dz
     *
     * This version assumes c1 and c3 are the SAME coefficients you used
     * in the old cell-centered kernel.
     */

    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = d_cfg.run.N;

    if (f > N) {
        return;
    }

    const int im2 = clampCellIndex(f - 2, N);
    const int im1 = clampCellIndex(f - 1, N);
    const int ip0 = clampCellIndex(f, N);
    const int ip1 = clampCellIndex(f + 1, N);

    /*
     * Face-centered 4-point first derivative numerator:
     *
     *   psi[im2] - 27 psi[im1] + 27 psi[ip0] - psi[ip1]
     *
     * This is 24 dz psi'(z_f) + higher-order terms.
     *
     * The old cell-centered first derivative numerator was
     *
     *   -psi[i+2] + 8 psi[i+1] - 8 psi[i-1] + psi[i-2]
     *
     * which is 12 dz psi'(z_i) + higher-order terms.
     *
     * Therefore, if c1 is the old coefficient, multiply by 1/2.
     */
    const double first_num =
        psi[im2] - 27.0 * psi[im1] + 27.0 * psi[ip0] - psi[ip1];

    /*
     * Face-centered third derivative numerator:
     *
     *   psi[ip1] - 3 psi[ip0] + 3 psi[im1] - psi[im2]
     *
     * This is dz^3 psi'''(z_f) + higher-order terms.
     *
     * The old cell-centered third derivative numerator was
     *
     *   psi[i+2] - 2 psi[i+1] + 2 psi[i-1] - psi[i-2]
     *
     * which is 2 dz^3 psi'''(z_i) + higher-order terms.
     *
     * Therefore, if c3 is the old coefficient, multiply by 2.
     */
    const double third_num =
        psi[ip1] - 3.0 * psi[ip0] + 3.0 * psi[im1] - psi[im2];

    I_face[f] =
        0.5 * c1 * first_num +
        2.0 * c3 * third_num;
}
__global__ void CuKernelCellToFace(const double *__restrict__ I_cell, double *__restrict__ I_face) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = d_cfg.run.N;

    if (j > N) {
        return;
    }

    if (j == 0) {
        I_face[j] = I_cell[0];
    } else if (j == N) {
        I_face[j] = I_cell[N - 1];
    } else {
        I_face[j] = 0.5 * (I_cell[j - 1] + I_cell[j]);
    }
}

__global__ void CuKernelDownSample(double *IIntp, double *I, int subDiv) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = d_cfg.run.N;

    if (i >= N || subDiv <= 0) {
        return;
    }

    if ((subDiv % 2) == 0) {
        const int mid = i * subDiv + subDiv / 2;
        I[i] = 0.5 * (IIntp[mid - 1] + IIntp[mid]);
    } else {
        const int mid = i * subDiv + subDiv / 2;
        I[i] = IIntp[mid];
    }
}

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

__global__ void CuKernelSplineEvalCellCentered(
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

    // Fine cell center coordinate in coarse-cell units:
    // x = z_tilde / DZ - 1/2
    double x = (double(k) + 0.5) / double(subDiv) - 0.5;

    // Clamp to the available spline interval [0, N-1]
    if (x <= 0.0) {
        y_intp[k] = y[0];
        return;
    }

    if (x >= double(N - 1)) {
        y_intp[k] = y[N - 1];
        return;
    }

    int j = int(floor(x));
    double dx = x - double(j);

    // Safety: spline segment j uses y[j]..y[j+1], so j <= N-2
    if (j >= N - 1) {
        j = N - 2;
        dx = 1.0;
    }

    y_intp[k] = y[j] + (b[j] + (c[j] + d[j] * dx) * dx) * dx;
}

/* convolution kernel */
__global__ void CuKernelConv(double *psi, double *I, double *convKernel, int M, int kernelN, int subDiv) {
    // Optimized 1D convolution kernel using constant memory for the kernel
    // coefficients and shared memory tiling for input data. The convKernel
    // parameter is unused but kept for signature compatibility.
    extern __shared__ double s_psi[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int halo = (kernelN - 1) / 2;
    int blockStart = blockIdx.x * blockDim.x;
    // load central data
    if (gid < M) {
        s_psi[tid + halo] = psi[gid];
    } else {
        s_psi[tid + halo] = 0.0;
    }
    // load left halo
    if (tid < halo) {
        int leftIdx = blockStart + tid - halo;
        s_psi[tid] = (leftIdx >= 0 ? psi[leftIdx] : 0.0);
    }
    // load right halo
    if (tid >= blockDim.x - halo) {
        int offset = tid - (blockDim.x - halo);
        int rightIdx = blockStart + tid + halo;
        int sIdx = halo + blockDim.x + offset;
        s_psi[sIdx] = (rightIdx < M ? psi[rightIdx] : 0.0);
    }
    __syncthreads();
    // perform convolution if in bounds
    if (gid < M) {
        double acc = 0.0;
        for (int k = 0; k < kernelN; ++k) {
            acc += s_psi[tid + k] * convKernel[k];
        }
        // apply scale factor outside the loop for efficiency
        I[gid] = acc * d_cfg.run.DZ / subDiv;
    }
}

__device__ inline double pressure_cell_value(
    const double *__restrict__ percoll,
    const double *__restrict__ gradWing,
    int idx,
    int N) {
    // return (idx > wingL && idx < N - 1 - wingL)
    //            ? percoll[idx]
    //            : gradWing[idx];
    return percoll[idx];
}

__global__ void CuKernelComputeFluxFVM(
    const double *__restrict__ phi,
    double *__restrict__ J,
    const double *__restrict__ percoll,
    const double *__restrict__ R,
    const double *__restrict__ I_face,
    const double *__restrict__ gradWing,
    double *__restrict__ faceVelocity) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // face index: 0 ... N
    const int j = blockIdx.y * blockDim.y + threadIdx.y; // rho index

    const int N = d_cfg.run.N;
    const int NR = d_cfg.run.N; // replace by d_cfg.run.NR if you have separate NR

    if (i > N || j >= NR) {
        return;
    }

    const int fidx = j * (N + 1) + i;

    // No-flux boundaries.
    if (i == 0 || i == N) {
        J[fidx] = 0.0;
        if (faceVelocity != nullptr) {
            faceVelocity[fidx] = 0.0;
        }
        return;
    }

    const int left = i - 1;
    const int right = i;

    const double P_left =
        pressure_cell_value(percoll, gradWing, left, N);

    const double P_right =
        pressure_cell_value(percoll, gradWing, right, N);

    const double P_face = 0.5 * (P_left + P_right);

    // No smoothing of I_face: the canonical taylor branch uses I directly,
    // so any low-pass filter here biases the linear-stability spectrum of
    // the pattern-formation instability and changes the selected wavelength.
    const double I_face_value = I_face[i];

    const double rp = R[j] + P_face - P0;

    const double v_face =
        -d_cfg.model.alpha * rp - d_cfg.model.beta * I_face_value;

    if (faceVelocity != nullptr) {
        faceVelocity[fidx] = v_face;
    }

    const double phi_left = phi[j * N + left];
    const double phi_right = phi[j * N + right];

    const double phi_up = (v_face > 0.0) ? phi_left : phi_right;

    // FIX: Diffusion Flux
    // const double D_Z = 10e-10;
    const double D_Z = 0.0;
    const double diff_flux = -D_Z * (phi_right - phi_left) / d_cfg.run.DZ;

    J[fidx] = v_face * phi_up + diff_flux;
}

__global__ void CuKernelUpdatePhiFVM(
    double *__restrict__ phi,
    const double *__restrict__ J) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // cell index: 0 ... N-1
    const int j = blockIdx.y * blockDim.y + threadIdx.y; // rho index

    const int N = d_cfg.run.N;
    const int NR = d_cfg.run.N; // replace by d_cfg.run.NR if available

    if (i >= N || j >= NR) {
        return;
    }

    const int pidx = j * N + i;

    const double flux_in = J[j * (N + 1) + i];
    const double flux_out = J[j * (N + 1) + i + 1];

    phi[pidx] +=
        -d_cfg.run.DT / d_cfg.run.DZ * (flux_out - flux_in);

    /*
     * Be careful:
     * This positivity clamp breaks exact mass conservation.
     * It is okay as a debugging guard, but if it triggers often,
     * your time step is probably too large.
     */
    if (phi[pidx] < 0.0) {
        phi[pidx] = 0.0;
    }
}
