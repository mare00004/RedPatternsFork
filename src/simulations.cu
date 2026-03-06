#include "cuda_kernel.cuh"
#include "cuda_utils.cuh"
#include "gradient.cuh"
#include "hdf5_file.h"
#include "parameters.h"
#include "sim_types.h"
#include "simulations.cuh"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <vector>

/* kernel function */

inline double fLJ(double r, double sigma, double U) {
    return (4 * U * (12 * pow(sigma, 12) / pow(r, 13) - 6 * pow(sigma, 6) / pow(r, 7)));
}

inline double g(double r, double d, double sigmaC) {
    return 4e7 * exp(-pow(r - d, 2) / (2 * pow(sigmaC, 2)));
}

void genConvKernel(double *intKernel, double DZ, double U) {
    // TODO figure out what does do and why i need the outside of the convolution version
    int kernelN = 31;
    double subDiv = 256.0;

    // compute effective potential
    double kernelL = (double(kernelN) - 1) * DZ / subDiv;
    double kernelDZ = kernelL / double(kernelN - 1);
    double subRes = 10000;
    double fineRes = subRes * (double(kernelN + 1) / 2);
    double force;
    double fineR;
    double gpdf;
    std::vector<double> kernelFine(int(fineRes), 0.0);
    double fineDR = kernelDZ / subRes; // only take positive values
    double sigma = 5.6e-6;
    double sigmaC = 0.5e-6;
    double eqDist =
        6.58546720106423709125472581993321341542468871921300888061523437500000000000000000e-06;

    // use central interval positions
    double sum = 0;
    kernelFine[0] = 0; // avoid divergence of force term at zero

    for (int i = 1; i < fineRes; i++) {

        fineR = double(i * fineDR);
        force = fLJ(fineR, sigma, U);
        gpdf = g(fineR, eqDist, sigmaC);

        if (fineR < 1e-8) { // make up for numerical error near divergence
            gpdf = 0.0;
        }
        kernelFine[i] = sum; // compute integral
        sum = sum + fineDR * force * gpdf;
    }

    // integration constant
    for (int i = 0; i < fineRes; i++)
        kernelFine[i] = kernelFine[int(fineRes) - 1] - kernelFine[i];
    // sampling of kernel
    intKernel[(kernelN - 1) / 2] = 0;
    double kernelZ;
    for (int i = (kernelN + 1) / 2; i < kernelN; i++) {
        kernelZ = double(i * kernelDZ) - kernelL / 2;
        intKernel[i] =
            kernelZ *
            kernelFine[int((i + 1 - double(kernelN + 1) / 2) * subRes)];
        intKernel[kernelN - 1 - i] = -intKernel[i];
    }
    printf("kernel length = %.32e m\n", kernelL);
}

/* initial values for phi */
// TODO wrong || used in statements
// TODO whats with the coeffiecients before the exponential funciton?
void initPhi(double *f, double *R, int N, double PSI) {
    double edgeZ = wingL + 2;
    double edgeR = wingL;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            f[i + N * j] = exp(-pow(R[j] - (Rmu), 2) / (2.0 * pow(Rsigma, 2)));
            if ((i < edgeZ) | (i > (N - 1 - edgeZ)))
                f[i + N * j] = 0.0;
            if ((j < edgeR) | (j > (N - 1 - edgeR)))
                f[i + N * j] = 0.0;
        }
    // normalization
    /*
    integral phi dz drho = intgral psi dz = L N <psi> = L N PSI
    sum phi IZ = N <psi> = N PSI
    */
    double phiSum = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            phiSum += f[i + N * j];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            f[i + N * j] = f[i + N * j] / phiSum * PSI * (N - 2 * edgeZ);
}

#define SET_OUT_FILE(FILENAME) \
    snprintf(outFilePath, sizeof(outFilePath), "%s/%s", cfg.run.outDir, FILENAME)

// FIX:
#define DEBUG_APPEND()                                                               \
    do {                                                                             \
        printf("\n [DEBUG] \n");                                                     \
        checkCuda(cudaMemcpy(h_phi.data(), d_phi, matSize, cudaMemcpyDeviceToHost)); \
        checkCuda(cudaMemcpy(h_psi.data(), d_psi, vecSize, cudaMemcpyDeviceToHost)); \
        ts_append(&w, 0.0, h_phi.data(), h_psi.data());                              \
    } while (0)

/* running simulation */
void runSim(SimConfig &cfg) {
    TSWriter w;
    char outFilePath[400];

    // TODO some values might not exist if i am not using the convolution version
    int N = cfg.run.N;
    int vecSize = N * sizeof(double);
    int matSize = N * N * sizeof(double);

    int M, kernelN, kernelSize, interpolationSize;
    if (cfg.model.modelType == CONV) {
        M = cfg.model.variant.Conv.M;
        kernelN = cfg.model.variant.Conv.kernelN;
        kernelSize = kernelN * sizeof(double);
        interpolationSize = M * sizeof(double);
    }

    printf("Creating save file...\n");
    SET_OUT_FILE("run.h5");
    ts_open(&w, outFilePath, (hsize_t)cfg.run.N);

    printf("Allocating host memory...\n");
    std::vector<double> h_R(N);
    std::vector<double> h_phi(N * N);
    std::vector<double> h_J(N * N);
    std::vector<double> h_dJ(N * N);
    std::vector<double> h_psi(N);
    std::vector<double> h_I(N);
    std::vector<double> h_percoll(N);
    std::vector<double> h_gradWing(N);
    std::vector<double> h_intKernel(N); // Only needs kernelN < N many elements, but kernelN is not defined for TAYL version

    printf("Allocating device memory...\n");
    double *d_R, *d_phi, *d_J, *d_dJ, *d_intKernel, *d_I, *d_psi, *d_psiIntp, *d_IIntp, *d_percoll, *d_gradWing, *d_b, *d_c, *d_d, *d_psiPow0, *d_psiPow1;
    cudaMalloc(&d_R, vecSize);
    cudaMalloc(&d_phi, matSize);
    cudaMalloc(&d_J, matSize);
    cudaMalloc(&d_dJ, matSize);
    cudaMalloc(&d_I, vecSize);
    cudaMalloc(&d_psi, vecSize);
    cudaMalloc(&d_percoll, vecSize);
    cudaMalloc(&d_gradWing, vecSize);
    cudaMalloc(&d_b, vecSize);
    cudaMalloc(&d_c, vecSize);
    cudaMalloc(&d_d, vecSize);

    cudaMalloc(&d_psiPow0, vecSize); // pow(1.0 - psi, mDeg)
    cudaMalloc(&d_psiPow1, vecSize); // pow(psi, mDeg)

    if (cfg.model.modelType == CONV) {
        cudaMalloc(&d_intKernel, kernelSize);
        cudaMalloc(&d_psiIntp, interpolationSize);
        cudaMalloc(&d_IIntp, interpolationSize);
    }

    printf("Initializing device memory...\n");
    // Initializing density coordinate rho.
    for (int j = 0; j < N; j++) {
        h_R[j] = RC - RL / 2 + RL * (double(j) / double(N - 1));
    }
    cudaMemcpy(d_R, h_R.data(), vecSize, cudaMemcpyHostToDevice);

    // Initializing phi.
    initPhi(h_phi.data(), h_R.data(), N, cfg.model.PSI);
    cudaMemcpy(d_phi, h_phi.data(), matSize, cudaMemcpyHostToDevice);

    // Initializing fluxes.
    cudaMemset(d_J, 0, matSize);
    cudaMemset(d_dJ, 0, matSize);

    if (cfg.model.modelType == CONV) {
        // Initializing convolution kernel
        genConvKernel(h_intKernel.data(), cfg.run.DZ, cfg.model.U);
        cudaMemcpy(d_intKernel, h_intKernel.data(), kernelSize, cudaMemcpyHostToDevice);

        // Initializing interpolated psi.
        cudaMemset(d_psiIntp, 0, interpolationSize);

        // Initializing interpolated integral.
        cudaMemset(d_IIntp, 0, interpolationSize);
    }

    // Interaction integral.
    cudaMemset(d_I, 0, vecSize);

    // Initializing psi.
    cudaMemset(d_psi, 0, vecSize);

    // Initializing percoll gradient and gradient wing.
    cudaMemset(d_percoll, 0, vecSize);
    cudaMemset(d_gradWing, 0, vecSize);

    // Initialize arrays for spline interpolation.
    cudaMemset(d_b, 0, vecSize);
    cudaMemset(d_c, 0, vecSize);
    cudaMemset(d_d, 0, vecSize);

    printf("starting timer.\n");
    // start time measurement
    float milliseconds;
    cudaEvent_t startEvent, stopEvent;
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    printf("defining grid and starting loop.\n");

    dim3 blockN(N, 1);
    dim3 gridN((N + blockN.x - 1) / blockN.x, 1);
    dim3 block2D(N, 1);
    dim3 grid2D((N + block2D.x - 1) / block2D.x, N);

    checkCuda(cudaEventRecord(startEvent, 0));

    /**************************************************************/
    ts_writeR(&w, h_R.data());
    // ts_writeZ(&w, Z);
    /**************************************************************/

    // iteration loop
    int n_out = cfg.run.NO;
    int NT = cfg.run.NT;
    double DT = cfg.run.DT;
    double t = 0.0;
    for (int i = 0; i < NT; i++) {
        /* integration */
        CuKernelInte<<<gridN, blockN>>>(d_phi, d_psi);

        if (cfg.model.modelType == CONV) {
            dim3 blockM(N, 1);
            dim3 gridM((M + blockM.x - 1) / blockM.x, 1);

            size_t shared_mem = 2ull * N * sizeof(double);

            int subDiv = cfg.model.variant.Conv.subDiv;
            int kernelN = cfg.model.variant.Conv.kernelN;

            CuKernelSplineCoeffs<<<1, 1, shared_mem>>>(d_psi, d_b, d_c, d_d, N);
            CuKernelSplineEval<<<gridM, blockM>>>(d_psi, d_b, d_c, d_d, d_psiIntp, N, M, subDiv);

            CuKernelConv<<<gridM, blockM, (N + kernelN - 1) * sizeof(double)>>>(
                d_psiIntp,
                d_IIntp,
                d_intKernel,
                M,
                kernelN,
                subDiv);

            CuKernelSplineDownSample<<<1, blockN>>>(d_IIntp, d_I, subDiv);
        } else if (cfg.model.modelType == TAYL) {
            CuKernelTayl<<<gridN, blockN>>>(d_psi, d_I, cfg.model.variant.Tayl.NU, cfg.model.variant.Tayl.MU);
        } else {
            printf("This branch should never be reached!");
        }

        // TODO: Store which model is used outside the loop! Maybe make a gradientType Enum
        if (strcmp(cfg.model.gradient, "linear") == 0) {
            CuKernelGradLinear<<<gridN, blockN>>>(d_percoll, t);
            CuKernelWingLinear<<<gridN, blockN>>>(d_percoll, d_gradWing, t);
        } else if (strcmp(cfg.model.gradient, "sigmoid") == 0) {
            CuKernelGradSigmoid<<<gridN, blockN>>>(d_percoll, t);
            CuKernelWingSigmoid<<<gridN, blockN>>>(d_percoll, d_gradWing, t);
            if (i == 1000) {
                cudaMemcpy(h_percoll.data(), d_percoll, vecSize, cudaMemcpyDeviceToHost);
                ts_writeZ(&w, h_percoll.data());
            }
        } else {
            printf("This branch should never be reached!");
        }

        CuKernelDegDiffPow<<<gridN, blockN>>>(d_psi, d_psiPow0, d_psiPow1);

        /* iteration */
        CuKernelIter<<<grid2D, block2D>>>(
            d_phi,
            d_J,
            d_dJ,
            d_percoll,
            d_R,
            d_I,
            d_psi,
            d_psiPow0,
            d_psiPow1,
            t,
            d_gradWing);

        cudaDeviceSynchronize();

        if ((((i - 1) % n_out) == 0) | (i == 1) | (i == NT)) {
            // retrieve data from GPU mem
            checkCuda(cudaMemcpy(h_phi.data(), d_phi, matSize, cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(h_I.data(), d_I, vecSize, cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(h_psi.data(), d_psi, vecSize, cudaMemcpyDeviceToHost));

            /**************************************/
            ts_append(&w, t, h_phi.data(), h_psi.data());
            /**************************************/

            // measure time
            checkCuda(cudaEventRecord(stopEvent, 0));
            checkCuda(cudaEventSynchronize(stopEvent));
            checkCuda(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

            printf("step: %d/%d\n", i, NT);
            printf("runtime (sec): %.5f\n", milliseconds / 1000.0);
            printf("remaining (sec): %.5f\n", milliseconds / 1000.0 * (NT - i) / i);
        }

        t += DT;
    }

    printf("finished.\n\n");

    // stop timer
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

    // show stats
    printf("   total steps: %d\n", NT);
    printf("   total time (ms): %f\n", milliseconds);
    printf("   average time (ms): %f\n", milliseconds / NT);

    // delete arrays and free memory
    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));

    cudaFree(d_R);
    cudaFree(d_phi);
    cudaFree(d_J);
    cudaFree(d_dJ);
    cudaFree(d_I);
    cudaFree(d_psi);
    cudaFree(d_percoll);
    cudaFree(d_gradWing);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);

    if (cfg.model.modelType == CONV) {
        cudaFree(d_intKernel);
        cudaFree(d_psiIntp);
        cudaFree(d_IIntp);
    }

    cudaFree(d_psiPow0);
    cudaFree(d_psiPow1);

    /****************************/
    ts_close(&w);
    /****************************/
}
