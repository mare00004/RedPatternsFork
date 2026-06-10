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
#include <stdlib.h>
#include <time.h>
#include <vector>

/* kernel function */

inline double fLJ(double r, double sigma, double U) {
    return (4 * U * (12 * pow(sigma, 12) / pow(r, 13) - 6 * pow(sigma, 6) / pow(r, 7)));
}

inline double g(double r, double d, double sigmaC) {
    return 4e7 * exp(-pow(r - d, 2) / (2 * pow(sigmaC, 2)));
}

void genConvKernel(double *intKernel, int kernelN, double DZ, double U) {
    // TODO figure out what does do and why i need the outside of the convolution version
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

#define TO_BYTES(num) num * sizeof(double)

namespace {
struct ProgressState {
    const char *status;
    int step;
    int total_steps;
    double elapsed_sec;
    double remaining_sec;
    double sim_time_sec;
    const char *run_h5_path;
    const char *error;
};

void writeJsonEscaped(FILE *stream, const char *value) {
    if (value == NULL) {
        fputs("", stream);
        return;
    }

    for (const unsigned char *p = (const unsigned char *)value; *p != '\0'; ++p) {
        switch (*p) {
        case '\\':
            fputs("\\\\", stream);
            break;
        case '"':
            fputs("\\\"", stream);
            break;
        case '\n':
            fputs("\\n", stream);
            break;
        case '\r':
            fputs("\\r", stream);
            break;
        case '\t':
            fputs("\\t", stream);
            break;
        default:
            fputc(*p, stream);
            break;
        }
    }
}

bool writeProgressFile(const char *progressPath, const ProgressState &state) {
    char tempPath[512];
    if (snprintf(tempPath, sizeof(tempPath), "%s.tmp", progressPath) >= (int)sizeof(tempPath)) {
        return false;
    }

    FILE *stream = fopen(tempPath, "w");
    if (stream == NULL) {
        return false;
    }

    fprintf(stream, "{\n");
    fprintf(stream, "  \"status\": \"");
    writeJsonEscaped(stream, state.status);
    fprintf(stream, "\",\n");
    fprintf(stream, "  \"step\": %d,\n", state.step);
    fprintf(stream, "  \"total_steps\": %d,\n", state.total_steps);
    fprintf(stream, "  \"elapsed_sec\": %.9f,\n", state.elapsed_sec);
    fprintf(stream, "  \"remaining_sec\": %.9f,\n", state.remaining_sec);
    fprintf(stream, "  \"sim_time_sec\": %.9f,\n", state.sim_time_sec);
    fprintf(stream, "  \"updated_at_unix\": %lld,\n", (long long)time(NULL));
    fprintf(stream, "  \"run_h5_path\": \"");
    writeJsonEscaped(stream, state.run_h5_path);
    fprintf(stream, "\"");

    if (state.error != NULL && state.error[0] != '\0') {
        fprintf(stream, ",\n  \"error\": \"");
        writeJsonEscaped(stream, state.error);
        fprintf(stream, "\"");
    }

    fprintf(stream, "\n}\n");

    if (fclose(stream) != 0) {
        remove(tempPath);
        return false;
    }

    if (rename(tempPath, progressPath) != 0) {
        remove(tempPath);
        return false;
    }

    return true;
}

void updateProgress(const char *progressPath, const ProgressState &state) {
    if (!writeProgressFile(progressPath, state)) {
        fprintf(stderr, "Warning: failed to update progress file %s\n", progressPath);
    }
}
} // namespace

/* running simulation */
int runSim(SimConfig &cfg) {
    TSWriter w;
    char outFilePath[400];
    char progressPath[400];
    snprintf(outFilePath, sizeof(outFilePath), "%s/%s", cfg.run.outDir, "run.h5");
    snprintf(progressPath, sizeof(progressPath), "%s/%s", cfg.run.outDir, "progress.json");

    const int N = cfg.run.N;
    const bool useConvolution = cfg.model.modelType == CONV;
    const int NT = cfg.run.NT;
    const double DT = cfg.run.DT;

    auto failRun = [&](const char *message, int step, double elapsedSec, double remainingSec, double simTimeSec) {
        updateProgress(
            progressPath,
            ProgressState{
                "failed",
                step,
                NT,
                elapsedSec,
                remainingSec,
                simTimeSec,
                outFilePath,
                message,
            });
        return 1;
    };

    // Dimensions of various arrays in units of grid points
    const int numZCells = N;
    const int numZFaces = N + 1;
    const int numRhoPoints = N;
    const int numPhiPoints = numZCells * numRhoPoints;
    const int numPsiPoints = numZCells;
    const int numFluxPoints = numZFaces * numRhoPoints;

    // Load external kernel file, or call `genConvKernel` if kernel file argument is not used
    int M = 0;
    int numFineZCells = 0;
    int numKernelPoints = 0;
    std::vector<double> h_intKernel;
    if (useConvolution) {
        M = cfg.model.variant.Conv.M;
        if (cfg.model.variant.Conv.kernelFile[0] != '\0') {
            double *loadedKernel = NULL;

            if (loadConvKernelFile(cfg.model.variant.Conv.kernelFile, &loadedKernel, &numKernelPoints) != 0) {
                fprintf(stderr, "Failed to load convolution kernel from %s\n", cfg.model.variant.Conv.kernelFile);
                return failRun("Failed to load convolution kernel", 0, 0.0, 0.0, 0.0);
            }

            h_intKernel.assign(loadedKernel, loadedKernel + numKernelPoints);
            free(loadedKernel);
        } else {
            numKernelPoints = cfg.model.variant.Conv.kernelN;
            h_intKernel.resize(numKernelPoints);
            genConvKernel(h_intKernel.data(), numKernelPoints, cfg.run.DZ, cfg.model.U);
        }

        cfg.model.variant.Conv.kernelN = numKernelPoints;
        numFineZCells = M;
    }

    printf("Allocating host memory...\n");
    std::vector<double> h_R(numRhoPoints); // rho grid points
    std::vector<double> h_Z(numZCells);    // z cell centers
    std::vector<double> h_phi(numPhiPoints);
    std::vector<double> h_psi(numPsiPoints);
    std::vector<double> h_I(numPsiPoints);

    std::vector<double> h_J(N * N);
    std::vector<double> h_dJ(N * N);
    std::vector<double> h_percoll(N);
    std::vector<double> h_gradWing(N);

    printf("Allocating device memory...\n");
    double *d_R = nullptr, *d_phi = nullptr, *d_F = nullptr, *d_intKernel = nullptr, *d_I = nullptr, *d_psi = nullptr, *d_psiIntp = nullptr,
           *d_IIntp = nullptr, *d_percoll = nullptr, *d_gradWing = nullptr, *d_b = nullptr, *d_c = nullptr, *d_d = nullptr, *d_Iface = nullptr;

    cudaMalloc(&d_R, TO_BYTES(numRhoPoints));
    cudaMalloc(&d_phi, TO_BYTES(numPhiPoints));
    cudaMalloc(&d_psi, TO_BYTES(numPsiPoints));
    cudaMalloc(&d_I, TO_BYTES(numPsiPoints));
    cudaMalloc(&d_F, TO_BYTES(numFluxPoints));
    cudaMalloc(&d_percoll, TO_BYTES(numZCells));
    cudaMalloc(&d_gradWing, TO_BYTES(numZCells));
    cudaMalloc(&d_Iface, TO_BYTES(numZFaces));

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    bool writer_created = false;

    auto cleanup = [&]() {
        if (startEvent != nullptr) {
            checkCuda(cudaEventDestroy(startEvent));
        }
        if (stopEvent != nullptr) {
            checkCuda(cudaEventDestroy(stopEvent));
        }

        cudaFree(d_R);
        cudaFree(d_phi);
        cudaFree(d_F);
        cudaFree(d_I);
        cudaFree(d_psi);
        cudaFree(d_Iface);
        cudaFree(d_percoll);
        cudaFree(d_gradWing);

        if (cfg.model.modelType == CONV) {
            cudaFree(d_intKernel);
            cudaFree(d_psiIntp);
            cudaFree(d_IIntp);
            cudaFree(d_b);
            cudaFree(d_c);
            cudaFree(d_d);
        }

        if (writer_created) {
            ts_close(&w);
        }
    };

    if (cfg.model.modelType == CONV) {
        cudaMalloc(&d_intKernel, TO_BYTES(numKernelPoints));
        cudaMalloc(&d_psiIntp, TO_BYTES(numFineZCells));
        cudaMalloc(&d_IIntp, TO_BYTES(numFineZCells));
        cudaMalloc(&d_b, TO_BYTES(numZCells));
        cudaMalloc(&d_c, TO_BYTES(numZCells));
        cudaMalloc(&d_d, TO_BYTES(numZCells));
    }

    printf("Initializing device memory...\n");

    // Initializing density coordinate rho.
    for (int j = 0; j < numRhoPoints; j++) {
        h_R[j] = RC - RL / 2 + RL * (double(j) / double(numRhoPoints - 1));
    }
    cudaMemcpy(d_R, h_R.data(), TO_BYTES(numRhoPoints), cudaMemcpyHostToDevice);

    // Initializing Z coordinate at cell centers
    for (int j = 0; j < numZCells; j++) {
        h_Z[j] = (double(j) + 0.5) * cfg.run.DZ;
    }

    // Initializing phi from an external file when requested.
    if (cfg.model.initialPhiFile[0] != '\0') {
        double *loadedPhi = NULL;

        if (loadInitialPhiFile(cfg.model.initialPhiFile, &loadedPhi, N) != 0) {
            fprintf(stderr, "Failed to load initial phi from %s\n", cfg.model.initialPhiFile);
            cleanup();
            return failRun("Failed to load initial phi", 0, 0.0, 0.0, 0.0);
        }

        h_phi.assign(loadedPhi, loadedPhi + N * N);
        free(loadedPhi);
    } else {
        initPhi(h_phi.data(), h_R.data(), N, cfg.model.PSI);
    }

    cudaMemcpy(d_phi, h_phi.data(), TO_BYTES(numPhiPoints), cudaMemcpyHostToDevice);
    cudaMemset(d_I, 0, TO_BYTES(numZCells));
    cudaMemset(d_psi, 0, TO_BYTES(numPhiPoints));
    cudaMemset(d_percoll, 0, TO_BYTES(numZCells));
    cudaMemset(d_gradWing, 0, TO_BYTES(numZCells));

    if (cfg.model.modelType == CONV) {
        cudaMemcpy(d_intKernel, h_intKernel.data(), TO_BYTES(numKernelPoints), cudaMemcpyHostToDevice);
        cudaMemset(d_psiIntp, 0, TO_BYTES(numFineZCells));
        cudaMemset(d_IIntp, 0, TO_BYTES(numFineZCells));
        cudaMemset(d_b, 0, TO_BYTES(numZCells));
        cudaMemset(d_c, 0, TO_BYTES(numZCells));
        cudaMemset(d_d, 0, TO_BYTES(numZCells));
    }

    printf("Creating save file...\n");
    if (ts_create(
            &w,
            outFilePath,
            &cfg,
            h_R.data(),
            h_Z.data(),
            h_phi.data(),
            useConvolution ? h_intKernel.data() : NULL,
            useConvolution ? numKernelPoints : 0) != 0) {
        fprintf(stderr, "Failed to create output file %s\n", outFilePath);
        cleanup();
        return failRun("Failed to create output file", 0, 0.0, 0.0, 0.0);
    }
    writer_created = true;

    updateProgress(
        progressPath,
        ProgressState{
            "running",
            0,
            NT,
            0.0,
            0.0,
            0.0,
            outFilePath,
            NULL,
        });

    printf("starting timer.\n");
    // start time measurement
    float milliseconds;
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    printf("defining grid and starting loop.\n");

    constexpr unsigned int one_dim_block_size = 256;
    const dim3 cell_block_dim(one_dim_block_size, 1, 1);
    const dim3 cell_grid_dim((numZCells + cell_block_dim.x - 1) / cell_block_dim.x, 1, 1);
    const dim3 face_block_dim(one_dim_block_size, 1, 1);
    const dim3 face_grid_dim((numZFaces + face_block_dim.x - 1) / face_block_dim.x, 1, 1);

    checkCuda(cudaEventRecord(startEvent, 0));

    // iteration loop
    int n_out = cfg.run.NO;
    double t = 0.0;
    for (int i = 0; i < NT; i++) {
        /* integration */
        CuKernelInte<<<cell_grid_dim, cell_block_dim>>>(d_phi, d_psi);

        if (useConvolution) {
            const int subDiv = cfg.model.variant.Conv.subDiv;
            const dim3 fine_cell_block_dim(one_dim_block_size, 1, 1);
            const dim3 fine_cell_grid_dim((numFineZCells + fine_cell_block_dim.x - 1) / fine_cell_block_dim.x, 1, 1);
            const size_t spline_shared_bytes = 2ull * numZCells * sizeof(double);

            CuKernelSplineCoeffs<<<1, 1, spline_shared_bytes>>>(d_psi, d_b, d_c, d_d, cfg.run.N);
            CuKernelSplineEvalCellCentered<<<fine_cell_grid_dim, fine_cell_block_dim>>>(d_psi, d_b, d_c, d_d, d_psiIntp, cfg.run.N, cfg.model.variant.Conv.M, subDiv);

            CuKernelConv<<<fine_cell_grid_dim, fine_cell_block_dim, (fine_cell_block_dim.x + numKernelPoints - 1) * sizeof(double)>>>(
                d_psiIntp,
                d_IIntp,
                d_intKernel,
                numFineZCells,
                numKernelPoints,
                subDiv);

            CuKernelDownSample<<<cell_grid_dim, cell_block_dim>>>(
                d_IIntp,
                d_I,
                subDiv);

            CuKernelCellToFace<<<face_grid_dim, face_block_dim>>>(
                d_I,
                d_Iface);
        } else if (cfg.model.modelType == TAYL) {
            double dz = cfg.run.DZ;
            double c1 = cfg.model.variant.Tayl.NU / (12.0 * dz);
            double c3 = cfg.model.variant.Tayl.MU / (2.0 * dz * dz * dz);

            CuKernelTaylFace<<<face_grid_dim, face_block_dim>>>(
                d_psi,
                d_Iface,
                c1,
                c3);
        } else {
            printf("This branch should never be reached!");
        }

        if (cfg.model.gradientType == LINEAR) {
            CuKernelGradLinear<<<cell_grid_dim, cell_block_dim>>>(d_percoll, t);
            CuKernelWingLinear<<<cell_grid_dim, cell_block_dim>>>(d_percoll, d_gradWing, t);
        } else if (cfg.model.gradientType == SIGMOID) {
            CuKernelGradSigmoid<<<cell_grid_dim, cell_block_dim>>>(d_percoll, t);
            CuKernelWingSigmoid<<<cell_grid_dim, cell_block_dim>>>(d_percoll, d_gradWing, t);
        } else {
            printf("This branch should never be reached!");
        }

        const dim3 flux_block_dim(16, 16, 1);
        const dim3 flux_grid_dim(
            (numZFaces + flux_block_dim.x - 1) / flux_block_dim.x,
            (numZCells + flux_block_dim.y - 1) / flux_block_dim.y,
            1);

        CuKernelComputeFluxFVM<<<flux_grid_dim, flux_block_dim>>>(
            d_phi,
            d_F,
            d_percoll,
            d_R,
            d_Iface,
            d_gradWing);

        const dim3 update_block_dim(16, 16, 1);
        const dim3 update_grid_dim(
            (numZCells + update_block_dim.x - 1) / update_block_dim.x,
            (numZCells + update_block_dim.y - 1) / update_block_dim.y,
            1);

        CuKernelUpdatePhiFVM<<<update_grid_dim, update_block_dim>>>(
            d_phi,
            d_F);

        cudaDeviceSynchronize();

        if ((((i - 1) % n_out) == 0) || (i == 1) || (i == NT - 1)) {
            const double savedTime = t + DT;
            const int currentStep = i + 1;
            CuKernelInte<<<cell_grid_dim, cell_block_dim>>>(d_phi, d_psi);

            // retrieve data from GPU mem
            checkCuda(cudaMemcpy(h_phi.data(), d_phi, TO_BYTES(numPhiPoints), cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(h_psi.data(), d_psi, TO_BYTES(numZCells), cudaMemcpyDeviceToHost));

            if (ts_append(&w, savedTime, h_phi.data(), h_psi.data()) != 0) {
                fprintf(stderr, "Failed to append timestep data to %s\n", outFilePath);
                cleanup();
                return failRun("Failed to append timestep data", currentStep, 0.0, 0.0, savedTime);
            }

            // measure time
            checkCuda(cudaEventRecord(stopEvent, 0));
            checkCuda(cudaEventSynchronize(stopEvent));
            checkCuda(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
            const double elapsedSec = milliseconds / 1000.0;
            const double remainingSec = elapsedSec * double(NT - currentStep) / double(currentStep);

            printf("step: %d/%d\n", currentStep, NT);
            printf("runtime (sec): %.5f\n", elapsedSec);
            printf("remaining (sec): %.5f\n", remainingSec);

            updateProgress(
                progressPath,
                ProgressState{
                    "running",
                    currentStep,
                    NT,
                    elapsedSec,
                    remainingSec,
                    savedTime,
                    outFilePath,
                    NULL,
                });
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

    ts_postRunInfo(&w, milliseconds / 1000);
    updateProgress(
        progressPath,
        ProgressState{
            "finished",
            NT,
            NT,
            milliseconds / 1000.0,
            0.0,
            cfg.run.T,
            outFilePath,
            NULL,
        });

    /****************************/
    cleanup();
    /****************************/
    return 0;
}
