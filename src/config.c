#include "config.h"
#include "sim_types.h"
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* API */
void setDefaults(SimConfig *c) {
    int N = 256;
    int subDiv = 256;

    double L = 0.06;                 // Physical system size [cm]
    double wingL = 0.005;            // wing size [cm]
    double sysL = L + (2.0 * wingL); // simulation system size [cm]

    /*
     * To reproduce the old code use: `int M = (N)*subDiv + 1;`
     */
    int M = (N - 1) * subDiv + 1;

    double DZ = sysL / N;
    double fineDZ = sysL / M;

    double DT = 0.005;

    StoreBitMap store = 0;
    BITMAP_ADD(store, PHI);
    BITMAP_ADD(store, PSI);

    *c = (SimConfig){
        .run = {
            .N = N,
            .NT = 240000,
            .T = 1200,
            .DT = DT,
            .DZ = DZ,
            .L = L,
            .wingL = wingL,
            .sysL = sysL,
            .storeTime = 500 * DT,
            .store = store,
            .outDir = "./",
        },
        .model = {
            .modelType = CONV,
            .gradientType = LINEAR,
            .alpha = 2.0e-05, // (a * V) / ZETA
            .beta = 7.4e23,   // (2 PI) / (ZETA * V)
            .initialPhiFile = "",
            .variant = {
                .Conv = (ConvParams){
                    .kernelN = 31, // TODO: why 31?
                    .subDiv = 256,
                    .fineDZ = fineDZ,
                    .M = M,
                    .kernelFile = "",
                },
            },
        }
    };
}

// TODO: update
int printConfig(SimConfig *c) {
    printf("-----------------------\n");
    printf("-  Simulation Config  -\n");
    printf("-----------------------\n");
    printf("-> Run:\n");
    printf("\t-> N: %d\n", c->run.N);
    printf("\t-> T: %f\n", c->run.T);
    printf("\t-> DT: %.5e\n", c->run.DT);
    printf("\t-> storeTime: %f\n", c->run.storeTime);
    printf("\t-> outDir: %s\n", c->run.outDir);
    if (c->model.modelType == CONV) {
        printf("-> Using Convolution-Model:\n");
    } else {
        printf("-> Using Taylor-Model:\n");
    }
    printf("-> Using %s gradient", (c->model.gradientType == LINEAR) ? "linear" : "sigmoid");
    if (strlen(c->model.initialPhiFile) > 0) {
        printf("\t-> initial phi file: %s\n", c->model.initialPhiFile);
    }
    if (c->model.modelType == TAYL) {
        printf("\t-> nu: %.5e\n", c->model.variant.Tayl.NU);
        printf("\t-> mu: %.5e\n", c->model.variant.Tayl.MU);
    } else if (strlen(c->model.variant.Conv.kernelFile) > 0) {
        printf("\t-> kernel file: %s\n", c->model.variant.Conv.kernelFile);
    }

    return 0;
}

int deriveAndValidateOrDie(SimConfig *c) {
    /************
     * VALIDATE *
     ************/
    if (c->run.storeTime <= 0) {
        fprintf(stderr, "storeTime needs to be positive!\n");
        return -1;
    }
    if ((c->run.N < 32) || (c->run.N % 2 != 0)) {
        fprintf(stderr, "N needs to be a power of 2 greater than 32");
        return -1;
    }
    if (c->run.T <= 0) {
        fprintf(stderr, "T needs to be positive!\n");
        return -1;
    }
    if (c->run.DT <= 0) {
        fprintf(stderr, "DT needs to be positive!\n");
        return -1;
    }
    DIR *dir = opendir(c->run.outDir);
    if (dir) {
        closedir(dir);
    } else {
        fprintf(stderr, "%s is not a valid directory\n", c->run.outDir);
        return -1;
    }
    if (!(c->model.gradientType == LINEAR || c->model.gradientType == SIGMOID)) {
        fprintf(stderr, "gradient has to be one of: linear, sigmoid!\n");
        return -1;
    }
    if (!(c->model.modelType == CONV || c->model.modelType == TAYL)) {
        fprintf(stderr, "modelType has to be convolution or taylor!\n");
        return -1;
    }
    if (strlen(c->model.initialPhiFile) > 0) {
        FILE *initialPhiFile = fopen(c->model.initialPhiFile, "rb");
        if (initialPhiFile == NULL) {
            fprintf(stderr, "%s is not a readable initial phi file\n", c->model.initialPhiFile);
            return -1;
        }
        fclose(initialPhiFile);
    }
    if (c->model.modelType == CONV && strlen(c->model.variant.Conv.kernelFile) > 0) {
        FILE *kernelFile = fopen(c->model.variant.Conv.kernelFile, "rb");
        if (kernelFile == NULL) {
            fprintf(stderr, "%s is not a readable kernel file\n", c->model.variant.Conv.kernelFile);
            return -1;
        }
        fclose(kernelFile);
    }

    /**********
     * DERIVE *
     **********/
    c->run.NT = ceil(c->run.T / c->run.DT);

    c->run.DZ = c->run.sysL / ((double)c->run.N);

    if (c->model.modelType == CONV) {
        c->model.variant.Conv.M = (c->run.N - 1) * c->model.variant.Conv.subDiv + 1;
        c->model.variant.Conv.fineDZ = c->run.sysL / ((double)c->model.variant.Conv.M);
    }

    if (c->model.gradientType == LINEAR) {
        c->model.alpha = 2.0e-05;
    } else if (c->model.gradientType == SIGMOID) {
        c->model.alpha = 2.0e-04;
    }

    return 0;
}
