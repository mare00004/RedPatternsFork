#include "config.h"
#include "sim_types.h"
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* API */
void setDefaults(SimConfig *c) {
    int N = 256;
    double fineDZ = 1.041412353515625e-6;
    int subDiv = 256;

    /*
     * To reproduce the old code use: `int M = (N)*subDiv + 1;`
     */
    int M = (N - 1) * subDiv + 1;

    double sysL = (double)(M - 1) * fineDZ;
    double DZ = sysL / ((double)N - 1);

    *c = (SimConfig){
        .run = {
            .N = N,
            .NT = 240000,
            .T = 1200,
            .DT = 0.005,
            .DZ = DZ,
            .fineDZ = fineDZ,
            .sysL = sysL,
            .NO = 2000,
            .outDir = "./",
        },
        .model = {
            .modelType = CONV,
            .gradientType = LINEAR,
            .U = 100e-18,
            .PSI = 0.02,
            .alpha = 2.0e-05,
            .beta = 7.4e23,
            .gamma = 3e-10,
            .delta = 1e-15,
            .kappa = 1e-15,
            .variant = {
                .Conv = (ConvParams){
                    .kernelN = 31, // TODO: why 31?
                    .subDiv = 256,
                    .M = M,
                    .kernelFile = "",
                },
            },
        }
    };
}

// TODO update
int printConfig(SimConfig *c) {
    printf("-----------------------\n");
    printf("-  Simulation Config  -\n");
    printf("-----------------------\n");
    printf("-> Run:\n");
    printf("\t-> N: %d\n", c->run.N);
    printf("\t-> T: %f\n", c->run.T);
    printf("\t-> DT: %.5e\n", c->run.DT);
    printf("\t-> NO: %d\n", c->run.NO);
    printf("\t-> outDir: %s\n", c->run.outDir);
    if (c->model.modelType == CONV) {
        printf("-> Using Convolution-Model:\n");
    } else {
        printf("-> Using Taylor-Model:\n");
    }
    printf("-> Using %s gradient", (c->model.gradientType == LINEAR) ? "linear" : "sigmoid");
    printf("\t-> U: %.5e\n", c->model.U);
    printf("\t-> PSI: %.5e\n", c->model.PSI);
    printf("\t-> gamma: %.5e\n", c->model.gamma);
    printf("\t-> delta: %.5e\n", c->model.delta);
    printf("\t-> kappa: %.5e\n", c->model.kappa);
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
    if (c->run.NO <= 0) {
        fprintf(stderr, "NO needs to be positive!\n");
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
    if (c->model.PSI <= 0 || c->model.PSI >= 1) {
        fprintf(stderr, "PSI needs to be between 0 and 1!\n");
        return -1;
    }
    if (c->model.U <= 0) {
        fprintf(stderr, "U needs to be positive\n");
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

    if (c->model.gradientType == LINEAR) {
        c->model.alpha = 2.0e-05;
    } else if (c->model.gradientType == SIGMOID) {
        c->model.alpha = 2.0e-04;
    }

    return 0;
}
