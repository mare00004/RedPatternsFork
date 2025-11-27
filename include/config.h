#ifndef CONFIG_H
#define CONFIG_H
#include "parameters.cuh"
#include <dirent.h>
#include <stdio.h>
#include <string.h>

// Tagged Union for Simulation Configuration.
#define textFieldSize 64

typedef enum {
    CONV = 0,
    TAYL = 1
} ModelType;

typedef struct {
    int N;
    int NT;
    double T;
    double DT;
    int NO;
    char outDir[256];
} RunParams;

typedef struct {
    unsigned char _unused;
} ConvParams;

typedef struct {
    double NU;
    double MU;
} TaylParams;

typedef struct {
    ModelType modelType;
    char gradient[textFieldSize];
    double U;
    double PSI;
    double gamma;
    double delta;
    double kappa;
    union {
        ConvParams Conv;
        TaylParams Tayl;
    } variant;
} ModelParams;

typedef struct {
    RunParams run;
    ModelParams model;
} SimConfig;

/* API */
void setDefaults(SimConfig *c) {
    *c = (SimConfig){
        .run = { .N = N, .NT = 1, .T = T, .DT = IT, .NO = NO, .outDir = "./" },
        .model = {
            .modelType = CONV,
            .gradient = "linear",
            .U = U,
            .PSI = PSI,
            .gamma = h_gamma,
            .delta = h_delta,
            .kappa = h_kappa,
            .variant = { .Conv = { 0 } } }
    };
}

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
    printf("\t-> gradient: %*s\n", textFieldSize, c->model.gradient);
    printf("\t-> U: %.5e\n", c->model.U);
    printf("\t-> PSI: %.5e\n", c->model.PSI);
    printf("\t-> gamma: %.5e\n", c->model.gamma);
    printf("\t-> delta: %.5e\n", c->model.delta);
    printf("\t-> kappa: %.5e\n", c->model.kappa);
    if (c->model.modelType == TAYL) {
        printf("\t-> nu: %.5e\n", c->model.variant.Tayl.NU);
        printf("\t-> mu: %.5e\n", c->model.variant.Tayl.MU);
    }

    return 0;
}

int loadTOMLConfig(const char *path, SimConfig *c);
int applyCLIOverrides(int argc, char **argv, SimConfig *c);

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
    if (strcmp(c->model.gradient, "linear") != 0 && strcmp(c->model.gradient, "sigmoid")) {
        fprintf(stderr, "gradient has to be one of: linear, sigmoid!\n");
        return -1;
    }
    if (c->model.modelType == CONV) {
        return 0;
    } else if (c->model.modelType == TAYL) {
        return 0;
    } else {
        fprintf(stderr, "modelType has to be convolution or taylor!\n");
        return -1;
    }

    /**********
     * DERIVE *
     **********/
    c->run.NT = ceil(c->run.T / c->run.DT);

    return 0;
}

int writeResolvedConfig(const SimConfig *c);

#endif
