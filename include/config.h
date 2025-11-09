#ifndef CONFIG_H
#define CONFIG_H
#include "parameters.cuh"
#include <stdio.h>

// Tagged Union for Simulation Configuration.

typedef enum {
    CONV = 0,
    TAYL = 1
} ModelType;

typedef struct {
    int N;
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
    *c = (SimConfig){ .run = { .N = N, .T = T, .DT = IT, .NO = NO },
        .model = { .modelType = CONV,
            .U = U,
            .PSI = PSI,
            .gamma = h_gamma,
            .delta = h_delta,
            .kappa = h_kappa } };
    sprintf(c->run.outDir, "./");
}

int loadTOMLConfig(const char *path, SimConfig *c);
int applyCLIOverrides(int argc, char **argv, SimConfig *c);
int deriveAndValidateOrDie(SimConfig *c);
int writeResolvedConfig(const SimConfig *c);

#endif
