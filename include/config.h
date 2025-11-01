#ifndef CONFIG_H
#define CONFIG_H
#include "parameters.cuh"

// Tagged Union for Simulation Configuration.

typedef enum { CONV = 0, TAYL = 1 } ModelType;

typedef struct {
    int N;
    double T;
    double DT;
    int NO;
    double U;
    double PSI;
    double gamma;
    double delta;
    double kappa;
    char outDir[256];
} RunParams;

typedef struct {
} ConvParams;

typedef struct {
    double NU;
    double MU;
} TaylParams;

typedef union {
    ConvParams Conv;
    TaylParams Tayl;
} ModelParams;

typedef struct {
    ModelType modelType;
    RunParams run;
    ModelParams model;
} SimConfig;

/* API */
int setDefaults(SimConfig *c) {
    c->modelType = CONV;
    c->run.N = N;
    c->run.T = T;
    c->run.DT = IT;
    c->run.NO = NO;
    c->run.U = U;
    c->run.PSI = PSI;
    c->run.gamma = h_gamma;
    c->run.delta = h_delta;
    c->run.kappa = h_kappa;
    strcpy(c->run.outDir, "./");

    return 0;
}

int loadTOMLConfig(const char *path, SimConfig *c);
int applyCLIOverrides(int argc, char **argv, SimConfig *c);
int deriveAndValidateOrDie(SimConfig *c);
int writeResolvedConfig(const SimConfig *c);

#endif
