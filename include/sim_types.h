#ifndef SIM_TYPES_H
#define SIM_TYPES_H
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

#endif // SIM_TYPES_H
