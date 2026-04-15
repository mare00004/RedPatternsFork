#ifndef SIM_TYPES_H
#define SIM_TYPES_H
// Tagged Union for Simulation Configuration.
#define textFieldSize 64

typedef enum {
    CONV = 0,
    TAYL = 1
} ModelType;

typedef enum {
    LINEAR = 0,
    SIGMOID = 1,
} GradientType;

typedef struct {
    int N;
    int NT;
    double T;
    double DT;
    double DZ;
    double fineDZ;
    double sysL;
    int NO;
    char outDir[256];
} RunParams;

typedef struct {
    int kernelN;
    int subDiv;
    int M;
    char kernelFile[256];
} ConvParams;

typedef struct {
    double NU;
    double MU;
} TaylParams;

typedef struct {
    ModelType modelType;
    GradientType gradientType;
    double U;
    double PSI;
    double alpha;
    double beta;
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
