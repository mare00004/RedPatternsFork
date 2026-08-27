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
    ZERO = 2,
    LINEAR_FULL = 3,
} GradientType;

typedef enum {
    PHI,
    PSI,
    PERCOLL,
    FACE_VELOCITY,
    FACE_FLUX,
    NUM_STORE_TYPES,
} StoreType;

#include <stdint.h>

typedef uint8_t StoreBitMap;

#define BITMAP_ADD(set, e) ((set) |= (1ULL << (e)))
#define BITMAP_REMOVE(set, e) ((set) &= ~(1ULL << (e)))
#define BITMAP_CONTAINS(set, e) (((set) >> (e)) & 1ULL)

typedef struct {
    int N;
    int NT;
    double T;
    double DT;
    double DZ;
    double L;
    double wingL;
    double sysL;
    double storeTime;
    StoreBitMap store;
    char outDir[256];
} RunParams;

typedef struct {
    int kernelN;
    int subDiv;
    int M;
    double fineDZ;
    char kernelFile[256];
} ConvParams;

typedef struct {
    double NU;
    double MU;
} TaylParams;

typedef struct {
    ModelType modelType;
    GradientType gradientType;
    double alpha;
    double beta;
    char initialPhiFile[256];
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
