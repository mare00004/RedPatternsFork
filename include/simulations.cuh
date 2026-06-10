#pragma once
#include "sim_types.h"

void genConvKernel(double *intKernel, int kernelN, double DZ, double U);
void initPhi(double *f, double *R, int N, double PSI);
int runSim(SimConfig &cfg);
