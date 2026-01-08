#pragma once
#include "sim_types.h"

void genConvKernel(double *intKernel, double DZ, double U);
void initPhi(double *f, double *R, int N, double PSI);
void runSim(SimConfig &cfg);
