#ifndef PARAMETERS_H
#define PARAMETERS_H
#include <math.h>

/*******
 * RUN *
 *******/

// time iteration parameters
double T = 1200.0;     // [s] total simulation time
double IT = 0.005;     // [s] time increment
int NO = 1000;         // [steps] output interval
int NT = ceil(T / IT); // number of time steps

const int N = 256; // grid size (N x N)

#define sysL (M * 1.041412353515625e-6) //[m] system length

// density dimension
#define RC 1100.0 // central density
#define RL 30.0   // density range (RC +- RL/2)

/*********
 * MODEL *
 *********/

// RBC parameters
double PSI = 0.02;  // [v/v] RBC average volume fraction
double U = 100e-18; // [J] RBC effective interaction energy

// initial RBC density function
#define Rsigma 4.0f // [g/l] gaussian width
#define Rmu 1100.0f // [g/l] central RBC density

// Percoll density gradient
#define gradL 0.06                  // [m] tube length
#define wingL 30                    // [grid] length of gradient wings
#define zShift ((sysL - gradL) / 2) // gradient spatial center
#define P0 1100.0                   // [g/l] central PC density

// host flux prefactors
double h_beta = 7.4e23; // interaction integral
double h_alpha = 2e-4;  // exp -4 for 20000 g, exp -5 for 2000 g
double h_gamma = 3e-10; // degenerate diffusion restriction phi 0
double h_delta = 1e-15; // degenerate diffusion restriction psi 0
double h_kappa = 1e-15; // degenerate diffusion restriction psi 1

// degenerate diffusion flux
#define mDeg 500
#define jDegDiffPhi0(i) (pow(1.0 - phi[i], mDeg))
#define jDegDiffPsi0(gi, i) (pow(1.0 - psi[i], mDeg) * abs(phi[gi]))
#define jDegDiffPsi1(gi, i) (-pow(psi[i], mDeg)) * abs(phi[gi])

/**********************
 * MODEL->Convolution *
 **********************/

#define kernelN 31                // kernel size
#define subDiv 256.0              // subdivision
#define M int(N * subDiv + 1)     // size of interpolated grid
const double IZ = sysL / (N - 1); // [m] space increment

/*****************
 * MODEL->Taylor *
 *****************/
#define nu_interaction -8.6565e-14
#define mu_interaction -1.3670e-20

/********
 * MISC *
 ********/

#define PI 3.141592653589793115997963468544185161590576171875

// interaction convolution kernel
double intKernel[kernelN]; // kernel array
// cuda device constants
__constant__ double c_IZ;    // cuda space increment
__constant__ double c_IT;    // cuda time increment
__constant__ double c_PSI;   // cuda concentration
__constant__ double c_alpha; // sedimentation
__constant__ double c_beta;  // interaction
__constant__ double c_gamma; // restriction phi 0
__constant__ double c_delta; // restriction psi 0
__constant__ double c_kappa; // restriction psi 1

__constant__ double c_nu;
__constant__ double c_mu;

#endif
