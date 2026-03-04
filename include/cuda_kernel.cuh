#pragma once

__global__ void CuKernelTayl(double *psi, double *I, double nu, double mu);
__global__ void CuKernelInte(double *phi, double *psi);
__global__ void CuKernelSplineCoeffs(
    const double *__restrict__ y,
    double *__restrict__ b,
    double *__restrict__ c,
    double *__restrict__ d,
    const int N);
__global__ void CuKernelSplineEval(
    const double *__restrict__ y,
    const double *__restrict__ b,
    const double *__restrict__ c,
    const double *__restrict__ d,
    double *__restrict__ y_intp,
    int N,
    int M,
    int subDiv);
__global__ void CuKernelSplineDownSample(double *IIntp, double *I, int subDiv);
__global__ void CuKernelConv(double *psi, double *I, double *convKernel, int M, int kernelN, int subDiv);
__global__ void CuKernelDegDiffPow(double *psi, double *psiPow0, double *psiPow1);
__global__ void CuKernelIter(
    double *phi,
    double *J,
    double *dJ,
    double *percoll,
    double *R,
    double *I,
    double *psi,
    double *psiPow0,
    double *psiPow1,
    double t,
    double *gradWing);
