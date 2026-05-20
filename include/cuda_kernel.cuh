#pragma once

__global__ void CuKernelInte(double *phi, double *psi);
__global__ void CuKernelSplineCoeffs(
    const double *__restrict__ y,
    double *__restrict__ b,
    double *__restrict__ c,
    double *__restrict__ d,
    const int N);
__global__ void CuKernelSplineEvalCellCentered(
    const double *__restrict__ y,
    const double *__restrict__ b,
    const double *__restrict__ c,
    const double *__restrict__ d,
    double *__restrict__ y_intp,
    int N,
    int M,
    int subDiv);
__global__ void CuKernelDownSample(double *IIntp, double *I, int subDiv);
__global__ void CuKernelCellToFace(const double *__restrict__ I_cell, double *__restrict__ I_face);
__global__ void CuKernelTaylFace(
    const double *__restrict__ psi,
    double *__restrict__ I_face,
    double c1,
    double c3);
__global__ void CuKernelConv(double *psi, double *I, double *convKernel, int M, int kernelN, int subDiv);
__global__ void CuKernelInte(double *phi, double *psi);
__global__ void CuKernelComputeFluxFVM(
    const double *__restrict__ phi,
    double *__restrict__ J,
    const double *__restrict__ percoll,
    const double *__restrict__ R,
    const double *__restrict__ I_face,
    const double *__restrict__ gradWing);
__global__ void CuKernelUpdatePhiFVM(
    double *__restrict__ phi,
    const double *__restrict__ J);
