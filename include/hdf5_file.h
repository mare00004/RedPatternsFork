#ifndef HDF5_FILE_H
#define HDF5_FILE_H

#include "H5Ipublic.h"
#include "H5public.h"
#include "sim_types.h"
#include <hdf5.h>

#ifdef __cplusplus
extern "C" {
#endif

// [T]ime [S]eries Writer
typedef struct {
    hid_t file;        // .h5 file
    hid_t dsetTime;    // /time (time, )
    hid_t dsetPhi;     // /phi (time, N, N)
    hid_t dsetPsi;     // /psi (time, N)
    hid_t dsetPercoll; // /percoll (time, N)
    hsize_t N;         // Grid Size
    hsize_t t;         // Current Time Step [Idx]
} TSWriter;

int loadConvKernelFile(const char *path, double **kernelValues, int *kernelN);
int loadInitialPhiFile(const char *path, double **phiValues, int expectedN);
int ts_create(
    TSWriter *w,
    const char *path,
    const SimConfig *cfg,
    const double *rho,
    const double *z,
    const double *initialPhi,
    const double *convKernel,
    int convKernelN);
int ts_append(TSWriter *w, double t, StoreBitMap store, const double *phi, const double *psi, const double *percoll);

/*
 * Add additional information to the HDF5 File.
 *  `runTime` in seconds
 */
void ts_postRunInfo(TSWriter *w, double runTime);
void ts_close(TSWriter *w);

#ifdef __cplusplus
}
#endif

#endif // HDF5_FILE_H
