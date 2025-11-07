#ifndef HDF5_FILE_H
#define HDF5_FILE_H

#include "H5Ipublic.h"
#include "H5public.h"
#include <hdf5.h>

#ifdef __cplusplus
extern "C" {
#endif

// [T]ime [S]eries Writer
typedef struct {
    hid_t file;     // .h5 file
    hid_t dsetTime; // /time (time, )
    hid_t dsetPhi;  // /phi (time, N, N)
    hid_t dsetZ;
    hid_t dsetR;
    hsize_t N; // Grid Size
    hsize_t t; // Current Time Step [Idx]
} TSWriter;

int ts_open(TSWriter *w, const char *path, hsize_t N);

int writeVec(TSWriter *w, hid_t *dset, const char *dset_name, double *data);

int ts_writeR(TSWriter *w, double *R);

int ts_writeZ(TSWriter *w, double *Z);

int ts_append(TSWriter *w, double t, const double *phi);

void ts_close(TSWriter *w);

#ifdef __cplusplus
}
#endif

#endif
