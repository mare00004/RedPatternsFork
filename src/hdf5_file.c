#include "hdf5_file.h"
#include "H5Dpublic.h"
#include "H5Fpublic.h"
#include "H5Ipublic.h"
#include "H5Ppublic.h"
#include "H5Spublic.h"
#include "H5Tpublic.h"
#include "H5public.h"
#include <hdf5.h>
#include <string.h>

int ts_open(TSWriter *w, const char *path, hsize_t N) {
    herr_t status;

    // Initialize Write
    memset(w, 0, sizeof(*w));
    w->N = N;
    w->t = 0;

    w->file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (w->file < 0)
        return -1;

    /* ATTRIBUTES */
    /*    TODO    */

    // Create /time (extendable 1-D float32)
    {
        int rank = 1;
        hsize_t dims[1] = { 0 };
        hsize_t maxdims[1] = { H5S_UNLIMITED };
        hid_t space = H5Screate_simple(rank, dims, maxdims);

        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        hsize_t chunk[1] = { 1024 };
        status = H5Pset_chunk(dcpl, 1, chunk);

        w->dsetTime = H5Dcreate2(w->file, "/time", H5T_IEEE_F64LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);

        H5Pclose(dcpl);
        H5Sclose(space);
    }

    // Create /phi (extendable along T (T, N, N) of float32)
    {
        int rank = 3;
        hsize_t dims[3] = { 0, N, N };
        hsize_t maxdims[3] = { H5S_UNLIMITED, N, N };
        hid_t space = H5Screate_simple(rank, dims, maxdims);

        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        hsize_t chunk[3] = { 1, N, N };
        status = H5Pset_chunk(dcpl, rank, chunk);

        w->dsetPhi = H5Dcreate2(w->file, "/phi", H5T_IEEE_F64LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);

        H5Pclose(dcpl);
        H5Sclose(space);
    }

    return 0;
}

int writeVec(TSWriter *w, hid_t *dset, const char *dset_name, double *data) {
    herr_t status;

    hid_t space = H5Screate_simple(1, (hsize_t[]){ w->N }, NULL);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    *dset = H5Dcreate2(w->file, dset_name, H5T_IEEE_F64LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);

    status = H5Dwrite(*dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    status = H5Dclose(*dset);
    status = H5Pclose(dcpl);
    status = H5Sclose(space);
    return 0;
}

int ts_writeR(TSWriter *w, double *R) {
    return writeVec(w, &w->dsetR, "rho", R);
}

int ts_writeZ(TSWriter *w, double *Z) {
    return writeVec(w, &w->dsetZ, "z", Z);
}

int ts_append(TSWriter *w, double t, const double *phi) {
    const hsize_t N = w->N;

    // Append one phi NxN array
    {
        hsize_t newSize[3] = { w->t + 1, N, N };
        if (H5Dset_extent(w->dsetPhi, newSize) < 0)
            return -1;

        hid_t fspace = H5Dget_space(w->dsetPhi);
        hsize_t start[3] = { w->t, 0, 0 };
        hsize_t count[3] = { 1, N, N };
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);

        hsize_t mdims[3] = { 1, N, N };
        hid_t mspace = H5Screate_simple(3, mdims, NULL);

        if (H5Dwrite(w->dsetPhi, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, phi)) {
            H5Sclose(mspace);
            H5Sclose(fspace);
            return -1;
        }
        H5Sclose(mspace);
        H5Sclose(fspace);
    }

    // Append one t value
    {
        hsize_t newSize[1] = { w->t + 1 };
        if (H5Dset_extent(w->dsetTime, newSize) < 0)
            return -1;

        hid_t fspace = H5Dget_space(w->dsetTime);
        hsize_t start[1] = { w->t };
        hsize_t count[1] = { 1 };
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);

        hsize_t mdims[1] = { 1 };
        hid_t mspace = H5Screate_simple(1, mdims, NULL);

        if (H5Dwrite(w->dsetTime, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, &t)) {
            H5Sclose(mspace);
            H5Sclose(fspace);
            return -1;
        }
        H5Sclose(mspace);
        H5Sclose(fspace);
    }

    w->t += 1;
    return 0;
}

void ts_close(TSWriter *w) {
    if (w->dsetPhi > 0)
        H5Dclose(w->dsetPhi);
    if (w->dsetTime > 0)
        H5Dclose(w->dsetTime);
    if (w->file > 0)
        H5Fclose(w->file);
}
