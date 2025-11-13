#include "hdf5_file.h"
#include "H5Apublic.h"
#include "H5Dpublic.h"
#include "H5Fpublic.h"
#include "H5Ipublic.h"
#include "H5Ppublic.h"
#include "H5Spublic.h"
#include "H5Tpublic.h"
#include "H5public.h"
#include <hdf5.h>
#include <string.h>

void put_u32_attr(hid_t dset, const char *name, unsigned int v) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(dset, name, H5T_STD_U32LE, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_UINT, &v);
    H5Aclose(attr);
    H5Sclose(space);
}

void put_str_attr(hid_t dset, const char *name, const char *s) {
    hid_t t = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, strlen(s) + 1);
    H5Tset_cset(t, H5T_CSET_UTF8);
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(dset, name, t, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, t, s);
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(t);
}

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

        put_str_attr(w->dsetTime, "long_name", "time since simulation start");
        put_str_attr(w->dsetTime, "units", "s");

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

        put_str_attr(w->dsetPhi, "long_name", "time series of specific volume fraction of RBCs");
        put_str_attr(w->dsetPhi, "units", "volume fractoin (unit-less)");
        put_str_attr(w->dsetPhi, "coordinates", "time rho z");
        put_str_attr(w->dsetPhi, "storage_order", "phi[i*N + j] = phi(rho_i, z_j)");

        H5Pclose(dcpl);
        H5Sclose(space);
    }

    // Create /psi (extendable along T (T, N) of float32)
    {
        int rank = 2;
        hsize_t dims[2] = { 0, N };
        hsize_t maxdims[2] = { H5S_UNLIMITED, N };
        hid_t space = H5Screate_simple(rank, dims, maxdims);

        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        hsize_t chunk[2] = { 1, N };
        status = H5Pset_chunk(dcpl, rank, chunk);

        w->dsetPsi = H5Dcreate2(w->file, "/psi", H5T_IEEE_F64LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);

        put_str_attr(w->dsetPsi, "long_name", "time series of total volume fraction of RBCs");
        put_str_attr(w->dsetPsi, "units", "total volume fraction (unit-less)");
        put_str_attr(w->dsetPsi, "coordinates", "time z");

        H5Pclose(dcpl);
        H5Sclose(space);
    }

    return 0;
}

int ts_writeR(TSWriter *w, double *R) {
    herr_t status;
    hid_t space = H5Screate_simple(1, (hsize_t[]){ w->N }, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    w->dsetR = H5Dcreate2(w->file, "rho", H5T_IEEE_F64LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);

    put_str_attr(w->dsetR, "long_name", "density");
    put_str_attr(w->dsetR, "units", "?"); // TODO
    put_u32_attr(w->dsetR, "N", (unsigned int)w->N);

    status = H5Dwrite(w->dsetR, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, R);
    status = H5Dclose(w->dsetR);
    status = H5Pclose(dcpl);
    status = H5Sclose(space);
    return 0;
}

int ts_writeZ(TSWriter *w, double *Z) {
    herr_t status;
    hid_t space = H5Screate_simple(1, (hsize_t[]){ w->N }, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    w->dsetZ = H5Dcreate2(w->file, "z", H5T_IEEE_F64LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);

    put_str_attr(w->dsetZ, "long_name", "height in tube");
    put_str_attr(w->dsetZ, "units", "?"); // TODO
    put_u32_attr(w->dsetZ, "N", (unsigned int)w->N);

    status = H5Dwrite(w->dsetZ, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Z);
    status = H5Dclose(w->dsetZ);
    status = H5Pclose(dcpl);
    status = H5Sclose(space);
    return 0;
}

int ts_append(TSWriter *w, double t, const double *phi, const double *psi) {
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

    // Append one psi N array
    {
        hsize_t newSize[2] = { w->t + 1, N };
        if (H5Dset_extent(w->dsetPsi, newSize) < 0)
            return -1;

        hid_t fspace = H5Dget_space(w->dsetPsi);
        hsize_t start[2] = { w->t, 0 };
        hsize_t count[2] = { 1, N };
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);

        hsize_t mdims[2] = { 1, N };
        hid_t mspace = H5Screate_simple(2, mdims, NULL);

        if (H5Dwrite(w->dsetPsi, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, psi)) {
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
