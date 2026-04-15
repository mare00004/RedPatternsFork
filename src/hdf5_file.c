#include "hdf5_file.h"
#include "H5Apublic.h"
#include "H5Dpublic.h"
#include "H5Fpublic.h"
#include "H5Gpublic.h"
#include "H5Ipublic.h"
#include "H5Ppublic.h"
#include "H5Spublic.h"
#include "H5Tpublic.h"
#include "H5public.h"
#include "build_info.h"
#include "sim_types.h"
#include <hdf5.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Write an UNSIGNED INT attribute to an object.
 */
void writeU32Attr(hid_t loc_id, const char *name, unsigned int v) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc_id, name, H5T_STD_U32LE, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_UINT, &v);
    H5Aclose(attr);
    H5Sclose(space);
}

/*
 * Write a DOUBLE attribute to an object.
 */
void writeF64Attr(hid_t loc_id, const char *name, double value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc_id, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attr);
    H5Sclose(space);
}

/*
 * Write a fixed length STRING with `nchars` characters (withtout `\0`) attribute to an object.
 */
void writeFixedStrAttr(hid_t loc_id, const char *name, const char *value, size_t nchars) {
    // Create fixed size string attribute type
    hid_t type_id = H5Tcopy(H5T_C_S1);
    size_t str_len = nchars + 1; // Space for NULL terminator
    H5Tset_size(type_id, str_len);
    H5Tset_strpad(type_id, H5T_STR_NULLPAD);

    hid_t space_id = H5Screate(H5S_SCALAR);
    hid_t attr_id = H5Acreate2(loc_id, name, type_id, space_id, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_id, type_id, value);

    H5Aclose(attr_id);
    H5Tclose(type_id);
    H5Sclose(space_id);
}

void writeStrAttr(hid_t dset, const char *name, const char *s) {
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

hid_t createExtendableF64Dataset(
    hid_t loc_id,
    const char *name,
    int rank,
    const hsize_t *dims,
    const hsize_t *maxDims,
    const hsize_t *chunkDims) {
    hid_t space, dcpl, dset;

    space = H5Screate_simple(rank, dims, maxDims);
    dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, rank, chunkDims);
    dset = H5Dcreate2(loc_id, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);

    H5Pclose(dcpl);
    H5Sclose(space);
    return dset;
}

/*
 * Write the Simulation Configuration to the `file` mirroring the `SimConfig` tagged union.
 */
void writeConfig(hid_t file, const SimConfig *cfg) {
    hid_t g_config, g_run, g_model, g_variant, g_active;

    g_config = H5Gcreate2(file, "config", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    g_run = H5Gcreate2(g_config, "run", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    g_model = H5Gcreate2(g_config, "model", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    g_variant = H5Gcreate2(g_model, "variant", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* /config/run */
    writeU32Attr(g_run, "N", cfg->run.N);
    writeU32Attr(g_run, "NT", cfg->run.NT);
    writeF64Attr(g_run, "T", cfg->run.T);
    writeF64Attr(g_run, "DT", cfg->run.DT);
    writeF64Attr(g_run, "DZ", cfg->run.DZ);
    writeU32Attr(g_run, "NO", cfg->run.NO);
    writeF64Attr(g_run, "fineDZ", cfg->run.fineDZ);
    writeF64Attr(g_run, "sysL", cfg->run.sysL);

    /* /config/model */
    writeFixedStrAttr(g_model, "modelType", (cfg->model.modelType == CONV) ? "CONV" : "TAYL", 4);
    writeFixedStrAttr(g_model, "gradientType", (cfg->model.gradientType == LINEAR) ? "LINEAR" : "SIGMOID", 7);
    writeF64Attr(g_model, "U", cfg->model.U);
    writeF64Attr(g_model, "PSI", cfg->model.PSI);
    writeF64Attr(g_model, "alpha", cfg->model.alpha);
    writeF64Attr(g_model, "beta", cfg->model.beta);
    writeF64Attr(g_model, "gamma", cfg->model.gamma);
    writeF64Attr(g_model, "delta", cfg->model.delta);
    writeF64Attr(g_model, "kappa", cfg->model.kappa);

    if (cfg->model.modelType == CONV) {
        g_active = H5Gcreate2(g_variant, "conv", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        writeU32Attr(g_active, "kernelN", cfg->model.variant.Conv.kernelN);
        writeU32Attr(g_active, "subDiv", cfg->model.variant.Conv.subDiv);
        writeU32Attr(g_active, "M", cfg->model.variant.Conv.M);
        writeStrAttr(g_active, "kernelSource", strlen(cfg->model.variant.Conv.kernelFile) > 0 ? cfg->model.variant.Conv.kernelFile : "internal");
    } else {
        g_active = H5Gcreate2(g_variant, "tayl", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        writeF64Attr(g_active, "NU", cfg->model.variant.Tayl.NU);
        writeF64Attr(g_active, "MU", cfg->model.variant.Tayl.MU);
    }

    H5Gclose(g_active);
    H5Gclose(g_variant);
    H5Gclose(g_model);
    H5Gclose(g_run);
    H5Gclose(g_config);
}

hid_t writeF64Vec(hid_t loc_id, const char *name, hsize_t n, const double *buf) {
    hid_t space, dset;
    hsize_t dims[1] = { n };

    space = H5Screate_simple(1, dims, NULL);
    dset = H5Dcreate2(loc_id, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

    H5Sclose(space);

    return dset;
}

int loadConvKernelFile(const char *path, double **kernelValues, int *kernelN) {
    hid_t file = -1;
    hid_t dset = -1;
    hid_t space = -1;
    hsize_t dims[1] = { 0 };
    double *values = NULL;
    int status = -1;

    *kernelValues = NULL;
    *kernelN = 0;

    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        goto cleanup;
    }

    dset = H5Dopen2(file, "/kernel/values", H5P_DEFAULT);
    if (dset < 0) {
        goto cleanup;
    }

    space = H5Dget_space(dset);
    if (space < 0 || H5Sget_simple_extent_ndims(space) != 1) {
        goto cleanup;
    }

    if (H5Sget_simple_extent_dims(space, dims, NULL) < 0 || dims[0] == 0 || (dims[0] % 2) == 0) {
        goto cleanup;
    }

    values = (double *)malloc((size_t)dims[0] * sizeof(double));
    if (values == NULL) {
        goto cleanup;
    }

    if (H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values) < 0) {
        goto cleanup;
    }

    *kernelValues = values;
    *kernelN = (int)dims[0];
    values = NULL;
    status = 0;

cleanup:
    if (values != NULL) {
        free(values);
    }
    if (space >= 0) {
        H5Sclose(space);
    }
    if (dset >= 0) {
        H5Dclose(dset);
    }
    if (file >= 0) {
        H5Fclose(file);
    }

    return status;
}

void writeConvKernelData(hid_t file, const SimConfig *cfg, const double *convKernel, int convKernelN) {
    hid_t g_conv;
    hid_t dset;

    if (cfg->model.modelType != CONV || convKernel == NULL || convKernelN <= 0) {
        return;
    }

    g_conv = H5Gopen2(file, "/config/model/variant/conv", H5P_DEFAULT);
    if (g_conv < 0) {
        return;
    }

    dset = writeF64Vec(g_conv, "kernel_values", (hsize_t)convKernelN, convKernel);
    H5Dclose(dset);
    H5Gclose(g_conv);
}

/*
 * Create the hdf5 file with metadata, configuration data and fixed datasets, as well as the extendible datasets.
 */
int ts_create(
    TSWriter *w,
    const char *path,
    const SimConfig *cfg,
    const double *rho,
    const double *z,
    const double *convKernel,
    int convKernelN) {
    int N = cfg->run.N;

    // Initialize Write
    memset(w, 0, sizeof(*w));
    w->t = 0;
    w->N = N;

    w->file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (w->file < 0) {
        return -1;
    }

    writeStrAttr(w->file, "git_commit", RP_BUILD_GIT_COMMIT);
    writeStrAttr(w->file, "git_describe", RP_BUILD_GIT_DESCRIBE);

    /* /config */
    writeConfig(w->file, cfg);
    writeConvKernelData(w->file, cfg, convKernel, convKernelN);

    /* /time (extendable 1-D float64) */
    w->dsetTime = createExtendableF64Dataset(w->file, "time", 1, (hsize_t[]){ 0 }, (hsize_t[]){ H5S_UNLIMITED }, (hsize_t[]){ 1024 });
    writeStrAttr(w->dsetTime, "long_name", "time since simulation start");
    writeStrAttr(w->dsetTime, "units", "s");

    /* /cords */
    hid_t g_coords = H5Gcreate2(w->file, "coords", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* /cords/rho */
    hid_t rho_ds = writeF64Vec(g_coords, "rho", N, rho);
    writeStrAttr(rho_ds, "long_name", "density");
    writeStrAttr(rho_ds, "units", "?"); // TODO:
    writeU32Attr(rho_ds, "N", (unsigned int)N);
    H5Dclose(rho_ds);

    hid_t z_ds = writeF64Vec(g_coords, "z", N, z);
    writeStrAttr(z_ds, "long_name", "height in tube");
    writeStrAttr(z_ds, "units", "m");
    writeU32Attr(z_ds, "N", (unsigned int)N);
    H5Dclose(z_ds);

    H5Gclose(g_coords);

    /* /fields */
    hid_t g_fields = H5Gcreate2(w->file, "fields", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* /fields/phi -> extendable along T (T, N, N) of float64 */
    w->dsetPhi = createExtendableF64Dataset(g_fields, "phi", 3, (hsize_t[]){ 0, N, N }, (hsize_t[]){ H5S_UNLIMITED, N, N }, (hsize_t[]){ 1, N, N });
    writeStrAttr(w->dsetPhi, "long_name", "time series of specific volume fraction of RBCs");
    writeStrAttr(w->dsetPhi, "units", "volume fractoin (unit-less)");
    writeStrAttr(w->dsetPhi, "coordinates", "time rho z");
    writeStrAttr(w->dsetPhi, "storage_order", "phi[i*N + j] = phi(rho_i, z_j)");

    /* /fields/psi -> extendable along T (T, N) of float64 */
    w->dsetPsi = createExtendableF64Dataset(g_fields, "psi", 2, (hsize_t[]){ 0, N }, (hsize_t[]){ H5S_UNLIMITED, N }, (hsize_t[]){ 1, N });
    writeStrAttr(w->dsetPsi, "long_name", "time series of total volume fraction of RBCs");
    writeStrAttr(w->dsetPsi, "units", "total volume fraction (unit-less)");
    writeStrAttr(w->dsetPsi, "coordinates", "time z");

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

void ts_postRunInfo(TSWriter *w, double runTime) {
    writeU32Attr(w->file, "runtime", (unsigned int)runTime);
}

void ts_close(TSWriter *w) {
    if (w->dsetPhi > 0)
        H5Dclose(w->dsetPhi);
    if (w->dsetPsi > 0)
        H5Dclose(w->dsetPsi);
    if (w->dsetTime > 0)
        H5Dclose(w->dsetTime);
    if (w->file > 0)
        H5Fclose(w->file);
}

#ifdef __cplusplus
}
#endif
