#include "parameters.cuh"
#include <assert.h>
#include <cmath>
#include <fstream>
#include <iostream>

/* saving 2D arrays to disk */
void saveArrToDrive(double *f, char *outFileName) {
    const uint16_t sampleSkip = ceil(double(N) / 256.0f);
    std::ofstream ofs(outFileName);
    for (int i = 0; i < N; i += sampleSkip) {
        for (int j = 0; j < N; j += sampleSkip) {
            if (j > 0)
                ofs << "\t";
            ofs << f[i + N * j];
        }
        ofs << "\n";
    }
    ofs.close();
}
/* saving interpolation to disk */
void saveIntVecToDrive(double *f, char *outFileName) {
    const uint16_t sampleSkip = 1;
    std::ofstream ofs(outFileName);
    for (int i = 0; i < M; i += sampleSkip) {
        if (i > 0)
            ofs << "\t";
        ofs << f[i];
    }
    ofs << "\n";
    ofs.close();
}
/* saving vector to disk */
void saveVecToDrive(double *f, char *outFileName) {
    const uint16_t sampleSkip = ceil(double(N) / 256.0f);
    std::ofstream ofs(outFileName);
    for (int i = 0; i < N; i += sampleSkip) {
        if (i > 0)
            ofs << "\t";
        ofs << f[i];
    }
    ofs << "\n";
    ofs.close();
}
/* saving n-vector to disk */
void saveNVecToDrive(double *f, char *outFileName, int n) {
    std::ofstream ofs(outFileName);
    for (int i = 0; i < n; i += 1) {
        if (i > 0)
            ofs << "\t";
        ofs << f[i];
    }
    ofs << "\n";
    ofs.close();
}
