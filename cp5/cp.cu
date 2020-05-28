#include "cp.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <cstdlib>
#include <iostream>
#include <sys/time.h>

using namespace std;

inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

// static double get_time() {
//     struct timeval tm;
//     gettimeofday(&tm, NULL);
//     return static_cast<double>(tm.tv_sec)
//         + static_cast<double>(tm.tv_usec) / 1E6;
// }

#define CHECK(x) check(x, #x)

inline int static divup(int a, int b) {
    return (a + b - 1)/b;
}

__global__ void correlateKernel(int ny, int nx, int nny,  const float* transposed, float* result) {

    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    int id = ic * 64 + ia;
    int jd = jc * 64 + ja + 56;
    if (jd < id)
        return;

    float c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            c[i][j] = 0;
        }
    }

    for (int k = 0; k < nx; k++) {
        float x[8];
        float y[8];
        for (int ib = 0; ib < 8; ++ib) {
            int i = ic * 64 + ib * 8 + ia;
            x[ib] = transposed[nny * k + i];
        }
        for (int jb = 0; jb < 8; ++jb) {
            int j = jc * 64 + jb * 8 + ja;
            y[jb] = transposed[nny * k + j];
        }
        for (int ib = 0; ib < 8; ++ib) {
            for (int jb = 0; jb < 8; ++jb) {
                c[ib][jb] += x[ib] * y[jb];
            }
        }
    }

    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = ic * 64 + ib * 8 + ia;
            int j = jc * 64 + jb * 8 + ja;
            if (i < ny && j < ny) {
                result[ny * i + j] = c[ib][jb];
            }
        }
    }
}

__global__ void normalizeKernel(int ny, int nx, int nny, float* transposed) {
    int i = threadIdx.x;
    int j = blockIdx.x;

    float average = 0;
    for (int k = 0; k < nx; k++) {
        average += transposed[k * nny + j * 64 + i];
    }
    average /= nx;
    for (int k = 0; k < nx; k++) {
        transposed[k * nny + j * 64 + i] -= average;
    }

    float divisor = 0;
    for (int k = 0; k < nx; k++) {
        float t = transposed[k * nny + j * 64 + i]
        divisor += t * t;
    }
    divisor = sqrt(divisor);
    divisor = 1 / divisor;
    for (int k = 0; k < nx; k++) {
        transposed[k * nny + j * 64 + i] *= divisor;
    }
}

__global__ void transposeKernel(int ny, int nx, int nny, const float* data, float* transposed) {
    int i = threadIdx.x;
    int j = blockIdx.y;

    for (int k = 0; k < nx; k += 64) {
        int in = k + i;
        //float v = in < nx ? data[j * nx + in] : 0;
        if(in < nx)
            transposed[j + (nx - in - 1) * nny] = data[j * nx + in];
    }
}


void correlate(int ny, int nx, const float* data, float* result) {

    constexpr int m = 64;
    int n = (ny + m - 1) / m;
    int nny = n * m;

    //double t6 = get_time();
    float* dataGPU = NULL;
    float* transposedGPU = NULL;
    CHECK(cudaMalloc((void**)&dataGPU, nx * ny * sizeof(float)));
    CHECK(cudaMalloc((void**)&transposedGPU, nx * nny * sizeof(float)));
    CHECK(cudaMemcpy(dataGPU, data, nx * ny * sizeof(float), cudaMemcpyHostToDevice));
    //CHECK(cudaMemcpy(transposedGPU, transposed, nx * nny * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlockT(m, 1);
    dim3 dimGridT(1, ny);
    transposeKernel<<<dimGridT, dimBlockT>>>(ny, nx, nny, dataGPU, transposedGPU);
    CHECK(cudaGetLastError());

    dim3 dimBlockP(m, 1);
    dim3 dimGridP(nny / 64, 1);
    normalizeKernel<<<dimGridP, dimBlockP>>>(ny, nx, nny, transposedGPU);
    CHECK(cudaGetLastError());
    //double t7 = get_time();
    //cout << "\n\nCopying data to GPU and preprocessing takes: " << t7 - t6 << " seconds!\n";

    //double t0 = get_time();
    //Normalize the input matrix rows to have mean of 0
    // for (int i = 0; i < ny; i++) {
    //     double average = 0;
    //     for (int j = 0; j < nx; j++) {
    //         average += data[j + i * nx];
    //     }
    //     average /= nx;
    //     for (int j = 0; j < nx; j++) {
    //         normalized[j + i * nx] = data[j + i * nx] - average;
    //     }
    // }

    // //Then normalize it again to have sum of squares to 1
    // for (int i = 0; i < ny; i++) {
    //     double divisor = 0;
    //     for (int j = 0; j < nx; j++) {
    //         divisor += pow(normalized[j + i * nx], 2);
    //     }
    //     divisor = sqrt(divisor);
    //     divisor = 1 / divisor;
    //     for (int j = 0; j < nx; j++) {
    //         normalized[j + i * nx] *= divisor;
    //     }
    // }
    // //double t1 = get_time();
    // //cout << "\nPreprocessing takes: " << t1 - t0 << " seconds.\n";

    // for (int i = 0; i < ny; i++) {
    //     for (int j = 0; j < nx; j++) {
    //         transposed[i + (nx - j - 1) * nny] = normalized[j + i * nx];
    //     }
    // }

    //double t2 = get_time();
    //float* transposedGPU = NULL;
    float* resultGPU = NULL;
    //CHECK(cudaMalloc((void**)&transposedGPU, nny * nx * sizeof(float)));
    CHECK(cudaMalloc((void**)&resultGPU, ny * ny * sizeof(float)));
    //CHECK(cudaMemcpy(transposedGPU, transposed, nny * nx * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(8, 8);
    dim3 dimGrid(divup(ny, 64), divup(ny, 64));
    correlateKernel<<<dimGrid, dimBlock>>>(ny, nx, nny, transposedGPU, resultGPU);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dataGPU));
    CHECK(cudaFree(resultGPU));
    CHECK(cudaFree(transposedGPU));
    //double t3 = get_time();
    //cout << "\nCalculating correlation takes: " << t3 - t2 << " seconds.\n\n";

    //free(normalized);
    //free(transposed);
}
