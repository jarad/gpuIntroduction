/*
 * smc_shared.cu
 *
 *  Created on: 10-Mar-2009
 *      Author: alee
 */

#include "smc_shared.ch"
#include "matrix.ch"
#include "matrix.h"
#include <stdio.h>

__global__ void resample_get_indices(float* cumw, int N, float* randu, int* indices, float sumw) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;

    int lo;
    int hi;
    int mid = 0;

    for (int i = tid; i < N; i += tt) {
        lo = mid;
        hi = N;
        //		float r = randu[i] * sumw;

        float r = sumw * (i + randu[i]) / N;

        float v;

        while (hi - lo > 1) {
            mid = (hi + lo) / 2;
            //			v = cumw[mid];
#ifdef __DEVICE_EMULATION__
            v = cumw[mid];
#else
            v = tex1Dfetch(tex_cw, mid);
#endif
            if (v < r) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (r <= cumw[lo] || lo == N - 1) {
            mid = lo;
        } else {
            mid = hi;
        }
        indices[i] = mid;
    }
}

//// SIMILAR SPEED
//__global__ void resample_get_indices(float* cumw, int N, float* randu, int* indices, float sumw) {
//    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
//    const int tt = blockDim.x * gridDim.x;
//
//    int lo = 0;
//    int hi = N;
//    int mid;
//
//    int M = N / tt;
//
//    // find my place
//    int i = tid * M;
//    float r = sumw * i / N;
//    float v;
//    while (hi - lo > 1) {
//        mid = (hi + lo) / 2;
//#ifdef __DEVICE_EMULATION__
//        v = cumw[mid];
//#else
//        v = tex1Dfetch(tex_cw, mid);
//#endif
//        if (v < r) {
//            lo = mid + 1;
//        } else {
//            hi = mid;
//        }
//    }
//
//    // go in order
//    for (i = tid * M; i < (tid + 1) * M; i++) {
//        r = sumw * (i + randu[i]) / N;
//
//        while (lo < N - 1 && r > tex1Dfetch(tex_cw, lo) ) {
//            lo++;
//        }
//        indices[i] = lo;
//    }
//}

__global__ void resample(float* x, int N, int* indices, float* xt_copy) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += tt) {
        x[i] = xt_copy[indices[i]];
    }
}

__global__ void resample(float* x, int N, int D, int* indices, float* xt_copy) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += tt) {
        d_vector_set(d_vector_get(x, D, i), d_vector_get(xt_copy, D, indices[i]), D);
    }
}

__global__ void history_identity(int* history, int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += tt) {
        history[i] = i;
    }
}

__global__ void historify(float* x, int N, int T, int* history, float* xcopy) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    int q;
    for (int i = tid; i < N; i += tt) {
        q = i;
        for (int j = T - 1; j >= 0; j--) {
            q = history[N * j + q];
            x[j * N + i] = xcopy[j * N + q];
        }
    }
}

__global__ void historify(float* x, int N, int D, int T, int* history, float* xcopy) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    int q;
    for (int i = tid; i < N; i += tt) {
        q = i;
        for (int j = T - 1; j >= 0; j--) {
            q = history[N * j + q];
            d_vector_set(d_vector_get(x, D, j * N + i), d_vector_get(xcopy, D, j * N + q), D);
        }
    }
}
