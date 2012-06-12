#ifndef GAUSS_CH_
#define GAUSS_CH_

#include "matrix.ch"
#include <float.h>
#include "gauss.h"
#include <math.h>

template<int D>
__device__ void d_compute_c1_c2(float* cov, float& c1, float* c2) {
    d_matrix_det_inv_pd<D> (cov, c1, c2);
    //    d_matrix_det_inv<D>(cov, c1, c2);
    c1 = 1.0f / (sqrtf(c1) * powf(2 * PI, D / 2.0f));
}

// 1D, precomputed constants
__device__ float gauss1_pdf(float x, float c1, float c2, float mean) {
    return c1 * expf(c2 * msquare((x-mean)));
}

// 1D, precomputed constants wrapper
__device__ float gauss1_pdf(float x, float* args) {
    return gauss1_pdf(x, args[0], args[1], args[2]);
}

// 1D, precomputed constants
__device__ float log_gauss1_pdf(float x, float c1, float c2, float mean) {
    return logf(c1) + c2 * msquare((x-mean));
}

// 1D, precomputed constants wrapper
__device__ float log_gauss1_pdf(float x, float* args) {
    return log_gauss1_pdf(x, args[0], args[1], args[2]);
}

// 1D
__device__ float gauss1_pdf(float x, float mu, float sigma) {
    return 1.0f / (sqrtf(2 * PI) * sigma) * expf(-1.0f / (2 * msquare(sigma)) * msquare(x - mu));
}

// multi-D, precomputed constants
template<int D>
__device__ float gauss_pdf(float* x, float c1, float* c2, float* mean) {
    float xmmu[D];
    d_matrix_minus(x, mean, xmmu, D, 1);
    return c1 * expf(-0.5f * d_matrix_xtmx<D> (c2, xmmu));
}

// multi-D, precomputed constants wrapper
template<int D>
__device__ float gauss_pdf(float* x, float* args) {
    return gauss_pdf<D> (x, args[0], args + 1, args + 1 + D * D);
}

// multi-D, precomputed constants
template<int D>
__device__ float log_gauss_pdf(float* x, float c1, float* c2, float* mean) {
    float xmmu[D];
    d_matrix_minus(x, mean, xmmu, D, 1);
    return logf(c1) - 0.5f * d_matrix_xtmx<D> (c2, xmmu);
}

// multi-D, precomputed constants wrapper
template<int D>
__device__ float log_gauss_pdf(float*x, float* args) {
    return log_gauss_pdf<D> (x, args[0], args + 1, args + 1 + D * D);
}

// multi-D, scale and compute with mean
template<int Dx, int Dy>
__device__ float gauss_unequal_pdf(float* x, float* y, float* args) {
    float realx[Dy];
    float* scale = args + 1 + Dy * Dy;
    d_matrix_times<Dx> (scale, x, realx, Dy, 1);
    return gauss_pdf<Dy> (realx, args[0], args + 1, y);
}

template<int Dx, int Dy>
__device__ float log_gauss_unequal_pdf(float* x, float* y, float* args) {
    float realx[Dy];
    float* scale = args + 1 + Dy * Dy;
    d_matrix_times<Dx> (scale, x, realx, Dy, 1);
    return log_gauss_pdf<Dy> (realx, args[0], args + 1, y);
}

// 1D with mean
__device__ float gauss1_mean_pdf(float x, float mu, float* args) {
    return gauss1_pdf(x, args[0], args[1], mu);
}

//// multi-D with mean
//template <int D>
//__device__ float gauss_mean_pdf(float* x, float* mu, float* args) {
//  return gauss_pdf<D>(x, args[0], args + 1, mu);
//}

// multi-D
template<int D>
__device__ float gauss_pdf(float* x, float* mu, float* cov) {
    float c1;
    float L[D * D];
    float w[D];
    float z[D];
    for (int i = 0; i < D; i++) {
        z[i] = x[i] - mu[i];
    }
    d_matrix_det_chol_pd<D>(cov, c1, L);
    c1 = (1.0f / (sqrtf(c1) * powf(2 * PI, D / 2.0f)));
    d_matrix_solve_pd(L, z, w, D);
    float v = d_vector_xty(z, w, D);
    return (c1 * expf(- 0.5 * v));
}

template<int D>
__device__ float log_gauss_pdf(float* x, float* mu, float* cov) {
    float c1;
    float c2[D * D];
    d_compute_c1_c2<D> (cov, c1, c2);
    float r = log_gauss_pdf<D> (x, c1, c2, mu);
    return r;
}

//// 1D, independent product
//__device__ float gauss1_pdf(float* x, int N, float c1, float c2, float mean) {
//  float logr = 0;
//  for (int i = 0; i < N; i++) {
//      logr += log_gauss1_pdf(x[i], c1, c2, mean);
//  }
//  return expf(logr);
//}

#endif /* GAUSS_CH_ */
