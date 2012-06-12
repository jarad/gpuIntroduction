/*
 * gauss.h
 *
 *  Created on: 20-Jan-2009
 *      Author: alee
 */

#ifndef GAUSS_H_
#define GAUSS_H_

#include <stdlib.h>
#include <math.h>
#include "matrix.h"

#define msquare(x) ((x) * (x))
#define PI 3.14159265358979f

//void compute_c1_c2(float sigma, float& c1, float& c2);
//void compute_c1_c2(float* cov, int D, float& c1, float* c2);
//
//float gauss1_pdfh(float x, float c1, float c2, float mean);
//float gauss1_pdfh(float x, float* args);
//
//float log_gauss1_pdfh(float x, float c1, float c2, float mean);
//float log_gauss1_pdfh(float x, float* args);
//
//float gauss1_pdfh(float x, float mu, float sigma);

//float gauss_pdfh(float* x, float c1, float* c2, float* mean, int D);
//float gauss_pdfh(float* x, float* args, int D);

//float log_gauss_pdfh(float*x, float c1, float* c2, float* mu, int D);
//float log_gauss_pdfh(float* x, float* args, int D);
//
//float gauss_unequal_pdfh(float* x, float* y, float* args, int Dx, int Dy);
//
//float gauss1_mean_pdfh(float x, float mu, float* args);

//float gauss_pdfh(float* x, float* mu, float* cov, int D);

// c1 = 1/sqrt(2*pi*sigma^2)
// c2 = 1/(2*sigma^2)
template <class T>
inline void compute_c1_c2(T sigma, T& c1, T& c2) {
    c1 = (T) (1.0 / (sqrt(2*PI) * sigma));
    c2 = (T) (-1.0 / (2.0 * msquare(sigma)));
}

template <class T>
inline void compute_c1_c2(T* cov, int D, T& c1, T* c2) {
    matrix_det_inv_pd(cov, c1, c2, D);
    c1 = (T) (1.0 / (sqrt(c1) * pow(2 * PI, D / 2.0f)));
//    c1 = 1.0f / (powf(2 * PI, D / 2.0f) * sqrtf(matrix_det(cov, D)));
//    matrix_inverse_pd(cov, c2, D);
}

// 1D, precomputed constants
template <class T>
inline T gauss1_pdfh(T x, T c1, T c2, T mean) {
    return (T) (c1 * exp(c2 * msquare((x-mean))));
}

// 1D, precomputed constants wrapper
template <class T>
inline T gauss1_pdfh(T x, T* args) {
    return gauss1_pdfh(x, args[0], args[1], args[2]);
}

// 1D, precomputed constants
template <class T>
inline T log_gauss1_pdfh(T x, T c1, T c2, T mean) {
    return (T) (log(c1) + c2 * msquare((x-mean)));
}

// 1D, precomputed constants wrapper
template <class T>
inline T log_gauss1_pdfh(T x, T* args) {
    return log_gauss1_pdfh(x, args[0], args[1], args[2]);
}

// 1D
template <class T>
inline T gauss1_pdfh(T x, T mu, T sigma) {
    return (T) (1.0 / (sqrt(2 * PI) * sigma) * exp(-1.0 / (2 * msquare(sigma)) * msquare(x - mu)));
}

// multi-D, precomputed constants
template <class T>
inline T log_gauss_pdfh(T*x, T c1, T* c2, T* mu, int D) {
    T* xmmu = (T*) malloc(D * sizeof(T));
    matrix_minus(x, mu, xmmu, D, 1);
    T r = (T) (log(c1) - 0.5 * matrix_xtmx(c2, xmmu, D));
    free(xmmu);
    return r;
}

// multi-D, precomputed constants wrapper
template <class T>
inline T log_gauss_pdfh(T*x, T* args, int D) {
    return log_gauss_pdfh(x, args[0], args + 1, args + 1 + D * D, D);
}

// multi-D, scale and compute with mean
template <class T>
inline T gauss_unequal_pdfh(T* x, T* y, T* args, int Dx, int Dy) {
    T* realx = (T*) malloc(Dy * sizeof(T));
    T* scale = args + 1 + Dy * Dy;
    matrix_times(scale, x, realx, Dy, Dx, Dx, 1);
    T r = gauss_pdfh(realx, args[0], args + 1, y, Dy);
    free(realx);
    return r;
}

// 1D with mean
template <class T>
inline T gauss1_mean_pdfh(T x, T mu, T* args) {
    return gauss1_pdfh(x, args[0], args[1], mu);
}



// multi-D, precomputed constants
template <class T>
inline T gauss_pdfh(T* x, T c1, T* c2, T* mean, int D) {
    T* xmmu = (T*) malloc(D * sizeof(T));
    matrix_minus(x, mean, xmmu, D, 1);
    T r = (T) (c1 * exp(((T) -0.5) * matrix_xtmx(c2, xmmu, D)));
    free(xmmu);
    return r;
}

// multi-D, precomputed constants wrapper
template <class T>
inline T gauss_pdfh(T* x, T* args, int D) {
    return gauss_pdfh(x, args[0], args + 1, args + 1 + D * D, D);
}

//template <class T>
//inline T gauss_pdfh(T* x, T* mu, T* cov, int D) {
//    T c1;
//    T* c2 = (T*) malloc(D * D * sizeof(T));
//    compute_c1_c2(cov, D, c1, c2);
//    T r = gauss_pdfh(x, c1, c2, mu, D);
//    free(c2);
//    return r;
//}

template <class T>
inline T gauss_pdfh(T* x, T* mu, T* cov, int D) {
    T c1;
    T* L = (T*) malloc(D * D * sizeof(T));
    T* w = (T*) malloc(D * sizeof(T));
    T* z = (T*) malloc(D * sizeof(T));
    for (int i = 0; i < D; i++) {
        z[i] = x[i] - mu[i];
    }
    matrix_det_chol_pd(cov, c1, L, D);
    c1 = (T) (1.0 / (sqrt(c1) * pow(2 * PI, D / 2.0f)));
    matrix_solve_pd(L, z, w, D);
    T v = vector_xty(z, w, D);
    T r = (T) (c1 * exp(((T) -0.5) * v));
    free(z);
    free(L);
    free(w);
    return r;
}

#endif /* GAUSS_H_ */
