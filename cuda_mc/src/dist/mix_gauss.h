/*
 * mix_gauss.h
 *
 *  Created on: 20-Mar-2009
 *      Author: alee
 */

#ifndef MIX_GAUSS_H_
#define MIX_GAUSS_H_

#include <stdlib.h>
#include <math.h>
#include "gauss.h"
#include "rng.h"

template <class T>
inline void compute_ci1_ci2(T sigma, T r, T&c1, T& c2) {
    c1 = r / (sqrt(2*PI) * sigma);
    c2 = -1.0f / (2 * msquare(sigma));
}

template <class T>
inline T mix_gauss1_pdfh(T x, int k, T* mus, T* c1s,
        T* c2s) {

    T result = 0;
    for (int i = 0; i < k; i++) {
        result += gauss1_pdfh(x, c1s[i], c2s[i], mus[i]);
    }
    return result;
}

template <class T>
inline T mix_gauss1_pdfh(T x, T* args) {
    int k = (int) args[0];
    T* mus = args + 1;
    T* c1s = args + 1 + k;
    T* c2s = args + 1 + 2 * k;
    return mix_gauss1_pdfh(x, k, mus, c1s, c2s);
}

template <class T>
inline T log_mix_gauss1_pdfh(T x, int k, T* c1, T* c2,
        T *mus) {
    T* vals = (T*) malloc(k * sizeof(T));
    for (int i = 0; i < k; i++) {
        vals[i] = log_gauss1_pdfh(x, c1[i], c2[i], mus[i]);
    }
    T maxv = vals[0];
    int i_maxv = 0;
    for (int i = 1; i < k; i++) {
        if (vals[i]> maxv) {
            maxv = vals[i];
            i_maxv = i;
        }
    }
    T result = 1;
    for (int i = 0; i < k; i++) {
        if (i != i_maxv) {
            result += (float) exp(vals[i] - maxv);
        }
    }
    free (vals);
    return (T) (maxv + log(result));
}

template <class T>
inline T log_mix_gauss1_pdfh(T x, T* args) {
    int k = (int) args[0];
    T* mus = args + 1;
    T* c1s = args + 1 + k;
    T* c2s = args + 1 + 2 * k;
    return log_mix_gauss1_pdfh(x, k, c1s, c2s, mus);
}

// SAME WEIGHTS / SIGMAS

template <class T>
inline T mix_gauss1_mus_pdfh(T x, int k, T c1, T c2, T *mus) {
    T result = 0;
    for (int i = 0; i < k; i++) {
        result += gauss1_pdfh(x, c1, c2, mus[i]);
    }
    return result;
}

template <class T>
inline T mix_gauss1_mus_pdfh(T* x, int N, int k, T c1, T c2,
        T *mus) {
    T logr = 0;
    for (int i = 0; i < N; i++) {
        logr += log(mix_gauss1_mus_pdfh(x[i], k, c1, c2, mus));
    }
    return exp(logr);
}

template <class T>
inline T log_mix_gauss1_mus_pdfh(T x, int k, T c1, T c2,
        T *mus) {
    T* vals = (T*) malloc(k * sizeof(T));
    for (int i = 0; i < k; i++) {
        vals[i] = log_gauss1_pdfh(x, c1, c2, mus[i]);
    }
    T maxv = vals[0];
    int i_maxv = 0;
    for (int i = 1; i < k; i++) {
        if (vals[i] > maxv) {
            maxv = vals[i];
            i_maxv = i;
        }
    }
    T result = 1;
    for (int i = 0; i < k; i++) {
        if (i != i_maxv) {
            result += exp(vals[i] - maxv);
        }
    }
    free(vals);
    return maxv + log(result);
}

template <class T>
inline T log_mix_gauss1_mus_pdfh(T* x, int N, int k, T c1, T c2,
        T *mus) {
    T logr = 0;
    for (int i = 0; i < N; i++) {
        logr += log_mix_gauss1_mus_pdfh(x[i], k, c1, c2, mus);
    }
    return logr;
}

//T approx_log_mix_gauss1_mus_pdfh(T x, int k, T c1, T c2,
//      T* mus) {
//  T lp = log_gauss1_pdfh(x, c1, c2, mus[0]);
//  T max_lp = lp;
//  for (int i = 1; i < k; i++) {
//      lp = log_gauss1_pdfh(x, c1, c2, mus[i]);
//      max_lp = max(max_lp, lp);
//  }
//  return max_lp;
//}

template <class T>
inline void generate_mix_data(int k, T sigma, T* mus, T* array, int N) {
    int* indices = (int*) malloc(N * sizeof(int));
    T* steps = (T*) malloc(N * sizeof(T));
    populate_randIK(indices, N, k);
    populate_randn(steps, N);
    for (int i = 0; i < N; i++) {
        array[i] = mus[indices[i]] + steps[i] * sigma;
    }
    free(indices);
    free(steps);
}

#endif /* MIX_GAUSS_H_ */
