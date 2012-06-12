/*
 * mix_gauss_normal.h
 *
 *  Created on: 26-Feb-2009
 *      Author: alee
 */

#ifndef MIX_GAUSS_NORMAL_H_
#define MIX_GAUSS_NORMAL_H_

#include <float.h>
#include <mix_gauss.h>

template <class T>
inline T mgu_pdfh(T* x, int N, T* mus, int k, T c1, T c2, T min, T max) {
    for (int i = 0; i < k; i++) {
        if (mus[i] < min || mus[i] > max) {
            return 0;
        }
    }
    return mix_gauss1_mus_pdfh(x, N, k, c1, c2, mus);
}

template <class T>
inline T mgu_mu_pdfh(T* mus, T* args, int k) {
    int N = (int) args[0];
    T* x = args + 1;
    T c1 = args[N + 1];
    T c2 = args[N + 2];
    T min = args[N + 3];
    T max = args[N + 4];
    return mgu_pdfh(x, N, mus, k, c1, c2, min, max);
}

template <class T>
inline T log_mgu_pdfh(T* x, int N, T* mus, int k, T c1, T c2, T min, T max) {
    for (int i = 0; i < k; i++) {
        if (mus[i] < min || mus[i] > max) {
            return -FLT_MAX;
        }
    }
    return log_mix_gauss1_mus_pdfh(x, N, k, c1, c2, mus);
}

template <class T>
inline T log_mgu_mu_pdfh(T* mus, T* args, int k) {
    int N = (int) args[0];
    T* x = args + 1;
    T c1 = args[N + 1];
    T c2 = args[N + 2];
    T min = args[N + 3];
    T max = args[N + 4];
    return log_mgu_pdfh(x, N, mus, k, c1, c2, min, max);
}

#endif /* MIX_GAUSS_NORMAL_H_ */
