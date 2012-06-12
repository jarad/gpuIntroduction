/*
 * smc_shared.h
 *
 *  Created on: 10-Mar-2009
 *      Author: alee
 */

#ifndef SMC_SHARED_H_
#define SMC_SHARED_H_

//void resample_get_indices_ref(float* cumw, int N, float* randu, int* indices,
//		float sumw);
//
//void resample_ref(float* x, int N, int* indices, float* xt_copy);
//
//void resample_ref(float* x, int N, int D, int* indices, float* xt_copy);
//
//void history_identity_ref(int* history, int N);
//
//void historify_ref(float* x, int N, int T, int* history, float* xcopy);
//
//void historify_ref(float* x, int N, int D, int T, int* history, float* xcopy);

// stratified
template <class T>
inline void resample_get_indices_ref(T* cumw, int N, T* randu, int* indices, T sumw) {
    int i, j;
    j = 0;
    for (i = 0; i < N; i++) {
        T r = sumw * (i + randu[i]) / N;
        while (j < N - 1 && r > cumw[j]) {
            j++;
        }
        indices[i] = j;
    }
}

template <class T>
inline void resample_ref(T* x, int N, int* indices, T* xt_copy) {
    int i;
    for (i = 0; i < N; i++) {
        x[i] = xt_copy[indices[i]];
    }
}

template <class T>
inline void resample_ref(T* x, int N, int D, int* indices, T* xt_copy) {
    for (int i = 0; i < N; i++) {
        vector_set(vector_get(x, D, i), vector_get(xt_copy, D, indices[i]), D);
    }
}

inline void history_identity_ref(int* history, int N) {
    for (int i = 0; i < N; i ++) {
        history[i] = i;
    }
}

template <class T>
inline void historify_ref(T* x, int N, int total_time, int* history, float* xcopy) {
    int q;
    for (int i = 0; i < N; i++) {
        q = i;
        for (int j = total_time - 1; j >= 0; j--) {
            q = history[N * j + q];
            x[j * N + i] = xcopy[j * N + q];
        }
    }
}

template <class T>
inline void historify_ref(T* x, int N, int D, int total_time, int* history, T* xcopy) {
    int q;
    for (int i = 0; i < N; i++) {
        q = i;
        for (int j = total_time - 1; j >= 0; j--) {
            q = history[N * j + q];
            vector_set(vector_get(x, D, j * N + i), vector_get(xcopy, D, j * N + q), D);
        }
    }
}

#endif /* SMC_SHARED_H_ */
