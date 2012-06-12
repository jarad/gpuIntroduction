/*
 * smc_shared_ref.cpp
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#include "matrix.h"
#include "smc_shared.h"

//// stratified
//void resample_get_indices_ref(float* cumw, int N, float* randu, int* indices, float sumw) {
//    int i, j;
//    j = 0;
//    for (i = 0; i < N; i++) {
//        float r = sumw * (i + randu[i]) / N;
//        while (j < N - 1 && r > cumw[j]) {
//            j++;
//        }
//        indices[i] = j;
//    }
//}
//
//void resample_ref(float* x, int N, int* indices, float* xt_copy) {
//    int i;
//    for (i = 0; i < N; i++) {
//        x[i] = xt_copy[indices[i]];
//
//    }
//}
//
//void resample_ref(float* x, int N, int D, int* indices, float* xt_copy) {
//    for (int i = 0; i < N; i++) {
//        vector_set(vector_get(x, D, i), vector_get(xt_copy, D, indices[i]), D);
//    }
//}
//
//void history_identity_ref(int* history, int N) {
//    for (int i = 0; i < N; i ++) {
//        history[i] = i;
//    }
//}
//
//void historify_ref(float* x, int N, int T, int* history, float* xcopy) {
//    int q;
//    for (int i = 0; i < N; i++) {
//        q = i;
//        for (int j = T - 1; j >= 0; j--) {
//            q = history[N * j + q];
//            x[j * N + i] = xcopy[j * N + q];
//        }
//    }
//}
//
//void historify_ref(float* x, int N, int D, int T, int* history, float* xcopy) {
//    int q;
//    for (int i = 0; i < N; i++) {
//        q = i;
//        for (int j = T - 1; j >= 0; j--) {
//            q = history[N * j + q];
//            vector_set(vector_get(x, D, j * N + i), vector_get(xcopy, D, j * N + q), D);
//        }
//    }
//}
