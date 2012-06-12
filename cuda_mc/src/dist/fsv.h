/*
 * fsv.h
 *
 *  Created on: 07-Mar-2009
 *      Author: alee
 */

#ifndef FSV_H_
#define FSV_H_

#include <stdlib.h>
#include "gauss.h"

//void generate_data_fsv(float* xs_real, float* ys_real, int Dx, int Dy, int T,
//		float* scale_step, float* cov_step, float* cov_like, float* B);

template <class T>
void generate_data_fsv(T* xs, T* ys, int Dx, int Dy, int total_time,
        T* scale_step, T* cov_step, T* Psi, T* B) {

    int Mx = max(total_time * Dx, 32768);
    int My = max(total_time * Dy, 32768);

    T* steps_x = (T*) malloc(Mx * sizeof(T));
    T* steps_y = (T*) malloc(My * sizeof(T));

    T* L_step = (T*) malloc(Dx * Dx * sizeof(T));
    T* L_like = (T*) malloc(Dy * Dy * sizeof(T));
    T* Bt = (T*) malloc(Dx * Dy * sizeof(T));
    T* BH = (T*) malloc(Dy * Dx * sizeof(T));
    T* cov_like = (T*) malloc(Dy * Dy * sizeof(T));
    T* temp = (T*) malloc(Dx * sizeof(T));
    T* H = (T*) malloc(Dx * sizeof(T));

    matrix_transpose(B, Bt, Dy, Dx);

    matrix_chol(cov_step, L_step, Dx);

    populate_randn(steps_x, Mx);
    populate_randn(steps_y, My);

    matrix_times(L_step, steps_x, temp, Dx, Dx, Dx, 1);
    vector_set(xs, temp, Dx);

    for (int i = 1; i < total_time; i++) {
        matrix_times(scale_step, vector_get(xs, Dx, i - 1), vector_get(xs, Dx,
                i), Dx, Dx, Dx, 1);
        matrix_times(L_step, vector_get(steps_x, Dx, i), temp, Dx, Dx, Dx, 1);
        vector_add(vector_get(xs, Dx, i), temp, vector_get(xs, Dx, i), Dx);
    }

    for (int i = 0; i < total_time; i++) {
        vector_exp(vector_get(xs, Dx, i), H, Dx);
        matrix_times_diag(B, H, BH, Dy, Dx);
        matrix_times(BH, Bt, cov_like, Dy, Dx, Dx, Dy);
        matrix_add(cov_like, Psi, cov_like, Dy, Dy);

        matrix_chol(cov_like, L_like, Dy);

        matrix_times(L_like, vector_get(steps_y, Dy, i), vector_get(ys, Dy, i),
                Dy, Dy, Dy, 1);
    }

    free(L_step);
    free(L_like);
    free(Bt);
    free(BH);
    free(cov_like);
    free(temp);
    free(H);
    free(steps_x);
    free(steps_y);
}

template <class T>
T fsv_pdfh(T* x, T* y, T* args, int Dx, int Dy) {
    T* temp = (T*) malloc(Dy * Dx * sizeof(T));
    T* cov = (T*) malloc(Dy * Dy * sizeof(T));
    // B is Dy x Dx
    T* B = args;
    T* Bt = args + Dx * Dy;
    // Psi is Dy * Dy
    T* Psi = args + 2 * Dx * Dy;
    T* H = (T*) malloc(Dx * sizeof(T));
    vector_exp(x, H, Dx);
    T* mu = (T*) malloc(Dy * sizeof(T));
    vector_zero(mu, Dy);

    matrix_times_diag(B, H, temp, Dy, Dx);

    matrix_times(temp, Bt, cov, Dy, Dx, Dx, Dy);

    matrix_add(cov, Psi, cov, Dy, Dy);

    T r = gauss_pdfh(y, mu, cov, Dy);

    free(temp);
    free(cov);
    free(H);
    free(mu);

    return r;
}

#endif /* FSV_H_ */
