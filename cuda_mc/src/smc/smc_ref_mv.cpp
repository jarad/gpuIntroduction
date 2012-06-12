/*
 * smc_ref_mv.cu
 *
 *  Created on: 10-Mar-2009
 *      Author: alee
 */

#ifndef SMC_REF_MV_CU_
#define SMC_REF_MV_CU_

#include <stdio.h>
#include <stdlib.h>
#include "rng.h"
#include "smc_shared.h"
#include "matrix.h"
#include "scan.h"

template <class T>
void FUNC( smc_step, TYPE)(
T* x, T* w, int N, int Dx, int Dy,
T* y, T* scale_step, T* step, T* x_out, int t, T* args) {
	T* x_o;
	T* x_i;
	T* yt = vector_get(y, Dy, t);

	for (int i = 0; i < N; i++) {
		x_o = vector_get(x_out, Dx, i);
		x_i = vector_get(x, Dx, i);

		matrix_times(scale_step, x_i, x_o, Dx, Dx, Dx, 1);
		vector_add(vector_get(step, Dx, i), x_o, x_o, Dx);

		w[i] *= LIKELIHOOD_H(x_o, yt, args, Dx, Dy);
	}
}

template <class T>
void FUNC( smc_ref, TYPE)(
T* x_init, T* x, T* w, T* y, int N, int Dx, int Dy,
int total_time, T* h_args_l, T* scale_step, T* cov_step, T* ll) {

	T* cumw = (T*) malloc(N * sizeof(T));

	T* steps = (T*) malloc(N * Dx * sizeof(T));

	T* randn = (T*) malloc(N * Dx * sizeof(T));

	T* randu = (T*) malloc(N * sizeof(T));

	int* indices = (int*) malloc(N * sizeof(int));

	T* xt_copy = (T*) malloc(N * Dx * total_time * sizeof(T));

	int i, t;

	for (i = 0; i < N; i++) {
		w[i] = 1.0;
	}

	int* history = (int*) malloc(N * total_time * sizeof(int));

    int* history_id = (int*) malloc(N * sizeof(int));

    history_identity_ref(history_id, N);

	T* L_step = (T*) malloc(Dx * Dx * sizeof(T));
	matrix_chol(cov_step, L_step, Dx);

	double old_sumw = (double) N;

	double lld = 0;

	for (t = 0; t < total_time; t++) {
		//		printf("%d\n", t);
		populate_randn(randn, N * Dx);
		populate_rand(randu, N);
		for (i = 0; i < N; i++) {
			matrix_times(L_step, vector_get(randn, Dx, i), vector_get(steps, Dx, i), Dx, Dx, Dx, 1);
		}

		FUNC(smc_step, TYPE)(x_init, w, N, Dx, Dy, y, scale_step, steps, x + t * Dx * N, t, h_args_l);

		double sumw = 0;
		double sumw2 = 0;
		for (i = 0; i < N; i++) {
			sumw += w[i];
			sumw2 += w[i] * w[i];
		}

		lld += log(sumw / old_sumw);

		old_sumw = sumw;

		double ESS = sumw * sumw / sumw2;

		if (ESS < N / 2) {

		    scan_ref(N, w, cumw);

			resample_get_indices_ref(cumw, N, randu, indices, (T) sumw);

			for (i = 0; i < N; i++) {
			    history[t*N + i] = indices[i];
			}

			resample_ref(x_init, N, Dx, indices, x + N * Dx * t);

			for (i = 0; i < N; i++) {
				w[i] = 1.0;
			}

			old_sumw = (double) N;
		} else {
		    if (sumw < 1) {
                for (i = 0; i < N; i++) {
                    w[i] = (T) (w[i] * N / sumw);
                }
                old_sumw = (double) N;
		    }
		    x_init = x + t * Dx * N;
		    for (i = 0; i < N; i++) {
		        history[t*N + i] = history_id[i];
		    }
		}

	}

	*ll = (T) lld;

	for (i = 0; i < N * Dx * total_time; i++) {
	    xt_copy[i] = x[i];
	}
    historify_ref(x, N, Dx, total_time, history, xt_copy);

	free(randn);
	free(cumw);
	free(steps);
	free(randu);
	free(indices);
	free(xt_copy);
	free(L_step);
	free(history);
	free(history_id);

}

template <class T>
void FUNC( smc_ref_forget, TYPE)(
T* x_init, T* x, T* w, T* y, int N, int Dx, int Dy,
int total_time, T* h_args_l, T* scale_step, T* cov_step, T* ll) {

	T* cumw = (T*) malloc(N * sizeof(T));

	T* steps = (T*) malloc(N * Dx * sizeof(T));

	T* randn = (T*) malloc(N * Dx * sizeof(T));

	T* randu = (T*) malloc(N * sizeof(T));

	int* indices = (int*) malloc(N * sizeof(int));

	int i, t;

	for (i = 0; i < N; i++) {
		w[i] = 1.0;
	}

	T* L_step = (T*) malloc(Dx * Dx * sizeof(T));
	matrix_chol(cov_step, L_step, Dx);

	double old_sumw = (T) N;

	double lld = 0;

	for (t = 0; t < total_time; t++) {
		//		printf("%d\n", t);
		populate_randn(randn, N * Dx);
		populate_rand(randu, N);
		for (i = 0; i < N; i++) {
			matrix_times(L_step, vector_get(randn, Dx, i), vector_get(steps, Dx, i), Dx, Dx, Dx, 1);
		}

		FUNC(smc_step, TYPE)(x_init, w, N, Dx, Dy, y, scale_step, steps, x, t, h_args_l);

		double sumw = 0;
		double sumw2 = 0;
		for (i = 0; i < N; i++) {
			sumw += w[i];
			sumw2 += w[i] * w[i];
		}

		lld += log(sumw / old_sumw);

		old_sumw = sumw;

		double ESS = sumw * sumw / sumw2;

		if (ESS < N / 2) {

		    scan_ref(N, w, cumw);

			resample_get_indices_ref(cumw, N, randu, indices, (T) sumw);

			resample_ref(x_init, N, Dx, indices, x);

			for (i = 0; i < N; i++) {
				w[i] = 1.0;
			}
			old_sumw = (double) N;

		} else {
		    if (sumw < 1) {
                for (i = 0; i < N; i++) {
                    w[i] = (T) (w[i] * N / sumw);
                }
                old_sumw = (double) N;
            }
		    for (i = 0; i < N * Dx; i++) {
                x_init[i] = x[i];
            }
		}

	}

	*ll = (T) lld;

	free(cumw);
	free(steps);
	free(randu);
	free(randn);
	free(indices);
	free(L_step);

}

#endif /* SMC_REF_MV_CU_ */
