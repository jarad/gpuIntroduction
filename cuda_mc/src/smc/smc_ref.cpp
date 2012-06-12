/*
 * smc_ref.c
 *
 *  Created on: 02-Mar-2009
 *      Author: alee
 */

#include <stdio.h>
#include <stdlib.h>
#include "gauss.h"
#include "rng.h"
#include "smc_shared.h"
#include <math.h>

void FUNC( smc_step, TYPE)(
float* x, float* w, int N, float y, float scale_step, float* step, float* x_out, float* args) {
	int i;
	for (i = 0; i < N; i++) {
		x_out[i] = scale_step * x[i] + step[i];
		w[i] *= LIKELIHOOD_H(x_out[i], y, args);
	}
}

void FUNC( smc_ref, TYPE)(
float* x_init, float* x, float * w, float* y, int N, int T, float* h_args_l, float scale_step,
float sigma_step, float* ll) {

	float* cumw = (float*) malloc(N * sizeof(float));

	float* steps = (float*) malloc(N * sizeof(float));

	float* randu = (float*) malloc(N * sizeof(float));

	int* indices = (int*) malloc(N * sizeof(int));

	float* xt_copy = (float*) malloc(N * T * sizeof(float));

	int i, t;

	for (i = 0; i < N; i++) {
		w[i] = 1.0;
	}

	int* history = (int*) malloc(N * T * sizeof(int));

    int* history_id = (int*) malloc(N * sizeof(int));

    history_identity_ref(history_id, N);

    double old_sumw = (float) N;

	double lld = 0;

	for (t = 0; t < T; t++) {
		//		printf("%d\n", t);
		populate_randn(steps, N);
		populate_rand(randu, N);

		for (i = 0; i < N; i++) {
			steps[i] *= sigma_step;
		}

		FUNC(smc_step, TYPE)(x_init, w, N, y[t], scale_step, steps, x + t * N, h_args_l);

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

			cumw[0] = w[0];
			for (i = 1; i < N; i++) {
				cumw[i] = cumw[i - 1] + w[i];
			}

			resample_get_indices_ref(cumw, N, randu, indices, (float) sumw);

	         for (i = 0; i < N; i++) {
                history[t*N + i] = indices[i];
            }

			resample_ref(x_init, N, indices, x + t * N);

			for (i = 0; i < N; i++) {
				w[i] = 1.0;
			}

			old_sumw = (double) N;
		} else {
		    if (sumw < 1) {
                for (i = 0; i < N; i++) {
                    w[i] = (float) (w[i] * N / sumw);
                }
                old_sumw = (double) N;
            }
		    x_init = x + t * N;
		    for (i = 0; i < N; i++) {
                history[t*N + i] = history_id[i];
            }
		}

	}

	*ll = (float) lld;

	free(cumw);
	free(steps);
	free(randu);
	free(indices);
	free(xt_copy);
	free(history);
	free(history_id);

}

void FUNC( smc_ref_forget, TYPE)(
float* x_init, float* x, float* w, float* y, int N, int T, float* h_args_l, float scale_step,
float sigma_step, float* ll) {

	float* cumw = (float*) malloc(N * sizeof(float));

	float* steps = (float*) malloc(N * sizeof(float));

	float* randu = (float*) malloc(N * sizeof(float));

	int* indices = (int*) malloc(N * sizeof(int));

	int i, t;

	for (i = 0; i < N; i++) {
		w[i] = 1.0;
	}

	double old_sumw = (float) N;

	double lld = 0;

	for (t = 0; t < T; t++) {
		populate_randn(steps, N);
		populate_rand(randu, N);

		for (i = 0; i < N; i++) {
			steps[i] *= sigma_step;
		}

		FUNC(smc_step, TYPE)(x_init, w, N, y[t], scale_step, steps, x, h_args_l);

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

			cumw[0] = w[0];
			for (i = 1; i < N; i++) {
				cumw[i] = cumw[i - 1] + w[i];
			}

			resample_get_indices_ref(cumw, N, randu, indices, (float) sumw);

			resample_ref(x_init, N, indices, x);

			for (i = 0; i < N; i++) {
				w[i] = 1.0;
			}
			old_sumw = (double) N;

		} else {
		    if (sumw < 1) {
                for (i = 0; i < N; i++) {
                    w[i] = (float) (w[i] * N / sumw);
                }
                old_sumw = (double) N;
            }
		    for (i = 0; i < N; i++) {
                x_init[i] = x[i];
            }
		}
	}

	*ll = (float) lld;

	free(cumw);
	free(steps);
	free(randu);
	free(indices);

}
