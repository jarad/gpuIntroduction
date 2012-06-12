/*
 * mcmc_ref.cu
 *
 *  Created on: 22-Mar-2009
 *      Author: alee
 */

#include "temper.h"
#include <math.h>
#include <stdlib.h>
#include "rng.h"

void FUNC(metropolis_rw_ref, TYPE)(
int N, float init, float sigma, float* array_out, float* args_p, int log) {
	int i;
	float x, y, w, ratio;
	x = init;

	float* array_uniform = (float*) malloc(N * sizeof(float));
	populate_rand(array_uniform, N);

	float* array_step = (float*) malloc(N * sizeof(float));
	populate_randn(array_step, N);
	if (sigma != 1.0) {
		for (i = 0; i < N; i++) {
			array_step[i] *= sigma;
		}
	}

	for (i = 0; i < N; i++) {
		w = array_step[i];
		y = x + w;
		if (log) {
			ratio = (float) exp(LOG_TARGET_H(y, args_p) - LOG_TARGET_H(x, args_p));
		} else {
			ratio = TARGET_H(y, args_p) / TARGET_H(x, args_p);
		}
		if (array_uniform[i] < ratio) {
			x = y;
		}
		array_out[i] = x;
	}

	free(array_uniform);
	free(array_step);

}

void FUNC(metropolis_rwpop_ref, TYPE)(
int N, float* init, float sigma, float* h_args_p, float* temps, float* h_array_out, int useLog, int nt) {

	float w,y,ratio;
	int M = N / nt;
	int i, j;

	float* array_uniform1 = (float*) malloc(N * sizeof(float));
	populate_rand(array_uniform1, N);
	float* array_uniform2 = (float*) malloc(N * sizeof(float));
	populate_rand(array_uniform2, N);

	float* array_normal = (float*) malloc(N * sizeof(float));
	populate_randn(array_normal, N);
	for (i = 0; i < N; i++) {
		array_normal[i] *= sigma;
	}

	int* array_types = (int*) malloc(M * sizeof(int));
	populate_randIK(array_types, M, 2);

	float* x = (float*) malloc(nt * sizeof(float));

	for (j = 0; j < nt; j++) {
		x[j] = init[j];
	}

	for (i = 0; i < M; i++) {
		for (j = 0; j < nt; j++) {
			float t = temps[j];

			w = array_normal[i*nt + j];
			y = x[j] + w;
			if (useLog) {
				ratio = (float) exp(LOG_TARGET_H(y, h_args_p) * t - LOG_TARGET_H(x[j], h_args_p) * t);
			} else {
				ratio = temperh(TARGET_H(y, h_args_p), t) / temperh(TARGET_H(x[j], h_args_p), t);
			}

			if (array_uniform1[i * nt + j] < ratio) {
				x[j] = y;
			}

			h_array_out[i * nt + j] = x[j];
		}

		int type = array_types[i];

		for (j = 0; j < nt; j++) {
			//			if ((type == 1 && j & 1 == 0) || (type == 0 && j & 1 == 1)) {
			if (j % 2 == type && j != nt - 1) {
				float t = temps[j];
				// 0-1, 2-3, 4-5, ...
				// 1-2, 3-4, 5-6, ...

				float t2 = temps[j + 1];
				y = x[j+1];
				if (useLog) {
					float ty = LOG_TARGET_H(y, h_args_p);
					float tx = LOG_TARGET_H(x[j], h_args_p);
					ratio = (float) exp(ty * (t - t2) + tx * (t2 - t));
				} else {
					float ty = TARGET_H(y, h_args_p);
					float tx = TARGET_H(x[j], h_args_p);
					ratio = temperh(ty, t - t2) * temperh(tx, t2 - t);
//					ratio = temperh(TARGET_H(y, h_args_p), t)
//					/ temperh(TARGET_H(y, h_args_p), t2)
//					* temperh(TARGET_H(x[j], h_args_p), t2)
//					/ temperh(TARGET_H(x[j], h_args_p), t);
				}
				if (array_uniform2[i] < ratio) {
					x[j+1] = x[j];
					x[j] = y;
				}

			}
		}

		for (j = 0; j < nt; j++) {
			h_array_out[i * nt + j] = x[j];
		}
	}

	free(array_types);
	free(array_normal);
	free(array_uniform1);
	free(array_uniform2);

}
