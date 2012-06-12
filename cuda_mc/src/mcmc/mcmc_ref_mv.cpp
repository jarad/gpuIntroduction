/*
 * mcmc_ref_mv.cu
 *
 *  Created on: 15-Mar-2009
 *      Author: alee
 */

//#include <cutil.h>

#include "matrix.h"
#include "rng.h"
#include "temper.h"

void FUNC(metropolis_rw_ref, TYPE)(
int N, int D, float* init, float sigma, float* array_out, float* args_p, int log) {
    int i;
    float* x = init;
    float* y = (float*) malloc(D * sizeof(float));
    float* w;
    float ratio;

    float* array_uniform = (float*) malloc(N * sizeof(float));
    populate_rand(array_uniform, N);

    float* array_step = (float*) malloc(N * D * sizeof(float));
    populate_randn(array_step, N * D);
    if (sigma != 1.0) {
        for (i = 0; i < N * D; i++) {
            array_step[i] *= sigma;
        }
    }

    for (i = 0; i < N; i++) {
        w = vector_get(array_step, D, i);
        vector_add(x, w, y, D);
        if (log == 0) {
            ratio = TARGET_H(y, args_p, D) / TARGET_H(x, args_p, D);
        } else {
            ratio = (float) (exp(LOG_TARGET_H(y, args_p, D) - LOG_TARGET_H(x, args_p, D)));
        }
        if (array_uniform[i] < ratio) {
            vector_set(x, y, D);
        }
        vector_set(vector_get(array_out, D, i), x, D);
    }

    free(y);
    free(array_uniform);
    free(array_step);

}

void FUNC(metropolis_rwpop_marginal_ref, TYPE)(
int N, int D, float* array_init, float sigma, float* h_args_p, float* temps, float* array_out, int log, int numChains) {

    //    unsigned int hTimer;
    //    double time;
    //    cutCreateTimer(&hTimer);

    //    cutResetTimer(hTimer);
    //    cutStartTimer(hTimer);
    //
    //    cutStopTimer(hTimer);
    //    time = cutGetTimerValue(hTimer);
    //    printf("Time = %f\n", time);

    //    cutResetTimer(hTimer);
    //    cutStartTimer(hTimer);

    int numSteps = N / numChains;

    int* array_types = (int*) malloc(numSteps * sizeof(int));

    populate_randIK(array_types, numSteps, 2);

    //	for (int i = 0; i < numSteps; i++) {
    //		printf("%d ", array_types[i]);
    //	}
    //	printf("\n");

    float* array_step = (float*) malloc(N * D * sizeof(float));

    populate_randn(array_step, N * D);

    if (sigma != 1.0) {
        for (int i = 0; i < N * D; i++) {
            array_step[i] *= sigma;
        }
    }

    float* array_uniform1 = (float*) malloc(N * sizeof(float));
    float* array_uniform2 = (float*) malloc(N * sizeof(float));
    populate_rand(array_uniform1, N);
    populate_rand(array_uniform2, N);

    float* array_temp = (float*) malloc(numChains * D * sizeof(float));

    float* y = (float*) malloc(D * sizeof(float));

    //    cutStopTimer(hTimer);
    //    time = cutGetTimerValue(hTimer);
    //    printf("Setup Time = %f\n", time);

    float* densities = (float*) malloc(numChains * sizeof(float));
    for (int i = 0; i < numChains; i++) {
        float* x =  vector_get(array_init, D, i);
        if (log == 0) {
            densities[i] = TARGET_H(x, h_args_p, D);
        } else {
            densities[i] = LOG_TARGET_H(x, h_args_p, D);
        }
    }

    for (int i = 0; i < numSteps; i++) {

        //		if (i % 100 == 0) {
        //			printf("Step %d\n", i);
        //		}

        //        cutResetTimer(hTimer);
        //        cutStartTimer(hTimer);

        // STEPS
        for (int j = 0; j < numChains; j++) {
            float t = temps[j];
            float* x = vector_get(array_init, D, j);
            float* w = vector_get(array_step, D, i * numChains + j);
            vector_add(x, w, y, D);

            float ratio;
            float ty;
            float tx = densities[j];

            if (log) {
                ty = LOG_TARGET_H(y, h_args_p, D);
                ratio = (float) exp((ty - tx) * t);
            } else {
                ty = TARGET_H(y, h_args_p, D);
                ratio = temperh(ty / tx, t);
            }
            if (array_uniform1[i * numChains + j] < ratio) {
                densities[j] = ty;
                vector_set(vector_get(array_temp, D, j), y, D);
            } else {
                vector_set(vector_get(array_temp, D, j), x, D);
            }
        }

        //        cutStopTimer(hTimer);
        //        time = cutGetTimerValue(hTimer);
        //        printf("Step Time = %f\n", time);

        // EXCHANGE

        int start = array_types[i];

        //        cutResetTimer(hTimer);
        //        cutStartTimer(hTimer);

        // 0-1, 2-3, 4-5, etc.
        for (int j = start; j < numChains; j += 2) {
            int oj = (j + 1) % numChains;
            float* p1 = vector_get(array_temp, D, j);
            float* p2 = vector_get(array_temp, D, oj);

            float t = temps[j];
            float t2 = temps[oj];

            float ratio;
            float tp2 = densities[oj];
            float tp1 = densities[j];
            if (log) {
                ratio = (float) exp(tp2 * (t - t2) + tp1 * (t2 - t));
            } else {
                ratio = temperh(tp2, t - t2) * temperh(tp1, t2 - t);
            }

            if (array_uniform2[i * numChains + j] < ratio) {
                densities[oj] = tp1;
                densities[j] = tp2;
                vector_swap(p1, p2, D);
            }
        }

        //        cutStopTimer(hTimer);
        //        time = cutGetTimerValue(hTimer);
        //        printf("Exchange Time = %f\n", time);

        // COPY

        for (int j = 0; j < numChains * D; j++) {
            array_init[j] = array_temp[j];
        }

        vector_set(vector_get(array_out, D, i), vector_get(array_temp, D, numChains - 1), D);

    }

    free(array_uniform1);
    free(array_uniform2);
    free(array_step);
    free(array_temp);
    free(array_types);
    free(y);
    free(densities);

}
