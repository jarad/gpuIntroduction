/*
 * smcs_ref_mv.cu
 *
 *  Created on: 17-Mar-2009
 *      Author: Owner
 */

//unsigned int hTimer;
//double time;
//cutCreateTimer(&hTimer);
//cutResetTimer( hTimer);
//cutStartTimer( hTimer);
//cutStopTimer( hTimer);
//time = cutGetTimerValue(hTimer);
//printf("Time = %f\n", time);

#include <cutil.h>
#include "temper.h"
#include "smc_shared.h"
#include "matrix.h"
#include "rng.h"
#include "scan.h"

void FUNC(smcs_step, TYPE)(
float* x, float* w, int N, int D, float* step, int numSteps, float* randu, float* temps, float* x_out, float* args_t1, float* args_t2, int time) {

    //	float weight;
    float* x_o;
    float* x_i;

    float* y = (float*) malloc(D * sizeof(float));
    float* z = (float*) malloc(D * sizeof(float));
    float temp;
    float temp_prev;

    for (int i = 0; i < N; i ++) {

        x_o = vector_get(x_out, D, i);
        x_i = vector_get(x, D, i);

        temp = temps[time];
        if (time == 0) {
            temp_prev = 0;
        } else {
            temp_prev = temps[time - 1];
        }

        vector_set(y, x_i, D);

        float tz = 0;
        float ty = 0;
        float new_weight = 0;
//        if (log) {
            float t1 = LOG_TARGET1_H(y, args_t1, D);
            float t2 = LOG_TARGET2_H(y, args_t2, D);
            ty = t1 * (1 - temp) + t2 * temp;
            new_weight = expf(t1 * (temp_prev - temp) + t2 * (temp - temp_prev));
//        }

        for (int j = 0; j < numSteps; j++) {
            vector_add(vector_get(step, D, j * N + i), y, z, D);
            float ratio;
//            if (log) {
                tz = LOG_TARGET1_H(z, args_t1, D) * (1 - temp) +
                LOG_TARGET2_H(z, args_t2, D) * temp;
                ratio = expf(tz - ty);
//            } else {
//                ratio = temperh(TARGET1_H(z, args_t1, D), 1 - temp)
//                / temperh(TARGET1_H(y, args_t1, D), 1 - temp)
//                * temperh(TARGET2_H(z, args_t2, D), temp)
//                / temperh(TARGET2_H(y, args_t2, D), temp);
//            }

            if (randu[j * N + i] < ratio) {
                ty = tz;
                vector_set(y, z, D);
            }
        }
        vector_set(x_o, y, D);
        vector_set(z, x_i, D);

        //        float new_weight;
        //        float t1;
        //        float t2;
//        if (log) {
//            //            t1 = LOG_TARGET1_H(z, args_t1, D);
//            //            t2 = LOG_TARGET2_H(z, args_t2, D);
//            //            new_weight = expf(t1 * (temp_prev - temp) + t2 * (temp - temp_prev));
//        } else {
//            float t1 = TARGET1_H(z, args_t1, D);
//            float t2 = TARGET2_H(z, args_t2, D);
//            new_weight = temperh(t1, temp_prev - temp) * temperh(t2, temp - temp_prev);
//        }

        w[i] *= new_weight;

    }

    free(y);
    free(z);
}

void FUNC(smcs_ref, TYPE)(
float* x_init, float* x, float* w, int N, int D,
int T, int numSteps, float* cov_step, float* h_args_t1, float* h_args_t2, float* temps, float* ll) {

    //    unsigned int hTimer;
    //    cutCreateTimer(&hTimer);
    //    double time;

    float* cumw = (float*) malloc(N * sizeof(float));

    float* randn = (float*) malloc(N * D * sizeof(float));
    float* steps = (float*) malloc(numSteps * N * D * sizeof(float));

    float* randu1 = (float*) malloc(numSteps * N * sizeof(float));

    float* randu2 = (float*) malloc(N * sizeof(float));

    int* indices = (int*) malloc(N * sizeof(int));

    double sumw;
    double sumw2;

    int i;

    for (i = 0; i < N; i++) {
        w[i] = 1.0;
    }

    double old_sumw = N;

    float* L_step = (float*) malloc(D * D * sizeof(float));
    if (cov_step != NULL) {
        matrix_chol(cov_step, L_step, D);
    }

    double lld = 0;

    //	cudaBindTexture(0, tex_cw, cumw, N * sizeof(float));

    for (int t = 0; t < T; t++) {
        if (cov_step != NULL) {
            populate_randn(randn, numSteps * N * D);
            for (i = 0; i < N; i++) {
                matrix_times(L_step, vector_get(randn, D, i), vector_get(steps, D, i), D, D, D, 1);
            }
        } else {
            populate_randn(steps, numSteps * N * D);
        }

        populate_rand(randu1, numSteps * N);

        populate_rand(randu2, N);

        //        cutResetTimer( hTimer);
        //        cutStartTimer( hTimer);

        FUNC(smcs_step, TYPE)(x_init, w, N, D, steps, numSteps, randu1, temps, x + t * D * N, h_args_t1, h_args_t2, t);

        //        cutStopTimer( hTimer);
        //        time = cutGetTimerValue(hTimer);
        //        printf("Time = %f\n", time);

        sumw = 0;
        sumw2 = 0;

        for (i = 0; i < N; i++) {
            sumw += w[i];
            sumw2 += w[i] * w[i];
        }

//        printf("time %d: %f\n", t, log(sumw / old_sumw));

        lld += log(sumw / old_sumw);

        old_sumw = sumw;

        double ESS = sumw * sumw / sumw2;

        if (ESS < N / 2) {

            scan_ref(N, w, cumw);

            resample_get_indices_ref(cumw, N, randu2, indices, (float) sumw);

            resample_ref(x_init, N, D, indices, x + t * N * D);

            for (i = 0; i < N; i++) {
                w[i] = 1.0;
            }

            old_sumw = N;
        } else {
            if (sumw < 1) {
                for (i = 0; i < N; i++) {
                    w[i] *= (float) (N / sumw);
                }
                old_sumw = N;
            }
            x_init = x + t * D * N;
        }

    }

    *ll = (float) lld;

    free(L_step);

    free(cumw);
    free(indices);
    free(randn);
    free(randu1);
    free(randu2);
    free(steps);

}
