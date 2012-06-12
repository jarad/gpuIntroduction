/*
 * smcs_kernel_mv.cu
 *
 *  Created on: 16-Mar-2009
 *      Author: alee
 */

#include "reduce.h"
#include "test_functions.h"
#include "gauss.h"
#include "rng.h"
#include "scan.h"
#include <stdio.h>
#include "square.h"
#include "matrix.h"
#include "smc_shared.ch"
#include "temper.ch"
#include "sharedmem.cuh"
#include "matrix.ch"
//#include "func.h"

__constant__ float args_t1[NUM_AT1];
__constant__ float args_t2[NUM_AT2];

template<int D>
__global__ void FUNC( smcs_step_gpu, TYPE)(float* x, float* w, int N, float* step, int numSteps, float* randu,
        float* temps, float* x_out, float* sumw, float* sumw2, int time) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    float s = 0;
    float s2 = 0;
    float weight;
    float* x_o;
    float* x_i;

    SharedMemory<float> smem;
    float* sdata = smem.getPointer();
    float* y = sdata + 2 * D * threadIdx.x;
    float* z = sdata + 2 * D * threadIdx.x + D;

    for (int i = tid; i < N; i += tt) {
        x_o = d_vector_get(x_out, D, i);
        x_i = d_vector_get(x, D, i);

        float temp = temps[time];
        float temp_prev;
        if (time == 0) {
            temp_prev = 0;
        } else {
            temp_prev = temps[time - 1];
        }

        d_vector_set(y, x_i, D);

        float tz = 0;
        float ty = 0;
        float new_weight;
//        if (log) {
            float t1 = LOG_TARGET1<D> (y, args_t1) ;
            float t2 = LOG_TARGET2<D> (y, args_t2);
            ty = t1 * (1 - temp) + t2 * temp;
            new_weight = expf(t1 * (temp_prev - temp) + t2 * (temp - temp_prev));
//        } else {
//            float t1 = TARGET1<D> (y, args_t1);
//            float t2 = TARGET2<D> (y, args_t2);
//            ty = temper(t1, (1 - temp)) * temper(t2, temp);
//            new_weight = temper(t1, temp_prev - temp) * temper(t2, temp - temp_prev);
//        }

        for (int j = 0; j < numSteps; j++) {

            d_vector_add(d_vector_get(step, D, j * N + i), y, z, D);
            float ratio;
//            if (log) {
                tz = LOG_TARGET1<D> (z, args_t1) * (1 - temp) + LOG_TARGET2<D> (z, args_t2) * temp;
                ratio = expf(tz - ty);
//            } else {
//                tz = temper(TARGET1<D> (z, args_t1), (1 - temp)) *  temper(TARGET2<D> (z, args_t2), temp);
//                ratio = tz / ty;
//            }

            if (randu[j * N + i] < ratio) {
                ty = tz;
                d_vector_set(y, z, D);
            }

        }
        d_vector_set(x_o, y, D);
        d_vector_set(z, x_i, D);

        weight = w[i] * new_weight;

        w[i] = weight;
        s += weight;
        s2 += weight * weight;
    }

    sumw[tid] = s;
    sumw2[tid] = s2;
}

template<int D>
void FUNC( smcs, TYPE)(float* x_init, float* x, float* w, int N, int T, int numSteps,
float* cov_step, float* h_args_t1, float* h_args_t2, float* d_temps, float* ll,
int nb, int nt) {

    int tt = nb * nt;

    float* cumw;
    cudaMalloc((void**) &cumw, N * sizeof(float));

    float* steps;
    cudaMalloc((void**) &steps, numSteps * N * D * sizeof(float));

    float* randu1;
    cudaMalloc((void**) &randu1, numSteps * N * sizeof(float));

    float* randu2;
    cudaMalloc((void**) &randu2, N * sizeof(float));

    int* indices;
    cudaMalloc((void**) &indices, N * sizeof(int));

    float* sw;
    float* sw2;
    cudaMalloc((void**) &sw, tt * sizeof(float));
    cudaMalloc((void**) &sw2, tt * sizeof(float));
    float* h_sw = (float*) malloc(tt * sizeof(float));
    float* h_sw2 = (float*) malloc(tt * sizeof(float));

    cudaMemcpyToSymbol(args_t1, h_args_t1, NUM_AT1 * sizeof(float));
    cudaMemcpyToSymbol(args_t2, h_args_t2, NUM_AT2 * sizeof(float));

    set(N, w, 1.0, nb, nt);
    double old_sumw = N;

    float* d_L_step = NULL;
    if (cov_step != NULL) {
        float* L_step = (float*) malloc(D * D * sizeof(float));
        matrix_chol(cov_step, L_step, D);

        float* d_L_step;
        cudaMalloc((void**) &d_L_step, D * D * sizeof(float));
        cudaMemcpy(d_L_step, L_step, D * D * sizeof(float), cudaMemcpyHostToDevice);

        free(L_step);
    }

    double lld = 0;

    cudaBindTexture(0, tex_cw, cumw, N * sizeof(float));

    for (int t = 0; t < T; t++) {

        populate_randn_d(steps, numSteps * N * D);
        if (cov_step != NULL) {
            multiply_matrix(numSteps * N, D, steps, steps, d_L_step, nb, nt);
        }
        populate_rand_d(randu1, numSteps * N);
        populate_rand_d(randu2, N);

        FUNC(smcs_step_gpu, TYPE)<D><<<nb,nt, 2 * D * nt * sizeof(float)>>>(x_init, w, N, steps, numSteps, randu1, d_temps, x + t * D * N, sw, sw2, t);

        cudaThreadSynchronize();

        double sumw = 0;
        double sumw2 = 0;

        cudaMemcpy(h_sw, sw, tt * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sw2, sw2, tt * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < tt; i++) {
            sumw += h_sw[i];
            sumw2 += h_sw2[i];
        }

        //		printf("time %d: %f\n", t, log(sumw / old_sumw));

        lld += log(sumw / old_sumw);

        old_sumw = sumw;

        double ESS = sumw * sumw / sumw2;

        if (ESS < N / 2) {

            scan(N, w, cumw, 16, 32);

            resample_get_indices<<<nb, nt>>>(cumw, N, randu2, indices, (float) sumw);

            cudaThreadSynchronize();

            resample<<<nb, nt>>>(x_init, N, D, indices, x + t * N * D);

            set(N, w, 1.0, nb, nt);

            old_sumw = (double) N;

            cudaThreadSynchronize();

        } else {
            if (sumw < 1) {
                multiply(N, w, w, (float) (N / sumw), nb, nt);
                old_sumw = (double) N;
            }
            x_init = x + t * D * N;
        }

    }

    *ll = (float) lld;

    if (cov_step != NULL) {
        cudaFree(d_L_step);
    }

    cudaFree(cumw);
    cudaFree(indices);
    cudaFree(randu1);
    cudaFree(randu2);
    cudaFree(steps);
    free(h_sw);
    free(h_sw2);
    cudaFree(sw);
    cudaFree(sw2);

}

template<int D>
void FUNC( smcs_forget, TYPE)(float* x_init, float* x, float* w, int N, int T, int numSteps,
float* cov_step, float* h_args_t1, float* h_args_t2, float* d_temps, float* ll,
int nb, int nt) {

    int tt = nb * nt;

    float* cumw;
    cudaMalloc((void**) &cumw, N * sizeof(float));

    float* steps;
    cudaMalloc((void**) &steps, numSteps * N * D * sizeof(float));

    float* randu1;
    cudaMalloc((void**) &randu1, numSteps * N * sizeof(float));

    float* randu2;
    cudaMalloc((void**) &randu2, N * sizeof(float));

    int* indices;
    cudaMalloc((void**) &indices, N * sizeof(int));

    float* sw;
    float* sw2;
    cudaMalloc((void**) &sw, tt * sizeof(float));
    cudaMalloc((void**) &sw2, tt * sizeof(float));
    float* h_sw = (float*) malloc(tt * sizeof(float));
    float* h_sw2 = (float*) malloc(tt * sizeof(float));

    cudaMemcpyToSymbol(args_t1, h_args_t1, NUM_AT1 * sizeof(float));
    cudaMemcpyToSymbol(args_t2, h_args_t2, NUM_AT2 * sizeof(float));

    set(N, w, 1.0, nb, nt);
    double old_sumw = N;

    float* d_L_step = NULL;
    if (cov_step != NULL) {
        float* L_step = (float*) malloc(D * D * sizeof(float));
        matrix_chol(cov_step, L_step, D);

        float* d_L_step;
        cudaMalloc((void**) &d_L_step, D * D * sizeof(float));
        cudaMemcpy(d_L_step, L_step, D * D * sizeof(float), cudaMemcpyHostToDevice);

        free(L_step);
    }

    double lld = 0;

    cudaBindTexture(0, tex_cw, cumw, N * sizeof(float));

    for (int t = 0; t < T; t++) {

        populate_randn_d(steps, numSteps * N * D);
        if (cov_step != NULL) {
            multiply_matrix(numSteps * N, D, steps, steps, d_L_step, nb, nt);
        }
        populate_rand_d(randu1, numSteps * N);
        populate_rand_d(randu2, N);

        FUNC(smcs_step_gpu, TYPE)<D><<<nb,nt, 2 * D * nt * sizeof(float)>>>(x_init, w, N, steps, numSteps, randu1, d_temps, x, sw, sw2, t);

        cudaThreadSynchronize();

        double sumw = 0;
        double sumw2 = 0;

        cudaMemcpy(h_sw, sw, tt * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sw2, sw2, tt * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < tt; i++) {
            sumw += h_sw[i];
            sumw2 += h_sw2[i];
        }

        lld += log(sumw / old_sumw);

        old_sumw = sumw;

        double ESS = sumw * sumw / sumw2;

        if (ESS < N / 2) {

            scan(N, w, cumw, 16, 32);

            resample_get_indices<<<nb, nt>>>(cumw, N, randu2, indices, (float) sumw);

            cudaThreadSynchronize();

            resample<<<nb, nt>>>(x_init, N, D, indices, x);

            set(N, w, 1.0, nb, nt);

            old_sumw = N;

            cudaThreadSynchronize();

        } else {
            if (sumw < 1) {
                multiply(N, w, w, (float) (N / sumw), nb, nt);
                old_sumw = (double) N;
            }
            cudaMemcpy(x_init, x, N * D * sizeof(float), cudaMemcpyDeviceToDevice);
        }


    }

    *ll = (float) lld;

    if (cov_step != NULL) {
        cudaFree(d_L_step);
    }

    cudaFree(cumw);
    cudaFree(indices);
    cudaFree(randu1);
    cudaFree(randu2);
    cudaFree(steps);
    free(h_sw);
    free(h_sw2);
    cudaFree(sw);
    cudaFree(sw2);

}
