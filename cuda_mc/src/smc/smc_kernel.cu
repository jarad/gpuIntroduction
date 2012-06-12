/*
 * smc_kernel.cu
 *
 *  Created on: 1-Mar-2009
 *      Author: Owner
 */

#ifndef SMC_KERNEL_CU_
#define SMC_KERNEL_CU_

#include "reduce.h"
#include "test_functions.h"
#include "gauss.h"
#include "rng.h"
#include "scan.h"
#include <stdio.h>
#include "square.h"
#include "matrix.h"
#include "fsv.h"
#include "smc_shared.ch"

__constant__ float args_l[NUM_AL];

__global__ void FUNC( smc_step_gpu, TYPE)(float* x, float* w, int N, float y, float scale_step, float* step, float* x_out,
        float* sumw, float* sumw2) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    float s = 0;
    float s2 = 0;
    float weight;

    for (int i = tid; i < N; i += tt) {
        x_out[i] = scale_step * x[i] + step[i];
        weight = w[i] * LIKELIHOOD(x_out[i], y, args_l);
        w[i] = weight;
        s += weight;
        s2 += weight * weight;
    }

    sumw[tid] = s;
    sumw2[tid] = s2;
}

// x should be size N * T, x_init size N, y size T.
void FUNC( smc, TYPE)(float* x_init, float* x, float* w, float* y, int N, int T, float* h_args_l,
        float scale_step, float sigma_step, float* ll, int nb, int nt) {

    int tt = nb * nt;

    float* cumw;
    cudaMalloc((void**) &cumw, N * sizeof(float));

    float* steps;
    cudaMalloc((void**) &steps, N * sizeof(float));

    float* randu;
    cudaMalloc((void**) &randu, N * sizeof(float));

    int* indices;
    cudaMalloc((void**) &indices, N * sizeof(int));

    float* xt_copy;
    cudaMalloc((void**) &xt_copy, T * N * sizeof(float));

    float* sw;
    float* sw2;
    cudaMalloc((void**) &sw, tt * sizeof(float));
    cudaMalloc((void**) &sw2, tt * sizeof(float));
    float* h_sw = (float*) malloc(tt * sizeof(float));
    float* h_sw2 = (float*) malloc(tt * sizeof(float));

    cudaMemcpyToSymbol(args_l, h_args_l, NUM_AL * sizeof(float));

    double old_sumw = (double) N;

    set(N, w, 1.0, nb, nt);

    double lld = 0;

    cudaBindTexture(0, tex_cw, cumw, N * sizeof(float));

    int* history;
    cudaMalloc((void**) &history, N * T * sizeof(int));

    int* history_id;
    cudaMalloc((void**) &history_id, N * sizeof(int));

    history_identity<<<nb,nt>>>(history_id, N);

    for (int t = 0; t < T; t++) {
        populate_randn_d(steps, N);
        populate_rand_d(randu, N);

        if (sigma_step != 1.0) {
            multiply(N, steps, steps, sigma_step, nb, nt);
        }

        FUNC(smc_step_gpu, TYPE)<<<nb, nt>>>(x_init, w, N, y[t], scale_step, steps, x + t * N, sw, sw2);

        cudaThreadSynchronize();

        double sumw = 0;
        double sumw2 = 0;

        cudaMemcpy(h_sw, sw, tt * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sw2, sw2, tt * sizeof(float), cudaMemcpyDeviceToHost);
        //
        for (int i = 0; i < tt; i++) {
            sumw += h_sw[i];
            sumw2 += h_sw2[i];
        }

        lld += log(sumw / old_sumw);

        old_sumw = sumw;

        double ESS = sumw * sumw / sumw2;

        if (ESS < N / 2) {

            scan(N, w, cumw, nb, nt);

            resample_get_indices<<<nb, nt>>>(cumw, N, randu, indices, (float) sumw);

            cudaThreadSynchronize();

            //			cudaMemcpy(xt_copy, x, N * (t + 1) * sizeof(float), cudaMemcpyDeviceToDevice);

            cudaMemcpy(history + t * N, indices, N * sizeof(int), cudaMemcpyDeviceToDevice);

            resample<<<nb, nt>>>(x_init, N, indices, x + N * t);

            set(N, w, 1.0, nb, nt);

            old_sumw = (double) N;

            cudaThreadSynchronize();
        } else {
            if (sumw < 1) {
                multiply(N, w, w, (float) (N / sumw), nb, nt);
                old_sumw = (double) N;
            }
            x_init = x + t * N;
            cudaMemcpy(history + t * N, history_id, N * sizeof(int), cudaMemcpyDeviceToDevice);
        }

    }

    *ll = (float) lld;

    cudaMemcpy(xt_copy, x, N * T * sizeof(float), cudaMemcpyDeviceToDevice);

    historify<<<nb, nt>>>(x, N, T, history, xt_copy);

    cudaThreadSynchronize();

    cudaFree(cumw);
    cudaFree(indices);
    cudaFree(randu);
    cudaFree(xt_copy);
    cudaFree(steps);
    free(h_sw);
    free(h_sw2);
    cudaFree(sw);
    cudaFree(sw2);

    cudaFree(history);
    cudaFree(history_id);

}

// x should be size N, x_init size N, y size T.
void FUNC( smc_forget, TYPE)(float* x_init, float* x, float* w, float* y, int N, int T,
        float* h_args_l, float scale_step, float sigma_step, float* ll, int nb, int nt) {

    int tt = nb * nt;

    float* cumw;
    cudaMalloc((void**) &cumw, N * sizeof(float));

    float* steps;
    cudaMalloc((void**) &steps, N * sizeof(float));

    float* randu;
    cudaMalloc((void**) &randu, N * sizeof(float));

    int* indices;
    cudaMalloc((void**) &indices, N * sizeof(int));

    float* sw;
    float* sw2;
    cudaMalloc((void**) &sw, tt * sizeof(float));
    cudaMalloc((void**) &sw2, tt * sizeof(float));
    float* h_sw = (float*) malloc(tt * sizeof(float));
    float* h_sw2 = (float*) malloc(tt * sizeof(float));

    cudaMemcpyToSymbol(args_l, h_args_l, NUM_AL * sizeof(float));

    set(N, w, 1.0, nb, nt);
    double old_sumw = (double) N;

    double lld = 0;

    cudaBindTexture(0, tex_cw, cumw, N * sizeof(float));

    for (int t = 0; t < T; t++) {
        populate_randn_d(steps, N);
        populate_rand_d(randu, N);
        if (sigma_step != 1.0) {
            multiply(N, steps, steps, sigma_step, nb, nt);
        }

        FUNC(smc_step_gpu, TYPE)<<<nb,nt>>>(x_init, w, N, y[t], scale_step, steps, x, sw, sw2);

        cudaThreadSynchronize();

        double sumw = 0;
        double sumw2 = 0;

        cudaMemcpy(h_sw, sw, tt * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sw2, sw2, tt * sizeof(float), cudaMemcpyDeviceToHost);
        //
        for (int i = 0; i < tt; i++) {
            sumw += h_sw[i];
            sumw2 += h_sw2[i];
        }

        lld += log(sumw / old_sumw);

        old_sumw = sumw;

        double ESS = sumw * sumw / sumw2;

        if (ESS < N / 2) {

            scan(N, w, cumw, 16, 32);

            resample_get_indices<<<nb, nt>>>(cumw, N, randu, indices, (float) sumw);

            cudaThreadSynchronize();

            resample<<<nb, nt>>>(x_init, N, indices, x);

            set(N, w, 1.0, nb, nt);

            old_sumw = (double) N;

            cudaThreadSynchronize();

        } else {
            if (sumw < 1) {
                multiply(N, w, w, (float) (N / sumw), nb, nt);
                old_sumw = (double) N;
            }
            cudaMemcpy(x_init, x, N * sizeof(float), cudaMemcpyDeviceToDevice);
        }

    }

    *ll = (float) lld;

    cudaFree(cumw);
    cudaFree(indices);
    cudaFree(randu);
    cudaFree(steps);
    free(h_sw);
    free(h_sw2);
    cudaFree(sw);
    cudaFree(sw2);

}

#endif /* SMC_KERNEL_CU_ */
