/*
 * smc_kernel_mv.cu
 *
 *  Created on: 10-Mar-2009
 *      Author: alee
 */

#ifndef SMC_KERNEL_MV_CU_
#define SMC_KERNEL_MV_CU_

#include "reduce.h"
#include "test_functions.h"
#include "gauss.h"
#include "rng.h"
#include "scan.h"
#include <stdio.h>
#include "square.h"
#include "matrix.h"
#include "matrix.ch"
#include "smc_shared.ch"
#include "func.h"

__constant__ float args_l[NUM_AL];

template<int Dx, int Dy>
__global__ void FUNC( smc_step_gpu, TYPE)(float* x, float* w, int N, float* y, float* scale_step, float* step,
        float* x_out, float* sumw, float* sumw2, int t) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    float s = 0;
    float s2 = 0;
    float weight;
    float* x_o;
    float* x_i;
    float* yt = d_vector_get(y, Dy, t);

    for (int i = tid; i < N; i += tt) {
        x_o = d_vector_get(x_out, Dx, i);
        x_i = d_vector_get(x, Dx, i);

        d_matrix_times<Dx> (scale_step, x_i, x_o, Dx, 1);
        d_vector_add(d_vector_get(step, Dx, i), x_o, x_o, Dx);

        weight = w[i] * LIKELIHOOD<Dx, Dy> (x_o, yt, args_l);
        w[i] = weight;
        s += weight;
        s2 += weight * weight;
    }
    sumw[tid] = s;
    sumw2[tid] = s2;
}

//template<int Dx, int Dy>
//__global__ void FUNC( smc_step_gpu_log, TYPE)(float* x, float* w, int N, float* y, float* scale_step, float* step,
//        float* x_out, float* sumw, float* sumw2, int t) {
//    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
//    const int tt = blockDim.x * gridDim.x;
//    float s = -FLT_MAX;
//    float s2 = -FLT_MAX;
//    float weight;
//    float* x_o;
//    float* x_i;
//    float* yt = d_vector_get(y, Dy, t);
//
//    for (int i = tid; i < N; i += tt) {
//        x_o = d_vector_get(x_out, Dx, i);
//        x_i = d_vector_get(x, Dx, i);
//
//        d_matrix_times<Dx> (scale_step, x_i, x_o, Dx, 1);
//        d_vector_add(d_vector_get(step, Dx, i), x_o, x_o, Dx);
//
//        weight = log(w[i]) + LOG_LIKELIHOOD<Dx, Dy> (x_o, yt, args_l);
//        w[i] = exp(weight);
//
//        s = logsumexp(s, weight);
//        s2 = logsumexp(s2, 2*weight);
//    }
//    sumw[tid] = exp(s);
//    sumw2[tid] = exp(s2);
//}

template<int Dx, int Dy>
void FUNC( smc, TYPE)(float* x_init, float* x, float* w, float* y, int N, int T, float* h_args_l,
        float* scale_step, float* cov_step, float* ll, int nb, int nt) {

    int tt = nb * nt;

    float* cumw;
    cudaMalloc((void**) &cumw, N * sizeof(float));

    float* steps;
    cudaMalloc((void**) &steps, N * Dx * sizeof(float));

    float* randu;
    cudaMalloc((void**) &randu, N * sizeof(float));

    int* indices;
    cudaMalloc((void**) &indices, N * sizeof(int));

    float* xt_copy;
    cudaMalloc((void**) &xt_copy, N * Dx * T * sizeof(float));

    float* sw;
    float* sw2;
    cudaMalloc((void**) &sw, tt * sizeof(float));
    cudaMalloc((void**) &sw2, tt * sizeof(float));
    float* h_sw = (float*) malloc(tt * sizeof(float));
    float* h_sw2 = (float*) malloc(tt * sizeof(float));

    cudaMemcpyToSymbol(args_l, h_args_l, NUM_AL * sizeof(float));

    set(N, w, 1.0, nb, nt);
    double old_sumw = (double) N;

    float* L_step = (float*) malloc(Dx * Dx * sizeof(float));
    matrix_chol(cov_step, L_step, Dx);

    float* d_L_step;
    cudaMalloc((void**) &d_L_step, Dx * Dx * sizeof(float));
    cudaMemcpy(d_L_step, L_step, Dx * Dx * sizeof(float), cudaMemcpyHostToDevice);

    float* d_scale_step;
    cudaMalloc((void**) &d_scale_step, Dx * Dx * sizeof(float));
    cudaMemcpy(d_scale_step, scale_step, Dx * Dx * sizeof(float), cudaMemcpyHostToDevice);

    free(L_step);

    double lld = 0;

    cudaBindTexture(0, tex_cw, cumw, N * sizeof(float));

    int* history;
    cudaMalloc((void**) &history, N * T * sizeof(int));

    int* history_id;
    cudaMalloc((void**) &history_id, N * sizeof(int));

    history_identity<<<nb,nt>>>(history_id, N);

    for (int t = 0; t < T; t++) {

        populate_randn_d(steps, N * Dx);
        multiply_matrix(N, Dx, steps, steps, d_L_step, nb, nt);
        populate_rand_d(randu, N);

        FUNC(smc_step_gpu, TYPE)<Dx,Dy><<<nb,nt>>>(x_init, w, N, y, d_scale_step, steps, x + t * Dx * N, sw, sw2, t);

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

            resample_get_indices<<<nb, nt>>>(cumw, N, randu, indices, (float) sumw);

            cudaThreadSynchronize();

            cudaMemcpy(history + t * N, indices, N * sizeof(int), cudaMemcpyDeviceToDevice);

            resample<<<nb, nt>>>(x_init, N, Dx, indices, x + N * Dx * t);

            set(N, w, 1.0, nb, nt);

            old_sumw = (double) N;

            cudaThreadSynchronize();

        } else {
            if (sumw < 1) {
                multiply(N, w, w, (float) (N / sumw), nb, nt);
                old_sumw = (double) N;
            }
            x_init = x + t * Dx * N;
            cudaMemcpy(history + t * N, history_id, N * sizeof(int), cudaMemcpyDeviceToDevice);

        }
    }

    *ll = (float) lld;

    cudaMemcpy(xt_copy, x, N * Dx * T * sizeof(float), cudaMemcpyDeviceToDevice);

    historify<<<nb, nt>>>(x, N, Dx, T, history, xt_copy);

    cudaThreadSynchronize();

    cudaFree(d_L_step);
    cudaFree(d_scale_step);
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

template<int Dx, int Dy>
void FUNC( smc_forget, TYPE)(float* x_init, float* x, float* w, float* y, int N, int T,
        float* h_args_l, float* scale_step, float* cov_step, float* ll, int nb, int nt) {

    int tt = nb * nt;

    float* cumw;
    cudaMalloc((void**) &cumw, N * sizeof(float));

    float* steps;
    cudaMalloc((void**) &steps, N * Dx * sizeof(float));

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

    //	if (sigma_step != 1.0) {
    float* L_step = (float*) malloc(Dx * Dx * sizeof(float));
    matrix_chol(cov_step, L_step, Dx);

    float* d_L_step;
    cudaMalloc((void**) &d_L_step, Dx * Dx * sizeof(float));
    cudaMemcpy(d_L_step, L_step, Dx * Dx * sizeof(float), cudaMemcpyHostToDevice);

    float* d_scale_step;
    cudaMalloc((void**) &d_scale_step, Dx * Dx * sizeof(float));
    cudaMemcpy(d_scale_step, scale_step, Dx * Dx * sizeof(float), cudaMemcpyHostToDevice);

    free(L_step);

    double lld = 0;

    cudaBindTexture(0, tex_cw, cumw, N * sizeof(float));

    for (int t = 0; t < T; t++) {

        //		printf("time = %d\n", t);

        populate_randn_d(steps, N * Dx);
        multiply_matrix(N, Dx, steps, steps, d_L_step, nb, nt);
        populate_rand_d(randu, N);

        FUNC(smc_step_gpu, TYPE)<Dx, Dy><<<nb,nt>>>(x_init, w, N, y, d_scale_step, steps, x, sw, sw2, t);

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

            resample_get_indices<<<nb, nt>>>(cumw, N, randu, indices, (float) sumw);

            cudaThreadSynchronize();

            resample<<<nb, nt>>>(x_init, N, Dx, indices, x);

            set(N, w, 1.0, nb, nt);

            old_sumw = (double) N;

            cudaThreadSynchronize();

        } else {
            if (sumw < 1) {
                multiply(N, w, w, (float) (N / sumw), nb, nt);
                old_sumw = (double) N;
            }
            cudaMemcpy(x_init, x, N * Dx * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    }

    *ll = (float) lld;

    cudaFree(d_L_step);
    cudaFree(d_scale_step);
    cudaFree(cumw);
    cudaFree(indices);
    cudaFree(randu);
    cudaFree(steps);
    free(h_sw);
    free(h_sw2);
    cudaFree(sw);
    cudaFree(sw2);

}

#endif /* SMC_KERNEL_MV_CU_ */
