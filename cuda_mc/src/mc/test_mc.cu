/*
 * test_mc.c
 *
 *  Created on: 02-Feb-2009
 *      Author: alee
 */

#include <stdio.h>
#include "rng.h"
#include <cutil.h>
#include "reduce.h"
#include "mc_gauss.h"
#include "mc_mix_gauss.h"
#include "mc_mix_gauss_mu.h"
#include "gauss.h"
#include "test_functions.h"
#include "mc_gauss_mv.h"
#include "mix_gauss.h"
#include "matrix.h"
#include "order.h"

void test_mcgaussmv_nolog(int N, int D, float* h_args_p, float* h_args_q, float* props) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* warray = (float*) malloc(N * sizeof(float));
    float sum[2];
    double sumd[2];
    float sumw = 0;
    double sumwd = 0;

    for (int j = 0; j < D; j++) {
        sum[j] = 0;
        sumd[j] = 0;
    }

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    is_ref_nn_mv(N, D, props, warray, h_args_p, h_args_q, 0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            sumd[j] += warray[i] * vector_get(props, D, i)[j];
        }
        sumwd += warray[i];
    }

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f\n", time);

    printf("HOST RESULT = (%f, %f)\n", sumd[0] / sumwd, sumd[1] / sumwd);

    free(warray);

    float* d_array;
    cudaMalloc((void **) &d_array, N * D * sizeof(float));
    float* d_warray;
    cudaMalloc((void **) &d_warray, N * sizeof(float));

    cudaMemcpy(d_array, props, N * D * sizeof(float), cudaMemcpyHostToDevice);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    is_nn_mv(N, D, d_array, d_warray, h_args_p, h_args_q, 0, 32, 128);

    cudaThreadSynchronize();

    multiply(N, D, d_array, d_array, d_warray, 32, 128);
    reduce(N, d_warray, sumw, 32, 128);

    reduce(N, D, d_array, sum, 32, 128);
    cudaThreadSynchronize();

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f\n", time);

    printf("RESULT = (%f,%f)\n", sum[0] / sumw, sum[1] / sumw);

    cudaFree(d_array);
    cudaFree(d_warray);
}

void test_mcgaussmv_log(int N, int D, float* h_args_p, float* h_args_q, float* props) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* warray = (float*) malloc(N * sizeof(float));
    float sum[2];
    double sumd[2];
    float sumw = 0;
    double sumwd = 0;

    for (int j = 0; j < D; j++) {
        sum[j] = 0;
        sumd[j] = 0;
    }

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    is_ref_nn_mv(N, D, props, warray, h_args_p, h_args_q, 1);

    float maxlw = warray[0];

    for (int i = 1; i < N; i++) {
        maxlw = max(maxlw, warray[i]);
    }

    for (int i = 0; i < N; i++) {
        warray[i] -= maxlw;
        warray[i] = exp(warray[i]);
        for (int j = 0; j < D; j++) {
            sumd[j] += warray[i] * vector_get(props, D, i)[j];
        }
        sumwd += warray[i];
    }

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f\n", time);

    printf("HOST RESULT = (%f, %f)\n", sumd[0] / sumwd, sumd[1] / sumwd);

    free(warray);

    float* d_array;
    cudaMalloc((void **) &d_array, N * D * sizeof(float));
    float* d_warray;
    cudaMalloc((void **) &d_warray, N * sizeof(float));

    cudaMemcpy(d_array, props, N * D * sizeof(float), cudaMemcpyHostToDevice);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    is_nn_mv(N, D, d_array, d_warray, h_args_p, h_args_q, 1, 32, 128);

    cudaThreadSynchronize();

    maximum(N, d_warray, maxlw, 32, 128);
    add(N, d_warray, d_warray, -maxlw, 32, 128);
    exp(N, d_warray, d_warray, 32, 128);

    multiply(N, D, d_array, d_array, d_warray, 32, 128);
    reduce(N, d_warray, sumw, 32, 128);

    reduce(N, D, d_array, sum, 32, 128);
    cudaThreadSynchronize();

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f\n", time);

    printf("RESULT = (%f,%f)\n", sum[0] / sumw, sum[1] / sumw);

    cudaFree(d_array);
    cudaFree(d_warray);
}

// importance sampling with multivariate Gaussian proposal and target distributions
void test_mcgauss_mv(int N) {
    const int D = 2;

    printf("\nIS: Gaussian-Gaussian 2D\n");

    float h_args_p[1 + D * D + D];
    float cov_p[D * D];
    matrix_set(cov_p, D, D, 0, 0, 1.0f);
    matrix_set(cov_p, D, D, 0, 1, 0.5f);
    matrix_set(cov_p, D, D, 1, 0, 0.5f);
    matrix_set(cov_p, D, D, 1, 1, 2.0f);
    compute_c1_c2(cov_p, D, h_args_p[0], h_args_p + 1);
    h_args_p[5] = 1;
    h_args_p[6] = 1;

    float h_args_q[1 + D * D + D];
    float cov_q[D * D];
    matrix_set(cov_q, D, D, 0, 0, 1.0f);
    matrix_set(cov_q, D, D, 0, 1, 0.0f);
    matrix_set(cov_q, D, D, 1, 0, 0.0f);
    matrix_set(cov_q, D, D, 1, 1, 1.0f);
    compute_c1_c2(cov_q, D, h_args_q[0], h_args_q + 1);
    h_args_q[5] = 0;
    h_args_q[6] = 0;

    float* array = (float*) malloc(N * D * sizeof(float));

    populate_randn(array, N * D);
    cudaThreadSynchronize();

    test_mcgaussmv_nolog(N, D, h_args_p, h_args_q, array);
    test_mcgaussmv_log(N, D, h_args_p, h_args_q, array);

    free(array);
}

// importance sampling with univariate Gaussian proposal and target distributions
void test_mcgauss(int N) {
    unsigned int hTimer;
    double ctime, gtime;
    cutCreateTimer(&hTimer);

    printf("\nIS: Gaussian-Gaussian 1D\n");
    float h_args_p[3];
    float h_args_q[3];

    // p is N(2,0.25), q is N(0,1)
    compute_c1_c2(0.5f, h_args_p[0], h_args_p[1]);
    compute_c1_c2(1.0f, h_args_q[0], h_args_q[1]);
    h_args_p[2] = 2;
    h_args_q[2] = 0;

    float* array = (float*) malloc(N * sizeof(float));
    float* warray = (float*) malloc(N * sizeof(float));

    populate_randn(array, N);

    float h_sum = 0;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    is_ref_nn(N, array, warray, h_args_p, h_args_q);

    for (int i = 0; i < N; i++) {
        h_sum += array[i] * warray[i];
    }

    cutStopTimer(hTimer);
    ctime = cutGetTimerValue(hTimer);
    printf("Time = %f\n", ctime);

    printf("HOST RESULT = %f\n", h_sum / N);

    free(array);
    free(warray);

    float* d_array;
    cudaMalloc((void **) &d_array, N * sizeof(float));
    float* d_array2;
    cudaMalloc((void **) &d_array2, N * sizeof(float));
    float* d_warray;
    cudaMalloc((void **) &d_warray, N * sizeof(float));

    populate_randn_d(d_array, N);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    is_nn(N, d_array, d_warray, h_args_p, h_args_q, 32, 128);

    cudaThreadSynchronize();

    multiply(N, d_array, d_array2, d_warray, 32, 128);

    cudaThreadSynchronize();

    reduce(N, d_array2, h_sum, 32, 128);
    cudaThreadSynchronize();

    cutStopTimer(hTimer);
    gtime = cutGetTimerValue(hTimer);
    printf("Time = %f\n", gtime);

    printf("RESULT = %f\n", h_sum / N);

    printf("speedup = %f\n", ctime / gtime);

    cudaFree(d_array);
    cudaFree(d_array2);
    cudaFree(d_warray);
}

// importance sampling with target distribution being a mixture of univariate Gaussians and
// proposal distribution being Gaussian
void test_mixgauss(int N) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    printf("\nIS: Mixture of Gaussians 1D\n");

    const int k = 2;
    float h_args_p[1 + 3 * k];
    float h_args_q[3];

    // p is N(2,2), q is N(0,1)
    h_args_p[0] = k;
    h_args_p[1] = 0;
    h_args_p[2] = 3;
    compute_ci1_ci2(0.5f, 0.5f, h_args_p[3], h_args_p[5]);
    compute_ci1_ci2(0.5f, 0.5f, h_args_p[4], h_args_p[6]);

    compute_c1_c2(1.0f, h_args_q[0], h_args_q[1]);
    h_args_q[2] = 0;

    float* array = (float*) malloc(N * sizeof(float));
    float* warray = (float*) malloc(N * sizeof(float));

    populate_randn(array, N);

    cudaThreadSynchronize();

    float h_sum = 0;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    is_ref_nmni(N, array, warray, h_args_p, h_args_q);

    for (int i = 0; i < N; i++) {
        h_sum += array[i] * warray[i];
    }

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f\n", time);

    printf("HOST RESULT = %f\n", h_sum / N);

    free(array);
    free(warray);

    float* d_array;
    cudaMalloc((void **) &d_array, N * sizeof(float));
    float* d_array2;
    cudaMalloc((void **) &d_array2, N * sizeof(float));
    float* d_warray;
    cudaMalloc((void **) &d_warray, N * sizeof(float));

    populate_randn_d(d_array, N);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    is_nmni(N, d_array, d_warray, h_args_p, h_args_q, 32, 128);

    cudaThreadSynchronize();

    multiply(N, d_array, d_array2, d_warray, 32, 128);
    cudaThreadSynchronize();

    reduce(N, d_array2, h_sum, 32, 128);
    cudaThreadSynchronize();

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f\n", time);
    printf("RESULT = %f\n", h_sum / N);

    cudaFree(d_array);
    cudaFree(d_array2);
    cudaFree(d_warray);
}

// importance sampling with target distribution being the posterior distribution of the means of a
// Gaussian mixture model given 100 observations with known and shared variance, equal weights with
// uniform prior on (-10,10)^4. actual means are -3, 0, 3, and 6.
// proposal distribution is uniform (-10,10)^4.
void test_mix(int N) {
    const int D = 4;
    int nb = 512;
    int nt = 128;

    const int L = 100;
    float sigma = 0.55f;
    float mus[4];
    mus[0] = -3;
    mus[1] = 0;
    mus[2] = 3;
    mus[3] = 6;

    float data_array[L];
    generate_mix_data(D, sigma, mus, data_array, L);

    float c1, c2;
    compute_ci1_ci2(sigma, 1.0f / D, c1, c2);

    float h_args_p[L + 5];
    h_args_p[0] = L;

    for (int i = 0; i < L; i++) {
        h_args_p[i + 1] = data_array[i];
    }

    h_args_p[L + 1] = c1;
    h_args_p[L + 2] = c2;
    h_args_p[L + 3] = -10;
    h_args_p[L + 4] = 10;

    float h_args_q[2];
    h_args_q[0] = -10;
    h_args_q[1] = 10;

    unsigned int hTimer;
    double time1, time2;
    cutCreateTimer(&hTimer);

    printf("\nIS: Mixture of Gaussians: Mean Inference\n");

    float* d_array;
    cudaMalloc((void **) &d_array, N * D * sizeof(float));
    populate_rand_d(d_array, N * D);
    multiply(N * D, d_array, d_array, 20, nb, nt);
    cudaThreadSynchronize();
    add(N * D, d_array, d_array, -10, nb, nt);
    cudaThreadSynchronize();

    float* array = (float*) malloc(N * D * sizeof(float));
    float* warray = (float*) malloc(N * sizeof(float));
    float sum[D];
    double sumd[D];
    float sumw = 0;
    double sumwd = 0;

    for (int j = 0; j < D; j++) {
        sum[j] = 0;
        sumd[j] = 0;
    }

    cudaMemcpy(array, d_array, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    is_ref_mgmu_mv(N, D, array, warray, h_args_p, h_args_q, 1);

    //    cutStopTimer(hTimer);
    //    time1 = cutGetTimerValue(hTimer);
    //    printf("Time = %f\n", time1);

    float maxlw = warray[0];

    for (int i = 1; i < N; i++) {
        maxlw = max(maxlw, warray[i]);
    }

    for (int i = 0; i < N; i++) {
        warray[i] -= maxlw;
        warray[i] = exp(warray[i]);
        for (int j = 0; j < D; j++) {
            sumd[j] += warray[i] * vector_get(array, D, i)[j];
        }
        sumwd += warray[i];
    }

    cutStopTimer(hTimer);
    time1 = cutGetTimerValue(hTimer);
    printf("Time = %f\n", time1);

    printf("HOST RESULT = (%f, %f, %f, %f)\n", sumd[0] / sumwd, sumd[1] / sumwd, sumd[2] / sumwd,
            sumd[3] / sumwd);

    free(warray);

    float* d_warray;
    cudaMalloc((void **) &d_warray, N * sizeof(float));

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    is_mgmu_mv(N, D, d_array, d_warray, h_args_p, h_args_q, 1, nb, nt);

    cudaThreadSynchronize();

    //    cutStopTimer(hTimer);
    //    time2 = cutGetTimerValue(hTimer);
    //    printf("Time = %f\n", time2);

    maximum(N, d_warray, maxlw, nb, nt);
    add(N, d_warray, d_warray, -maxlw, nb, nt);
    exp(N, d_warray, d_warray, nb, nt);

    multiply(N, D, d_array, d_array, d_warray, nb, nt);
    reduce(N, d_warray, sumw, nb, nt);

    reduce(N, D, d_array, sum, nb, nt);
    cudaThreadSynchronize();

    cutStopTimer(hTimer);
    time2 = cutGetTimerValue(hTimer);
    printf("Time = %f\n", time2);

    printf("RESULT = (%f, %f, %f, %f)\n", sum[0] / sumw, sum[1] / sumw, sum[2] / sumw, sum[3]
            / sumw);

    cudaFree(d_array);
    cudaFree(d_warray);

    printf("speedup = %f\n", time1 / time2);

}

int main(int argc, char **argv) {
    seed_rng();
    int N = 1048576;
    //    int N = 131072;
    //        int N = 65536;
    //    int N = 16777216;
    test_mcgauss(N);
    test_mcgauss_mv(N);
    test_mixgauss(N);
    test_mix(N);
    kill_rng();
}

