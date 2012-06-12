/*
 * test_smc.cu
 *
 *  Created on: 1-Mar-2009
 *      Author: Owner
 */

#include <stdio.h>
#include <cutil.h>
#include "rng.h"
#include "gauss.h"
#include "output.h"
#include "kalman.h"
#include "matrix.h"
#include "fsv.h"
#include "smc_fsv.h"
#include "smc_lg.h"
#include "smc_usv.h"
#include "smc_mvlg.h"
#include "scan.h"
#include "usv.h"

void generate_data(float* xs, float* ys, int T, float sigma_x, float sigma_y) {
    const int M = 32768;
    float steps[M];

    populate_randn(steps, M);

    // xs_{-1} = 0;
    xs[0] = steps[0];
    ys[0] = xs[0] + steps[1];
    for (int i = 1; i < T; i++) {
        xs[i] = xs[i - 1] + steps[i * 2];
        ys[i] = xs[i] + steps[i * 2 + 1];
    }
}

template<class T>
void generate_data_mv(T* xs, T* ys, int Dx, int Dy, int total_time, T* scale_step, T* cov_step,
        T* scale_like, T* cov_like) {
    int Mx = max(total_time * Dx, 32768);
    int My = max(total_time * Dy, 32768);

    T* steps_x = (T*) malloc(Mx * sizeof(T));
    T* steps_y = (T*) malloc(My * sizeof(T));

    T* L_step = (T*) malloc(Dx * Dx * sizeof(T));
    T* L_like = (T*) malloc(Dy * Dy * sizeof(T));

    T* temp_x = (T*) malloc(Dx * sizeof(T));
    T* temp_y = (T*) malloc(Dy * sizeof(T));

    matrix_chol(cov_step, L_step, Dx);
    matrix_chol(cov_like, L_like, Dy);

    //    matrix_print(L_step, Dx, Dx);
    //    matrix_print(L_like, Dy, Dy);

    populate_randn(steps_x, Mx);
    populate_randn(steps_y, My);

    // xs_{-1} = 0;
    matrix_times(L_step, steps_x, temp_x, Dx, Dx, Dx, 1);
    vector_set(xs, temp_x, Dx);

    matrix_times(scale_like, xs, ys, Dy, Dx, Dx, 1);
    matrix_times(L_like, vector_get(steps_y, Dy, 0), temp_y, Dy, Dy, Dy, 1);
    vector_add(ys, temp_y, ys, Dy);

    for (int i = 1; i < total_time; i++) {

        matrix_times(scale_step, vector_get(xs, Dx, i - 1), vector_get(xs, Dx, i), Dx, Dx, Dx, 1);
        matrix_times(L_step, vector_get(steps_x, Dx, i), temp_x, Dx, Dx, Dx, 1);
        vector_add(vector_get(xs, Dx, i), temp_x, vector_get(xs, Dx, i), Dx);

        matrix_times(scale_like, vector_get(xs, Dx, i), vector_get(ys, Dy, i), Dy, Dx, Dx, 1);
        matrix_times(L_like, vector_get(steps_y, Dy, i), temp_y, Dy, Dy, Dy, 1);
        vector_add(vector_get(ys, Dy, i), temp_y, vector_get(ys, Dy, i), Dy);

    }

    free(L_step);
    free(L_like);
    free(temp_x);
    free(temp_y);
    free(steps_x);
    free(steps_y);
}

void test_smc_lg_kalman(int T, float* ys_real, float sigma_like, float sigma_step) {

    float ll_kalman;

    float* kalman_xs = (float*) malloc(T * sizeof(float));

    kalman(0.0, kalman_xs, ys_real, T, sigma_like, sigma_step, &ll_kalman);

    to_file(kalman_xs, T, "xs_lg_kalman.txt");

    printf("ll_lg_kalman = %f\n", ll_kalman);

    free(kalman_xs);

}

void test_smc_lg(int N, int T, float* ys_real, float* h_args_l, float scale_step, float sigma_step,
        int nb, int nt) {

    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* d_xs;
    float* x_init;

    float* xs = (float*) malloc(N * T * sizeof(float));
    cudaMalloc((void**) &d_xs, N * T * sizeof(float));

    float* ws = (float*) malloc(N * sizeof(float));
    float* d_ws;
    cudaMalloc((void**) &d_ws, N * sizeof(float));

    cudaMalloc((void**) &x_init, N * sizeof(float));
    populate_randn_d(x_init, N);

    float ll;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_lg(x_init, d_xs, d_ws, ys_real, N, T, h_args_l, scale_step, sigma_step, &ll, nb, nt);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_lg = %f\n", ll);

    cudaMemcpy(xs, d_xs, N * T * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ws, d_ws, N * sizeof(float), cudaMemcpyDeviceToHost);
    to_file(xs, N * T, "xs_lg.txt");
    to_file(ws, N, "ws_lg.txt");

    free(ws);
    cudaFree(d_ws);
    free(xs);
    cudaFree(d_xs);
    cudaFree(x_init);

}

void test_smc_lg_forget(int N, int T, float* ys_real, float* h_args_l, float scale_step,
        float sigma_step, int nb, int nt) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* d_xs;
    float* x_init;

    float* xs = (float*) malloc(N * sizeof(float));
    cudaMalloc((void**) &d_xs, N * sizeof(float));
    cudaMalloc((void**) &x_init, N * sizeof(float));

    float* ws = (float*) malloc(N * sizeof(float));
    float* d_ws;
    cudaMalloc((void**) &d_ws, N * sizeof(float));

    populate_randn_d(x_init, N);

    float ll_forget;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_forget_lg(x_init, d_xs, d_ws, ys_real, N, T, h_args_l, scale_step, sigma_step, &ll_forget,
            nb, nt);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_lg_forget = %f\n", ll_forget);

    free(xs);
    free(ws);
    cudaFree(d_ws);
    cudaFree(d_xs);
    cudaFree(x_init);
}

void test_smc_lg_ref(int N, int T, float* ys_real, float* h_args_l, float scale_step,
        float sigma_step) {

    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* xs = (float*) malloc(N * T * sizeof(float));

    float* ws = (float*) malloc(N * sizeof(float));

    float* hx_init = (float*) malloc(N * sizeof(float));

    populate_randn(hx_init, N);

    float ll_ref;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_ref_lg(hx_init, xs, ws, ys_real, N, T, h_args_l, scale_step, sigma_step, &ll_ref);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_lg_ref = %f\n", ll_ref);

    to_file(xs, N * T, "xs_lg_ref.txt");
    to_file(ws, N, "ws_lg_ref.txt");

    free(ws);
    free(hx_init);
    free(xs);

}

void test_smc_lg_forget_ref(int N, int T, float* ys_real, float* h_args_l, float scale_step,
        float sigma_step) {

    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* xs = (float*) malloc(N * sizeof(float));
    float* hx_init = (float*) malloc(N * sizeof(float));
    populate_randn(hx_init, N);

    float* ws = (float*) malloc(N * sizeof(float));
    float* d_ws;
    cudaMalloc((void**) &d_ws, N * sizeof(float));

    float ll_forget_ref;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_ref_forget_lg(hx_init, xs, ws, ys_real, N, T, h_args_l, scale_step, sigma_step,
            &ll_forget_ref);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_lg_forget_ref = %f\n", ll_forget_ref);

    free(ws);
    cudaFree(d_ws);
    free(hx_init);
    free(xs);
}

template<class T>
void test_smc_usv_ref(int N, int total_time, T* ys_real, T* h_args_l, T alpha, T sigma) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    T* xs = (float*) malloc(N * total_time * sizeof(float));

    T* ws = (float*) malloc(N * sizeof(float));

    float* hx_init = (float*) malloc(N * sizeof(float));

    populate_randn(hx_init, N);

    float ll_ref;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_ref_usv(hx_init, xs, ws, ys_real, N, total_time, h_args_l, alpha, sigma, &ll_ref);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_usv_ref = %f\n", ll_ref);

    char filename_xs[] = "xs_usv_ref.txt";
    char filename_ws[] = "xs_usv_ref.txt";

    to_file(xs, N * total_time, filename_xs);
    to_file(ws, N, filename_ws);

    free(ws);
    free(hx_init);
    free(xs);

}

template<class T>
void test_smc_usv_forget_ref(int N, int total_time, T* ys_real, T* h_args_l, T alpha, T sigma) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    T* xs = (float*) malloc(N * sizeof(float));

    T* ws = (float*) malloc(N * sizeof(float));

    float* hx_init = (float*) malloc(N * sizeof(float));

    populate_randn(hx_init, N);

    float ll_ref;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_ref_forget_usv(hx_init, xs, ws, ys_real, N, total_time, h_args_l, alpha, sigma, &ll_ref);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_usv_forget_ref = %f\n", ll_ref);

    //    to_file(xs, N * total_time, "xs_usv_ref.txt");
    //    to_file(ws, N, "ws_usv_ref.txt");

    free(ws);
    free(hx_init);
    free(xs);

}

void test_smc_usv(int N, int T, float* ys_real, float* h_args_l, float alpha, float sigma, int nb,
        int nt) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* d_xs;
    float* x_init;

    float* xs = (float*) malloc(N * T * sizeof(float));
    cudaMalloc((void**) &d_xs, N * T * sizeof(float));
    cudaMalloc((void**) &x_init, N * sizeof(float));

    float* ws = (float*) malloc(N * sizeof(float));
    float* d_ws;
    cudaMalloc((void**) &d_ws, N * sizeof(float));

    populate_randn_d(x_init, N);

    float ll;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_usv(x_init, d_xs, d_ws, ys_real, N, T, h_args_l, alpha, sigma, &ll, nb, nt);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_usv = %f\n", ll);

    cudaMemcpy(xs, d_xs, N * T * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ws, d_ws, N * sizeof(float), cudaMemcpyDeviceToHost);
    to_file(xs, N * T, "xs_usv.txt");
    to_file(ws, N, "ws_usv.txt");

    free(xs);
    free(ws);
    cudaFree(d_ws);
    cudaFree(d_xs);
    cudaFree(x_init);
}

void test_smc_usv_forget(int N, int T, float* ys_real, float* h_args_l, float alpha, float sigma,
        int nb, int nt) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* d_xs;
    float* x_init;

    float* xs = (float*) malloc(N * sizeof(float));
    cudaMalloc((void**) &d_xs, N * sizeof(float));
    cudaMalloc((void**) &x_init, N * sizeof(float));

    float* ws = (float*) malloc(N * sizeof(float));
    float* d_ws;
    cudaMalloc((void**) &d_ws, N * sizeof(float));

    populate_randn_d(x_init, N);

    float ll_forget;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_forget_usv(x_init, d_xs, d_ws, ys_real, N, T, h_args_l, alpha, sigma, &ll_forget, nb, nt);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_usv_forget = %f\n", ll_forget);

    free(xs);
    free(ws);
    cudaFree(d_ws);
    cudaFree(d_xs);
    cudaFree(x_init);
}

void test_smc_mvlg_forget(int N, int Dx, int Dy, int T, float* ys_real, float* scale_step,
        float* cov_step, float* h_args_l, int nb, int nt) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* d_xs;
    float* x_init;
    float* d_ys_real;

    float* xs = (float*) malloc(N * Dx * sizeof(float));
    cudaMalloc((void**) &d_xs, N * Dx * sizeof(float));
    cudaMalloc((void**) &x_init, N * Dx * sizeof(float));
    cudaMalloc((void**) &d_ys_real, T * Dy * sizeof(float));
    cudaMemcpy(d_ys_real, ys_real, T * Dy * sizeof(float), cudaMemcpyHostToDevice);

    float* ws = (float*) malloc(N * sizeof(float));
    float* d_ws;
    cudaMalloc((void**) &d_ws, N * sizeof(float));

    float* hx_init = (float*) malloc(N * Dx * sizeof(float));
    matrix_zero(hx_init, N, Dx);
    cudaMemcpy(x_init, hx_init, N * Dx * sizeof(float), cudaMemcpyHostToDevice);
    free(hx_init);

    float ll_forget_D;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_forget_mvlg(x_init, d_xs, d_ws, d_ys_real, N, Dx, Dy, T, h_args_l, scale_step, cov_step,
            &ll_forget_D, nb, nt);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_mvlg_forget = %f\n", ll_forget_D);

    free(ws);
    cudaFree(d_ws);
    free(xs);
    cudaFree(d_xs);
    cudaFree(x_init);
    cudaFree(d_ys_real);
}

void test_smc_mvlg(int N, int Dx, int Dy, int T, float* ys_real, float* scale_step,
        float* cov_step, float* h_args_l, int nb, int nt) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* d_xs;
    float* x_init;
    float* d_ys_real;

    float* xs = (float*) malloc(N * Dx * T * sizeof(float));
    cudaMalloc((void**) &d_xs, N * Dx * T * sizeof(float));
    cudaMalloc((void**) &x_init, N * Dx * sizeof(float));
    cudaMalloc((void**) &d_ys_real, T * Dy * sizeof(float));
    cudaMemcpy(d_ys_real, ys_real, T * Dy * sizeof(float), cudaMemcpyHostToDevice);

    float* ws = (float*) malloc(N * sizeof(float));
    float* d_ws;
    cudaMalloc((void**) &d_ws, N * sizeof(float));

    float* hx_init = (float*) malloc(N * Dx * sizeof(float));
    matrix_zero(hx_init, N, Dx);
    cudaMemcpy(x_init, hx_init, N * Dx * sizeof(float), cudaMemcpyHostToDevice);
    free(hx_init);

    float ll_D;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_mvlg(x_init, d_xs, d_ws, d_ys_real, N, Dx, Dy, T, h_args_l, scale_step, cov_step, &ll_D,
            nb, nt);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_mvlg = %f\n", ll_D);

    cudaMemcpy(xs, d_xs, N * Dx * T * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ws, d_ws, N * sizeof(float), cudaMemcpyDeviceToHost);

    to_file(xs, N * Dx * T, "xs_mvlg.txt");
    to_file(ws, N, "ws_mvlg.txt");

    free(ws);
    cudaFree(d_ws);
    free(xs);
    cudaFree(d_xs);
    cudaFree(x_init);
    cudaFree(d_ys_real);

}

template<class T>
void test_smc_mvlg_ref(int N, int Dx, int Dy, int total_time, T* ys_real, T* scale_step,
        T* cov_step, T* h_args_l) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    T* x_init = (T*) malloc(N * Dx * sizeof(T));
    T* xs = (T*) malloc(N * Dx * total_time * sizeof(T));

    T* ws = (T*) malloc(N * sizeof(T));

    matrix_zero(x_init, N, Dx);

    T ll_ref;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_ref_mvlg(x_init, xs, ws, ys_real, N, Dx, Dy, total_time, h_args_l, scale_step, cov_step,
            &ll_ref);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_mvlg_ref = %f\n", ll_ref);

    char filename_xs[] = "xs_mvlg_ref.txt";
    char filename_ws[] = "ws_mvlg_ref.txt";

    to_file(xs, N * Dx * total_time, filename_xs);
    to_file(ws, N, filename_ws);

    free(ws);
    free(xs);
    free(x_init);

}

void test_smc_mvlg_forget_ref(int N, int Dx, int Dy, int T, float* ys_real, float* scale_step,
        float* cov_step, float* h_args_l) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* x_init = (float*) malloc(N * Dx * sizeof(float));
    float* xs = (float*) malloc(N * Dx * sizeof(float));

    float* ws = (float*) malloc(N * sizeof(float));

    matrix_zero(x_init, N, Dx);

    float ll_ref;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_ref_forget_mvlg(x_init, xs, ws, ys_real, N, Dx, Dy, T, h_args_l, scale_step, cov_step,
            &ll_ref);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_mvlg_forget_ref = %f\n", ll_ref);

    free(ws);
    free(xs);
    free(x_init);

}

template<class T>
void test_smc_mvlg_kalman(int Dx, int Dy, int total_time, T* ys_real, T* scale_step, T* cov_step,
        T* scale_like, T* cov_like) {

    T ll_kalman_D;

    T* kalman_xs = (T*) malloc(Dx * total_time * sizeof(T));

    T* init_xs = (T*) malloc(Dx * sizeof(T));

    for (int i = 0; i < Dx; i++) {
        init_xs[i] = 0;
    }

    kalman(init_xs, kalman_xs, ys_real, Dx, Dy, total_time, scale_step, cov_step, scale_like,
            cov_like, &ll_kalman_D);

    char filename[] = "xs_mvlg_kalman.txt";

    to_file(kalman_xs, Dx * total_time, filename);

    printf("ll_mvlg_kalman = %f\n", ll_kalman_D);

    free(kalman_xs);
    free(init_xs);

}

void test_smc_fsv_forget(int N, int Dx, int Dy, int T, float* ys_real, float* scale_step,
        float* cov_step, float* h_args_l, int nb, int nt) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* d_xs;
    float* x_init;
    float* d_ys_real;

    float* xs = (float*) malloc(N * Dx * sizeof(float));
    cudaMalloc((void**) &d_xs, N * Dx * sizeof(float));
    cudaMalloc((void**) &x_init, N * Dx * sizeof(float));
    cudaMalloc((void**) &d_ys_real, T * Dy * sizeof(float));
    cudaMemcpy(d_ys_real, ys_real, T * Dy * sizeof(float), cudaMemcpyHostToDevice);

    float* ws = (float*) malloc(N * sizeof(float));
    float* d_ws;
    cudaMalloc((void**) &d_ws, N * sizeof(float));

    //	populate_randn_d(x_init, N * D);

    float* hx_init = (float*) malloc(N * Dx * sizeof(float));
    matrix_zero(hx_init, N, Dx);
    cudaMemcpy(x_init, hx_init, N * Dx * sizeof(float), cudaMemcpyHostToDevice);
    free(hx_init);

    float ll_forget_fsv;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_forget_fsv(x_init, d_xs, d_ws, d_ys_real, N, Dx, Dy, T, h_args_l, scale_step, cov_step,
            &ll_forget_fsv, nb, nt);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_fsv_forget = %f\n", ll_forget_fsv);

    free(ws);
    free(xs);
    cudaFree(d_xs);
    cudaFree(x_init);
    cudaFree(d_ys_real);
}

void test_smc_fsv(int N, int Dx, int Dy, int T, float* ys_real, float* scale_step, float* cov_step,
        float* h_args_l, int nb, int nt) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* d_xs;
    float* x_init;
    float* d_ys_real;

    float* xs = (float*) malloc(N * Dx * T * sizeof(float));
    cudaMalloc((void**) &d_xs, N * Dx * T * sizeof(float));
    cudaMalloc((void**) &x_init, N * Dx * sizeof(float));
    cudaMalloc((void**) &d_ys_real, T * Dy * sizeof(float));
    cudaMemcpy(d_ys_real, ys_real, T * Dy * sizeof(float), cudaMemcpyHostToDevice);

    float* ws = (float*) malloc(N * sizeof(float));
    float* d_ws;
    cudaMalloc((void**) &d_ws, N * sizeof(float));

    //	populate_randn_d(x_init, N * D);

    float* hx_init = (float*) malloc(N * Dx * sizeof(float));
    matrix_zero(hx_init, N, Dx);
    cudaMemcpy(x_init, hx_init, N * Dx * sizeof(float), cudaMemcpyHostToDevice);
    free(hx_init);

    float ll_forget_fsv;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_fsv(x_init, d_xs, d_ws, d_ys_real, N, Dx, Dy, T, h_args_l, scale_step, cov_step,
            &ll_forget_fsv, nb, nt);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_fsv = %f\n", ll_forget_fsv);

    cudaMemcpy(xs, d_xs, N * Dx * T * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ws, d_ws, N * sizeof(float), cudaMemcpyDeviceToHost);
    to_file(xs, N * Dx * T, "xs_fsv.txt");
    to_file(ws, N, "ws_fsv.txt");

    free(ws);
    cudaFree(d_ws);
    free(xs);
    cudaFree(d_xs);
    cudaFree(x_init);
    cudaFree(d_ys_real);

}

void test_smc_fsv_ref(int N, int Dx, int Dy, int T, float* ys_real, float* scale_step,
        float* cov_step, float* h_args_l) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* x_init = (float*) malloc(N * Dx * sizeof(float));

    float* xs = (float*) malloc(N * Dx * T * sizeof(float));

    float* ws = (float*) malloc(N * sizeof(float));

    matrix_zero(x_init, N, Dx);

    float ll_forget_fsv;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_ref_fsv(x_init, xs, ws, ys_real, N, Dx, Dy, T, h_args_l, scale_step, cov_step,
            &ll_forget_fsv);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_fsv_ref = %f\n", ll_forget_fsv);

    to_file(xs, N * Dx * T, "xs_fsv.txt");
    to_file(ws, N, "ws_fsv.txt");

    free(ws);
    free(xs);
    free(x_init);
}

void test_smc_fsv_forget_ref(int N, int Dx, int Dy, int T, float* ys_real, float* scale_step,
        float* cov_step, float* h_args_l) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    float* x_init = (float*) malloc(N * Dx * sizeof(float));

    float* xs = (float*) malloc(N * Dx * sizeof(float));

    float* ws = (float*) malloc(N * sizeof(float));

    matrix_zero(x_init, N, Dx);

    float ll_forget_fsv;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_ref_forget_fsv(x_init, xs, ws, ys_real, N, Dx, Dy, T, h_args_l, scale_step, cov_step,
            &ll_forget_fsv);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_fsv_forget_ref = %f\n", ll_forget_fsv);

    free(ws);
    free(xs);
    free(x_init);
}

void test_smc_fsv_ref(int N, int Dx, int Dy, int T, double* ys_real, double* scale_step,
        double* cov_step, double* h_args_l) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    double* x_init = (double*) malloc(N * Dx * sizeof(double));

    double* xs = (double*) malloc(N * Dx * T * sizeof(double));

    double* ws = (double*) malloc(N * sizeof(double));

    matrix_zero(x_init, N, Dx);

    double ll_forget_fsv;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_ref_fsv(x_init, xs, ws, ys_real, N, Dx, Dy, T, h_args_l, scale_step, cov_step,
            &ll_forget_fsv);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_fsv_ref = %f\n", ll_forget_fsv);

    to_file(xs, N * Dx * T, "xs_fsv.txt");
    to_file(ws, N, "ws_fsv.txt");

    free(ws);
    free(xs);
    free(x_init);
}

void test_smc_fsv_forget_ref(int N, int Dx, int Dy, int T, double* ys_real, double* scale_step,
        double* cov_step, double* h_args_l) {
    unsigned int hTimer;
    double time;
    cutCreateTimer(&hTimer);

    double* x_init = (double*) malloc(N * Dx * sizeof(double));

    double* xs = (double*) malloc(N * Dx * sizeof(double));

    double* ws = (double*) malloc(N * sizeof(double));

    matrix_zero(x_init, N, Dx);

    double ll_forget_fsv;

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    smc_ref_forget_fsv(x_init, xs, ws, ys_real, N, Dx, Dy, T, h_args_l, scale_step, cov_step,
            &ll_forget_fsv);

    cutStopTimer(hTimer);
    time = cutGetTimerValue(hTimer);
    printf("Time = %f, ", time);

    printf("ll_fsv_forget_ref = %f\n", ll_forget_fsv);

    free(ws);
    free(xs);
    free(x_init);
}

void test_1D(int N, int T, int nb, int nt) {
    float sigma_like = 1.0;
    float sigma_step = 1.0;
    float scale_step = 1.0;

    float* xs_real = (float*) malloc(T * sizeof(float));
    float* ys_real = (float*) malloc(T * sizeof(float));

    generate_data(xs_real, ys_real, T, sigma_step, sigma_like);

    to_file(xs_real, T, "xs_real_lg.txt");
    to_file(ys_real, T, "ys_real_lg.txt");

    test_smc_lg_kalman(T, ys_real, sigma_like, sigma_step);

    float h_args_l[2];
    compute_c1_c2(sigma_like, h_args_l[0], h_args_l[1]);

    test_smc_lg(N, T, ys_real, h_args_l, scale_step, sigma_step, nb, nt);

    test_smc_lg_forget(N, T, ys_real, h_args_l, scale_step, sigma_step, nb, nt);

    test_smc_lg_ref(N, T, ys_real, h_args_l, scale_step, sigma_step);

    test_smc_lg_forget_ref(N, T, ys_real, h_args_l, scale_step, sigma_step);

    free(xs_real);
    free(ys_real);
}

void test_usv(int N, int T, int nb, int nt) {
    float alpha = 0.9f;
    float sigma = 1.0f;
    float beta = 1.0f;

    float* xs_real = (float*) malloc(T * sizeof(float));
    float* ys_real = (float*) malloc(T * sizeof(float));

    generate_data_usv(xs_real, ys_real, T, alpha, sigma, beta);

    to_file(xs_real, T, "xs_real_usv.txt");
    to_file(ys_real, T, "ys_real_usv.txt");

    float h_args_l[1];
    h_args_l[0] = beta;

    //    kill_rng();
    //    seed_rng(16384, 32, 128);
    //
    //    test_smc_usv(N, T, ys_real, h_args_l, alpha, sigma, nb, nt);

    kill_rng();
    seed_rng(16384, 32, 128);

    test_smc_usv_forget(N, T, ys_real, h_args_l, alpha, sigma, nb, nt);

    //    kill_rng();
    //    seed_rng(16384, 32, 128);
    //
    //    test_smc_usv_ref(N, T, ys_real, h_args_l, alpha, sigma);

    kill_rng();
    seed_rng(16384, 32, 128);

    test_smc_usv_forget_ref(N, T, ys_real, h_args_l, alpha, sigma);

    //    test_smc_lg_forget(N, T, ys_real, h_args_l, sigma_step, nb, nt);
    //
    //    test_smc_lg_ref(N, T, ys_real, h_args_l, sigma_step);
    //
    //    test_smc_lg_forget_ref(N, T, ys_real, h_args_l, sigma_step);

    free(xs_real);
    free(ys_real);
}

void test_2D(int N, int T, int nb, int nt) {
    const int D = 2;

    float scale_step[D * D] = { 0.5f, 0.0f, 0.0f, 0.5f };
    float cov_step[D * D] = { 1.0f, 0.8f, 0.8f, 1.0f };
    float scale_like[D * D] = { 1.0f, 0.0f, 0.0f, 1.0f };
    float cov_like[D * D] = { 0.5f, 0.0f, 0.0f, 0.5f };

    float* xs_real = (float*) malloc(T * D * sizeof(float));
    float* ys_real = (float*) malloc(T * D * sizeof(float));

    generate_data_mv(xs_real, ys_real, D, D, T, scale_step, cov_step, scale_like, cov_like);

    to_file(xs_real, T * D, "xs_real_mvlg.txt");
    to_file(ys_real, T * D, "ys_real_mvlg.txt");

    test_smc_mvlg_kalman(D, D, T, ys_real, scale_step, cov_step, scale_like, cov_like);

    float h_args_l[1 + D * D + D * D];

    compute_c1_c2(cov_like, D, h_args_l[0], h_args_l + 1);

    for (int i = 0; i < D * D; i++) {
        h_args_l[1 + D * D + i] = scale_like[i];
    }

    test_smc_mvlg(N, D, D, T, ys_real, scale_step, cov_step, h_args_l, nb, nt);

    test_smc_mvlg_forget(N, D, D, T, ys_real, scale_step, cov_step, h_args_l, nb, nt);

    test_smc_mvlg_ref(N, D, D, T, ys_real, scale_step, cov_step, h_args_l);

    test_smc_mvlg_forget_ref(N, D, D, T, ys_real, scale_step, cov_step, h_args_l);

    free(xs_real);
    free(ys_real);
}

void test_3by5D(int N, int T, int nb, int nt, int n_burn_filter) {
    const int Dx = 3;
    const int Dy = 5;

    float scale_step[Dx * Dx] = { 0.5f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.5f };
    float cov_step[Dx * Dx] = { 1.0f, 0.8f, 0.0f, 0.8f, 1.0f, 0.4f, 0.0f, 0.4f, 1.0f };
    float scale_like[Dy * Dx] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.4f, 0.3f,
            0.3f, 0.2f, 0.5f, 0.3f };
    float cov_like[Dy * Dy];
    matrix_identity(cov_like, Dy);
    matrix_times(cov_like, cov_like, 0.5f, Dy, Dy);

    float* xs_real = (float*) malloc(T * Dx * sizeof(float));
    float* ys_real = (float*) malloc(T * Dy * sizeof(float));

    kill_rng();
    seed_rng(16384, 32, 128);

    generate_data_mv(xs_real, ys_real, Dx, Dy, T, scale_step, cov_step, scale_like, cov_like);

    to_file(xs_real, T * Dx, "xs_real_mvlg.txt");
    to_file(ys_real, T * Dy, "ys_real_mvlg.txt");

    test_smc_mvlg_kalman(Dx, Dy, T, ys_real, scale_step, cov_step, scale_like, cov_like);

    float h_args_l[1 + Dy * Dy + Dy * Dx];

    compute_c1_c2(cov_like, Dy, h_args_l[0], h_args_l + 1);

    for (int i = 0; i < Dy * Dx; i++) {
        h_args_l[1 + Dy * Dy + i] = scale_like[i];
    }

    seed_rng(n_burn_filter, 32, 128);

    test_smc_mvlg(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l, nb, nt);

    seed_rng(n_burn_filter, 32, 128);

    test_smc_mvlg_forget(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l, nb, nt);

    seed_rng(n_burn_filter, 32, 128);

    test_smc_mvlg_ref(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l);

    seed_rng(n_burn_filter, 32, 128);

    test_smc_mvlg_forget_ref(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l);

    free(xs_real);
    free(ys_real);
}

void test_3by5D_double(int N, int T, int n_burn_filter) {
    const int Dx = 3;
    const int Dy = 5;

    double scale_step[Dx * Dx] = { 0.5f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.5f };
    double cov_step[Dx * Dx] = { 1.0f, 0.8f, 0.0f, 0.8f, 1.0f, 0.4f, 0.0f, 0.4f, 1.0f };
    double scale_like[Dy * Dx] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.4f,
            0.3f, 0.3f, 0.2f, 0.5f, 0.3f };
    double cov_like[Dy * Dy];
    matrix_identity(cov_like, Dy);
    matrix_times(cov_like, cov_like, 0.5, Dy, Dy);

    double* xs_real = (double*) malloc(T * Dx * sizeof(double));
    double* ys_real = (double*) malloc(T * Dy * sizeof(double));

    kill_rng();
    seed_rng(16384, 32, 128);

    generate_data_mv(xs_real, ys_real, Dx, Dy, T, scale_step, cov_step, scale_like, cov_like);

    to_file(xs_real, T * Dx, "xs_real_mvlg.txt");
    to_file(ys_real, T * Dy, "ys_real_mvlg.txt");

    test_smc_mvlg_kalman(Dx, Dy, T, ys_real, scale_step, cov_step, scale_like, cov_like);

    double h_args_l[1 + Dy * Dy + Dy * Dx];

    compute_c1_c2(cov_like, Dy, h_args_l[0], h_args_l + 1);

    for (int i = 0; i < Dy * Dx; i++) {
        h_args_l[1 + Dy * Dy + i] = scale_like[i];
    }

    seed_rng(n_burn_filter, 32, 128);

    test_smc_mvlg_ref(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l);

    //    seed_rng(8192, 32, 128);
    //
    //    test_smc_mvlg_forget_ref(N, Dx, Dy, T, ys_real, scale_step, cov_step,
    //            h_args_l);

    free(xs_real);
    free(ys_real);
}

void test_fsv(int N, int T, int nb, int nt) {
    const int Dx = 3;
    const int Dy = 5;

    float scale_step[Dx * Dx];
    matrix_identity(scale_step, Dx);
    matrix_times(scale_step, scale_step, 0.9f, Dx, Dx);

    float cov_step[Dx * Dx] = { 0.5f, 0.2f, 0.1f, 0.2f, 0.5f, 0.2f, 0.1f, 0.2f, 0.5f };

    float Psi[Dy * Dy];
    matrix_identity(Psi, Dy);
    matrix_times(Psi, Psi, 0.5f, Dy, Dy);

    float B[Dy * Dx] = { 1.0f, 0.0f, 0.0f, 0.5f, 1.0f, 0.0f, 0.5f, 0.5f, 1.0f, 0.2f, 0.6f, 0.3f,
            0.8f, 0.7f, 0.5f };

    float* xs_real = (float*) malloc(T * Dx * sizeof(float));
    float* ys_real = (float*) malloc(T * Dy * sizeof(float));

    kill_rng();
    seed_rng(16384, 32, 128);

    generate_data_fsv(xs_real, ys_real, Dx, Dy, T, scale_step, cov_step, Psi, B);

    printf("%f\n", xs_real[T - 1]);

    to_file(xs_real, T * Dx, "fsv_xs_real.txt");
    to_file(ys_real, T * Dy, "fsv_ys_real.txt");

    float h_args_l[Dy * Dx + Dx * Dy + Dy * Dy];
    matrix_transpose(B, h_args_l + Dy * Dx, Dy, Dx);
    for (int i = 0; i < Dy * Dx; i++) {
        h_args_l[i] = B[i];
    }
    for (int i = 0; i < Dy * Dy; i++) {
        h_args_l[2 * Dy * Dx + i] = Psi[i];
    }

    kill_rng();
    seed_rng(8192, 32, 128);

    test_smc_fsv(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l, nb, nt);

    kill_rng();
    seed_rng(8192, 32, 128);

    test_smc_fsv_forget(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l, nb, nt);

    kill_rng();
    seed_rng(8192, 32, 128);

    test_smc_fsv_ref(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l);

    kill_rng();
    seed_rng(8192, 32, 128);

    test_smc_fsv_forget_ref(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l);

    free(xs_real);
    free(ys_real);
}

void test_fsv_2_3(int N, int T, int nb, int nt) {
    const int Dx = 2;
    const int Dy = 3;

    float scale_step[Dx * Dx];
    matrix_identity(scale_step, Dx);
    matrix_times(scale_step, scale_step, 0.9f, Dx, Dx);

    float cov_step[Dx * Dx] = { 0.5f, 0.2f, 0.2f, 0.5f };

    float Psi[Dy * Dy];
    matrix_identity(Psi, Dy);
    matrix_times(Psi, Psi, 0.5f, Dy, Dy);

    float B[Dy * Dx] = { 1.0f, 0.0f, 0.3f, 0.7f, 0.6f, 0.4f };

    float* xs_real = (float*) malloc(T * Dx * sizeof(float));
    float* ys_real = (float*) malloc(T * Dy * sizeof(float));

    kill_rng();
    seed_rng(16384, 32, 128);

    generate_data_fsv(xs_real, ys_real, Dx, Dy, T, scale_step, cov_step, Psi, B);

    printf("%f\n", xs_real[T - 1]);

    to_file(xs_real, T * Dx, "fsv_xs_real.txt");
    to_file(ys_real, T * Dy, "fsv_ys_real.txt");

    float h_args_l[Dy * Dx + Dx * Dy + Dy * Dy];
    matrix_transpose(B, h_args_l + Dy * Dx, Dy, Dx);
    for (int i = 0; i < Dy * Dx; i++) {
        h_args_l[i] = B[i];
    }
    for (int i = 0; i < Dy * Dy; i++) {
        h_args_l[2 * Dy * Dx + i] = Psi[i];
    }

    kill_rng();
    seed_rng(8192, 32, 128);

    test_smc_fsv(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l, nb, nt);

    kill_rng();
    seed_rng(8192, 32, 128);

    test_smc_fsv_forget(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l, nb, nt);

    //  kill_rng();
    //  seed_rng(8192, 32, 128);
    //
    //  test_smc_fsv_ref(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l);

    kill_rng();
    seed_rng(8192, 32, 128);

    test_smc_fsv_forget_ref(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l);

    free(xs_real);
    free(ys_real);
}

void test_fsv_2_2(int N, int T, int nb, int nt) {
    const int Dx = 2;
    const int Dy = 2;

    float scale_step[Dx * Dx];
    matrix_identity(scale_step, Dx);
    matrix_times(scale_step, scale_step, 0.9f, Dx, Dx);

    float cov_step[Dx * Dx] = { 0.5f, 0.2f, 0.2f, 0.5f };

    float Psi[Dy * Dy];
    matrix_identity(Psi, Dy);
    matrix_times(Psi, Psi, 0.5f, Dy, Dy);

    float B[Dy * Dx] = { 1.0f, 0.0f, 0.3f, 0.7f };

    float* xs_real = (float*) malloc(T * Dx * sizeof(float));
    float* ys_real = (float*) malloc(T * Dy * sizeof(float));

    kill_rng();
    seed_rng(16384, 32, 128);

    generate_data_fsv(xs_real, ys_real, Dx, Dy, T, scale_step, cov_step, Psi, B);

    printf("%f\n", xs_real[T - 1]);

    to_file(xs_real, T * Dx, "fsv_xs_real.txt");
    to_file(ys_real, T * Dy, "fsv_ys_real.txt");

    float h_args_l[Dy * Dx + Dx * Dy + Dy * Dy];
    matrix_transpose(B, h_args_l + Dy * Dx, Dy, Dx);
    for (int i = 0; i < Dy * Dx; i++) {
        h_args_l[i] = B[i];
    }
    for (int i = 0; i < Dy * Dy; i++) {
        h_args_l[2 * Dy * Dx + i] = Psi[i];
    }

    //    kill_rng();
    //    seed_rng(8192, 32, 128);
    //
    //    test_smc_fsv(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l, nb, nt);

    kill_rng();
    seed_rng(8192, 32, 128);

    test_smc_fsv_forget(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l, nb, nt);

    //  kill_rng();
    //  seed_rng(8192, 32, 128);
    //
    //  test_smc_fsv_ref(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l);

    kill_rng();
    seed_rng(8192, 32, 128);

    test_smc_fsv_forget_ref(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l);

    free(xs_real);
    free(ys_real);
}

void test_fsv_1_1(int N, int T, int nb, int nt) {
    const int Dx = 1;
    const int Dy = 1;

    float scale_step[Dx * Dx];
    matrix_identity(scale_step, Dx);
    matrix_times(scale_step, scale_step, 0.9f, Dx, Dx);

    float cov_step[Dx * Dx] = { 0.5f };

    float Psi[Dy * Dy];
    matrix_identity(Psi, Dy);
    matrix_times(Psi, Psi, 0.5f, Dy, Dy);

    float B[Dy * Dx] = { 1.0f };

    float* xs_real = (float*) malloc(T * Dx * sizeof(float));
    float* ys_real = (float*) malloc(T * Dy * sizeof(float));

    kill_rng();
    seed_rng(16384, 32, 128);

    generate_data_fsv(xs_real, ys_real, Dx, Dy, T, scale_step, cov_step, Psi, B);

    printf("%f\n", xs_real[T - 1]);

    to_file(xs_real, T * Dx, "fsv_xs_real.txt");
    to_file(ys_real, T * Dy, "fsv_ys_real.txt");

    float h_args_l[Dy * Dx + Dx * Dy + Dy * Dy];
    matrix_transpose(B, h_args_l + Dy * Dx, Dy, Dx);
    for (int i = 0; i < Dy * Dx; i++) {
        h_args_l[i] = B[i];
    }
    for (int i = 0; i < Dy * Dy; i++) {
        h_args_l[2 * Dy * Dx + i] = Psi[i];
    }

    //    kill_rng();
    //    seed_rng(8192, 32, 128);
    //
    //    test_smc_fsv(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l, nb, nt);

    kill_rng();
    seed_rng(8192, 32, 128);

    test_smc_fsv_forget(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l, nb, nt);

    //  kill_rng();
    //  seed_rng(8192, 32, 128);
    //
    //  test_smc_fsv_ref(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l);

    kill_rng();
    seed_rng(8192, 32, 128);

    test_smc_fsv_forget_ref(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l);

    free(xs_real);
    free(ys_real);
}

// HOST ONLY
void test_fsv_double(int N, int T) {
    const int Dx = 3;
    const int Dy = 5;

    double scale_step[Dx * Dx];
    matrix_identity(scale_step, Dx);
    matrix_times(scale_step, scale_step, 0.9, Dx, Dx);

    double cov_step[Dx * Dx] = { 0.5, 0.2, 0.1, 0.2, 0.5, 0.2, 0.1, 0.2, 0.5 };

    double Psi[Dy * Dy];
    matrix_identity(Psi, Dy);
    matrix_times(Psi, Psi, 0.5, Dy, Dy);

    double
            B[Dy * Dx] = { 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.5, 0.5, 1.0, 0.2, 0.6, 0.3, 0.8, 0.7,
                    0.5 };

    double* xs_real = (double*) malloc(T * Dx * sizeof(double));
    double* ys_real = (double*) malloc(T * Dy * sizeof(double));

    kill_rng();
    seed_rng(16384, 32, 128);

    generate_data_fsv(xs_real, ys_real, Dx, Dy, T, scale_step, cov_step, Psi, B);

    to_file(xs_real, T * Dx, "fsv_xs_real.txt");
    to_file(ys_real, T * Dy, "fsv_ys_real.txt");

    printf("%f\n", xs_real[T - 1]);

    double h_args_l[Dy * Dx + Dx * Dy + Dy * Dy];
    matrix_transpose(B, h_args_l + Dy * Dx, Dy, Dx);
    for (int i = 0; i < Dy * Dx; i++) {
        h_args_l[i] = B[i];
    }
    for (int i = 0; i < Dy * Dy; i++) {
        h_args_l[2 * Dy * Dx + i] = Psi[i];
    }

    //	kill_rng();
    //	seed_rng(8192, 32, 128);
    //
    //	test_smc_fsv_ref(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l);

    kill_rng();
    seed_rng(8192, 32, 128);

    test_smc_fsv_forget_ref(N, Dx, Dy, T, ys_real, scale_step, cov_step, h_args_l);

    free(xs_real);
    free(ys_real);
}

int main(int argc, char **argv) {

    //    int N = 8192;
    //    int N = 16384;
    //    int N = 32768;
    //    int N = 65536;
    //    int N = 262144;

    //    int N = 8192;
    //   int N = 16384;
    //    int N = 32768;
            int N = 65536;

//    int N = 131072;

    int nb = 256;
    int nt = 64;

    int T = 200;

    seed_rng(8192, 32, 128);

    scan_init(N);

    //        test_1D(N, T, nb, nt);
    //    test_2D(N, T, nb, nt);
    //        test_3by5D(N, T, nb, nt);
    //    test_fsv(4096, T, nb, nt);
    //	test_fsv(8192, T, 128, nt);
    //	test_fsv(16384, T, nb, nt);
    //	test_fsv(32768, T, nb, nt);
    //	test_fsv(65536, T, nb, nt);
    //	test_fsv(131072, T, nb, nt);

    test_fsv(N, T, nb, nt);
    //	    test_fsv_double(N, T);

    //    test_3by5D(N, T, nb, nt, 8192*4);
    //
    //    test_3by5D_double(N, T, 8192*4);

    //    test_1D(N, T, nb, nt);

    //    test_usv(N, T, nb, nt);

    //    test_fsv_2_3(N, T, nb, nt);

    //        test_fsv_2_2(N, T, nb, nt);

    //    test_fsv_1_1(N, T, nb, nt);

    kill_rng();
    scan_destroy();
}
