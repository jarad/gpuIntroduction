#include <stdio.h>
#include "cutil.h"

#include "rng.h"
#include "reduce.h"
#include "test_functions.h"

#define PI 3.14159265358979f

// p(x) = 1/2 * N(x;-1,0.5^2) + 1/2 * N(x;1.5,0.5^2)
__device__ float target_pdf(float x) {
    return 1.0f / sqrtf(2 * PI) * exp(-(x - 1.5) * (x - 1.5) / 0.5f) + 1.0f / sqrtf(2 * PI) * exp(
            -(x + 1) * (x + 1) / 0.5f);
}

// q(x) = N(x;0,1)
__device__ float proposal_pdf(float x) {
    return 1.0f / sqrtf(2 * PI) * exp(-x * x / 2.0f);
}

float target_pdfh(float x) {
    return 1.0f / sqrtf(2 * PI) * exp(-(x - 1.5f) * (x - 1.5f) / 0.5f) + 1.0f / sqrtf(2 * PI) * ((float) exp(
            -(x + 1) * (x + 1) / 0.5f));
}

float proposal_pdfh(float x) {
    return 1.0f / sqrtf(2 * PI) * ((float) exp(-x * x / 2.0f));
}

__device__ float phi(float x) {
    return x * x;
}

float phih(float x) {
    return x * x;
}

__global__ void is(int N, float* d_array, float* d_array_out) {
    // get thread identifier
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // get total number of threads
    const int tt = blockDim.x * gridDim.x;
    int i;
    float w, x;
    for (i = tid; i < N; i += tt) {
        x = d_array[i];
        w = target_pdf(x) / proposal_pdf(x);
        d_array_out[i] = phi(x) * w;
    }
}

void eg_v1() {
    unsigned int hTimer;
    double ctime, gtime;
    cutCreateTimer(&hTimer);
    seed_rng();

    //    int N = 1048576;
    int N = 16777216;
    int nb = 64;
    int nt = 128;

    float h_sum;

    float* d_array;
    float* d_array_out;
    //        float* d_warray;
    cudaMalloc((void **) &d_array, N * sizeof(float));
    cudaMalloc((void **) &d_array_out, N * sizeof(float));
    //        cudaMalloc((void **) &d_warray, N * sizeof(float));

    populate_randn_d(d_array, N);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    is<<<nb,nt>>>(N, d_array, d_array_out);

    //        cudaThreadSynchronize();

    //        multiply(N, d_array, d_array2, d_warray, nb, nt);

    cudaThreadSynchronize();

    reduce(N, d_array_out, h_sum, nb, nt);

    cudaThreadSynchronize();

    cutStopTimer(hTimer);
    gtime = cutGetTimerValue(hTimer);
    printf("Time = %f\n", gtime);

    printf("RESULT = %f\n", h_sum / N);

    float* array = (float*) malloc(N * sizeof(float));

    cudaMemcpy(array, d_array, N * sizeof(float), cudaMemcpyDeviceToHost);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    double h_sum2 = 0;
    for (int i = 0; i < N; i++) {
        float x = array[i];
        h_sum2 += phih(x) * target_pdfh(x) / proposal_pdfh(x);
    }
    cutStopTimer(hTimer);
    ctime = cutGetTimerValue(hTimer);
    printf("Time = %f\n", ctime);

    printf("RESULT = %f\n", h_sum2 / N);

    printf("speedup = %f\n", ctime / gtime);

    kill_rng();
}

void eg_v2() {
    unsigned int hTimer;
    double ctime, gtime;
    cutCreateTimer(&hTimer);
    seed_rng();

    int N = 16777216;

    float h_sum, result;

    float* d_array;
    float* d_array_out;
    cudaMalloc((void **) &d_array, N * sizeof(float));
    cudaMalloc((void **) &d_array_out, N * sizeof(float));

    float* array = (float*) malloc(N * sizeof(float));
    populate_randn(array, N);
    cudaMemcpy(d_array, array, N * sizeof(float), cudaMemcpyHostToDevice);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    is<<<64,128>>>(N, d_array, d_array_out);

    reduce(N, d_array_out, h_sum, 64, 128);

    cutStopTimer(hTimer);
    gtime = cutGetTimerValue(hTimer);
    printf("Time = %f\n", gtime);

    result = h_sum / N;
    printf("RESULT = %f\n", result);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);

    double h_sum2 = 0;
    for (int i = 0; i < N; i++) {
        float x = array[i];
        h_sum2 += phih(x) * target_pdfh(x) / proposal_pdfh(x);
    }

    cutStopTimer(hTimer);
    ctime = cutGetTimerValue(hTimer);
    printf("Time = %f\n", ctime);

    printf("RESULT = %f\n", h_sum2 / N);

    printf("speedup = %f\n", ctime / gtime);

    free(array);
    cudaFree(d_array);
    cudaFree(d_array_out);

    kill_rng();
}

int main(int argc, char **argv) {
    //    eg_v1();
    eg_v2();
}
