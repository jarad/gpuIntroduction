/*
 * demo_log_speed.cu
 *
 *  Created on: 07-Apr-2009
 *      Author: alee
 */

#include <cutil.h>
#include <stdio.h>

__global__ void logtest(int size, float* d_array, int M) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    int i, j;
    float x;
    for (i = tid; i < size; i += tt) {
        x = ((float) i + 1) / size;
        for (j = 0; j < M; j++) {
            x = logf(x);
            x = expf(x);
        }
        d_array[i] = x;
    }
}

void logtestref(int size, float* array, int M) {
    float x;

    for (int i = 0; i < size; i++) {
        x = ((float) i + 1) / size;
        for (int j = 0; j < M; j++) {
            x = logf(x);
            x = expf(x);
        }
        array[i] = x;
    }
}

void testLogSpeed(int N, int M, int nb, int nt) {
    unsigned int hTimer;
    double gtime, ctime;
    cutCreateTimer(&hTimer);
    float* array = (float*) malloc(N * sizeof(float));
    float* d_array;
    cudaMalloc((void**) &d_array, N * sizeof(float));

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);
    logtest<<<nb,nt>>>(N, d_array, M);
    cudaThreadSynchronize();
    cutStopTimer(hTimer);
    gtime = cutGetTimerValue(hTimer);
    printf("log test time = %f\n", gtime);

    cutResetTimer(hTimer);
    cutStartTimer(hTimer);
    logtestref(N, array, M);
    cutStopTimer(hTimer);
    ctime = cutGetTimerValue(hTimer);
    printf("ref log test time = %f\n", ctime);

    printf("speedup = %f\n", ctime / gtime);

    free(array);
    cudaFree(d_array);
}

int main(int argc, char **argv) {
    int nb = 256;
    int nt = 128;
//    int N = 65536;
    int N = 262144;
    int M = 1024;
    testLogSpeed(N, M, nb, nt);
}
