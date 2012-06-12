#include <stdio.h>
#include "func.h"
#include <float.h>
#include "test_functions.h"

#ifdef USE_CUDPP
#include "cudpp.h"
CUDPPHandle scanplan = 0;
#endif

__global__ void scan_gpu_step1(int N, float* d_idata, float* d_odata, float* sums) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    int M = N / tt;
    float sum = 0;
    int i;
    int s = M * tid;
    for (i = 0; i < M; i++) {
        sum += d_idata[s + i];
        d_odata[s + i] = sum;
    }
    sums[tid] = sum;
}

__global__ void scan_gpu_step2(int N, float* d_odata, float* sums) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    int M = N / tt;
    int i;
    int s = M * tid;
    if (tid == 0) {
        return;
    }
    float toadd = sums[tid - 1];
    for (i = 0; i < M; i++) {
        d_odata[s + i] += toadd;
    }
}

#ifdef USE_CUDPP

void scan_init(int N) {
    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    cudppPlan(&scanplan, config, N, 1, 0);
}

void scan_destroy() {
    cudppDestroyPlan(scanplan);
}

void scan(int N, float *d_idata, float* d_odata, int nb, int nt) {
    cudppScan(scanplan, d_odata, d_idata, N);
}

#else

void scan_init(int N) {

}

void scan_destroy() {

}

void scan(int N, float *d_idata, float* d_odata, int nb, int nt) {
    float* sums;
    int tt = nb * nt;
    cudaMalloc((void**) &sums, tt * sizeof(float));

    scan_gpu_step1<<<nb, nt>>>(N, d_idata, d_odata, sums);

    float* h_sums = (float*) malloc(tt * sizeof(float));
    cudaMemcpy(h_sums, sums, tt * sizeof(float), cudaMemcpyDeviceToHost);

    double c = h_sums[0];
    for (int i = 1; i < tt; i++) {
        c += h_sums[i];
        h_sums[i] = (float) c;
        //        h_sums[i] += h_sums[i - 1];
    }

    cudaMemcpy(sums, h_sums, tt * sizeof(float), cudaMemcpyHostToDevice);

    scan_gpu_step2<<<nb, nt>>>(N, d_odata, sums);

    cudaThreadSynchronize();

    cudaFree(sums);
    free(h_sums);

}

#endif

//void scan(int N, float *d_idata, float* d_odata, int nb, int nt) {
//    unsigned int extra_space = N / NUM_BANKS;
//    const unsigned int shared_mem_size = sizeof(float) * (N
//                + extra_space);
//    int nbr = N / 2 / 64;
//    int ntr = 64;
//    scan_best<<<nbr,ntr,shared_mem_size>>>(d_odata, d_idata, N);
////    scan_best<<<N/2,1,shared_mem_size>>>(d_odata, d_idata, N);
//    cudaThreadSynchronize();
//
//}

__global__ void scan_log_gpu_step1(int N, float* d_idata, float* d_odata, float* sums) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    int M = N / tt;
    float sum;
    int i;
    int s = M * tid;
    //    sum = d_idata[s];
    //    d_odata[s] = expf(sum);
    //    for (i = 1; i < M; i++) {
    //        sum = logsumexp(sum, d_idata[s + i]);
    //        d_odata[s + i] = expf(sum);
    //    }
    //    sums[tid] = expf(sum);
    sum = 0;
    for (i = 0; i < M; i++) {
        sum += expf(d_idata[s + i]);
        d_odata[s + i] = sum;
    }
    sums[tid] = sum;
}

void scan_log(int N, float *d_idata, float* d_odata, int nb, int nt) {
    float* sums;
    int tt = nb * nt;
    cudaMalloc((void**) &sums, tt * sizeof(float));

    scan_log_gpu_step1<<<nb, nt>>>(N, d_idata, d_odata, sums);

    float* h_sums = (float*) malloc(tt * sizeof(float));
    cudaMemcpy(h_sums, sums, tt * sizeof(float), cudaMemcpyDeviceToHost);

    double c = h_sums[0];
    for (int i = 1; i < tt; i++) {
        c += h_sums[i];
        h_sums[i] = (float) c;
    }

    cudaMemcpy(sums, h_sums, tt * sizeof(float), cudaMemcpyHostToDevice);

    scan_gpu_step2<<<nb, nt>>>(N, d_odata, sums);

    cudaThreadSynchronize();

    cudaFree(sums);
    free(h_sums);

}
