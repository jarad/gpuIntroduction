#include "sharedmem.cuh"
#include "reduce.h"
#include <stdio.h>

#define square(x) ((x) * (x))

__global__ void reduce_gpu(int size, float* d_idata, float* d_odata) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    int i;
    float s = 0;
    for (i = tid; i < size; i += tt) {
        s += d_idata[i];
    }
    d_odata[tid] = s;
}

__global__ void reduce_gpu_ss(int size, float* d_idata, float* d_odata) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    int i;
    float s = 0;
    for (i = tid; i < size; i += tt) {
        s += square(d_idata[i]);
    }
    d_odata[tid] = s;
}

__global__ void reduce_gpu(int size, int D, float* d_idata, float* d_odata) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    int i;
    int j;

    SharedMemory<float> smem;
    float* sdata = smem.getPointer();
    const int tidib = threadIdx.x;

    for (j = 0; j < D; j++) {
        sdata[tidib * D + j] = 0;
    }
    for (i = tid; i < size; i += tt) {
        for (j = 0; j < D; j++) {
            sdata[tidib * D + j] += d_idata[i * D + j];
        }
    }
    for (j = 0; j < D; j++) {
        d_odata[tid * D + j] = sdata[tidib * D + j];
    }

}

void reduce(int size, float* d_idata, float& h_sum, int nb, int nt) {
    const int tt = nb * nt;

    float* d_sum;
    cudaMalloc((void **) &d_sum, tt * sizeof(float));

    float* tmp = (float*) malloc(tt * sizeof(float));

    reduce_gpu<<< nb, nt >>>(size, d_idata, d_sum);

    cudaMemcpy(tmp, d_sum, tt * sizeof(float), cudaMemcpyDeviceToHost);

    double sum = 0;
    for (int i = 0; i < tt; i++) {
        sum += tmp[i];
    }

    h_sum = (float) sum;

    cudaFree(d_sum);
    free(tmp);

}

void reduce_ss(int size, float* d_idata, float& h_sum, int nb, int nt) {
    const int tt = nb * nt;

    float* d_sum;
    cudaMalloc((void **) &d_sum, tt * sizeof(float));

    float* tmp = (float*) malloc(tt * sizeof(float));
    //	float* tmp;
    //	cudaMallocHost((void **) &tmp, tt * sizeof(float));

    reduce_gpu_ss<<< nb, nt >>>(size, d_idata, d_sum);

    cudaMemcpy(tmp, d_sum, tt * sizeof(float), cudaMemcpyDeviceToHost);

    double sum = 0;
    for (int i = 0; i < tt; i++) {
        sum += tmp[i];
    }

    h_sum = (float) sum;

    cudaFree(d_sum);
    //	cudaFreeHost(tmp);
    free(tmp);

}

void reduce(int size, int D, float* d_idata, float* h_sum, int nb, int nt) {
    const int tt = nb * nt;

    float* d_sum;
    cudaMalloc((void **) &d_sum, tt * D * sizeof(float));

    float* tmp = (float*) malloc(tt * D * sizeof(float));
    //	float* tmp;
    //	cudaMallocHost((void **) &tmp, tt * D * sizeof(float));

    if (nt * D * sizeof(float) >= 16384) {
        printf("not enough shared memory!\n");
    }

    reduce_gpu<<< nb, nt, nt * D * sizeof(float) >>>(size, D, d_idata, d_sum);

    cudaMemcpy(tmp, d_sum, tt * D * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < D; i++) {
        h_sum[i] = 0;
    }

    for (int i = 0; i < tt; i++) {
        for (int j = 0; j < D; j++) {
            h_sum[j] += tmp[i * D + j];
        }
    }

    cudaFree(d_sum);
    //	cudaFreeHost(tmp);
    free(tmp);

}

//__global__ void reduce_gpu2(int size, float *d_idata, float *d_odata) {
//
//    SharedMemory<float> smem;
//    float *sdata = smem.getPointer();
//
//    // perform first level of reduction,
//    // reading from global memory, writing to shared memory
//    int threads_per_block = blockDim.x;
//    int blockId = blockIdx.x;
//    int numBlocks = gridDim.x;
//    unsigned int tid = threadIdx.x;
//    unsigned int i = blockId * (threads_per_block * 2) + tid;
//    unsigned int gridSize = threads_per_block * 2 * numBlocks;
//    sdata[tid] = 0;
//
//    // we reduce multiple elements per thread.  The number is determined by the
//    // number of active thread blocks (via gridSize).  More blocks will result
//    // in a larger gridSize and therefore fewer elements per thread
//    while (i < size) {
//        sdata[tid] += d_idata[i] + d_idata[i+threads_per_block];
//        i += gridSize;
//    }
//    __syncthreads();
//
//    if (threads_per_block >= 512) {
//        if (tid < 256) {
//            sdata[tid] += sdata[tid + 256];
//        }
//        __syncthreads();
//    }
//    if (threads_per_block >= 256) {
//        if (tid < 128) {
//            sdata[tid] += sdata[tid + 128];
//        }
//        __syncthreads();
//    }
//    if (threads_per_block >= 128) {
//        if (tid < 64) {
//            sdata[tid] += sdata[tid + 64];
//        }
//        __syncthreads();
//    }
//
//#ifndef __DEVICE_EMULATION__
//    // TODO: WHY!??
//    if (tid < 32)
//#endif
//    {
//        if (threads_per_block >= 64) {
//            sdata[tid] += sdata[tid + 32];
//            __syncthreads();
//        }
//        if (threads_per_block >= 32) {
//            sdata[tid] += sdata[tid + 16];
//            __syncthreads();
//        }
//        if (threads_per_block >= 16) {
//            sdata[tid] += sdata[tid + 8];
//            __syncthreads();
//        }
//        if (threads_per_block >= 8) {
//            sdata[tid] += sdata[tid + 4];
//            __syncthreads();
//        }
//        if (threads_per_block >= 4) {
//            sdata[tid] += sdata[tid + 2];
//            __syncthreads();
//        }
//        if (threads_per_block >= 2) {
//            sdata[tid] += sdata[tid + 1];
//            __syncthreads();
//        }
//    }
//
//    // write result for this block to global mem
//    if (tid == 0) {
//        d_odata[blockIdx.x] = sdata[0];
//    }
//
//}
//
//void reduce(int size, float *d_idata, float& h_sum, int nb, int nt) {
//    int smemSize = nt * sizeof(float);
//    //
//    //  N_BLOCKS
//    float* d_sum;
//    cudaMalloc((void **) &d_sum, nb * sizeof(float));
//
//    float* tmp;
//    cudaMallocHost((void **) &tmp, nb * sizeof(float));
//
//    reduce_gpu2<<< nb, nt, smemSize >>>(size, d_idata, d_sum);
//
//    cudaMemcpy(tmp, d_sum, nb * sizeof(float), cudaMemcpyDeviceToHost);
//
//    h_sum = 0;
//    for (int i = 0; i < nb; i++) {
//        h_sum += tmp[i];
//    }
//
//    cudaFree(d_sum);
//    cudaFreeHost(tmp);
//
//}
