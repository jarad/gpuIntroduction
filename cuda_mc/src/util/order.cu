/*
 * order.cu
 *
 *  Created on: 25-Mar-2009
 *      Author: alee
 */

__global__ void max_gpu(int size, float* d_idata, float* d_odata) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    int i;
    float m = d_idata[tid];
    for (i = tid + tt; i < size; i += tt) {
        m = max(m, d_idata[i]);
    }
    d_odata[tid] = m;
}

__global__ void min_gpu(int size, float* d_idata, float* d_odata) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    int i;
    float m = d_idata[tid];
    for (i = tid + tt; i < size; i += tt) {
        m = min(m, d_idata[i]);
    }
    d_odata[tid] = m;
}

void maximum(int size, float *d_idata, float& h_max, int nb, int nt) {
    const int tt = nb * nt;

    float* d_max;
    cudaMalloc((void **) &d_max, tt * sizeof(float));

    float* tmp = (float*) malloc(tt * sizeof(float));

    max_gpu<<< nb, nt >>>(size, d_idata, d_max);

    cudaMemcpy(tmp, d_max, tt * sizeof(float), cudaMemcpyDeviceToHost);

    h_max = tmp[0];
    for (int i = 1; i < tt; i++) {
        h_max = max(tmp[i], h_max);
    }

    cudaFree(d_max);
    free(tmp);
}

void minimum(int size, float *d_idata, float& h_min, int nb, int nt) {
    const int tt = nb * nt;

    float* d_min;
    cudaMalloc((void **) &d_min, tt * sizeof(float));

    float* tmp = (float*) malloc(tt * sizeof(float));

    min_gpu<<< nb, nt >>>(size, d_idata, d_min);

    cudaMemcpy(tmp, d_min, tt * sizeof(float), cudaMemcpyDeviceToHost);

    h_min = tmp[0];
    for (int i = 1; i < tt; i++) {
        h_min = min(tmp[i], h_min);
    }

    cudaFree(d_min);
    free(tmp);
}
