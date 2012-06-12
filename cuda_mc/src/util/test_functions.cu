/*
 * test_functions.c
 *
 *  Created on: 02-Feb-2009
 *      Author: alee
 */

#include "test_functions.h"
#include "matrix.ch"
#include "sharedmem.cuh"
#include "func.h"

__global__ void add_gpu(int size, float *d_data, float* d_output, float to_add) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int nt = blockDim.x * gridDim.x;
    int i;
    for (i = 0; i * nt + tid < size; i++) {
        d_output[i * nt + tid] = d_data[i * nt + tid] + to_add;
    }
}

__global__ void add_gpu(int size, float *d_data, float* d_output, float* to_add) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int nt = blockDim.x * gridDim.x;
    int i;
    for (i = 0; i * nt + tid < size; i++) {
        d_output[i * nt + tid] = d_data[i * nt + tid] + to_add[i * nt + tid];
    }
}

__global__ void multiply_gpu(int size, float *d_data, float* d_output, float to_multiply) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int nt = blockDim.x * gridDim.x;
    int i;
    for (i = 0; i * nt + tid < size; i++) {
        d_output[i * nt + tid] = d_data[i * nt + tid] * to_multiply;
    }
}

__global__ void multiply_gpu(int size, float *d_data, float* d_output, float* to_multiply) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int nt = blockDim.x * gridDim.x;
    int i;
    for (i = 0; i * nt + tid < size; i++) {
        d_output[i * nt + tid] = d_data[i * nt + tid] * to_multiply[i * nt + tid];
    }
}

__global__ void multiply_gpu(int size, int D, float *d_data, float* d_output, float* to_multiply) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int nt = blockDim.x * gridDim.x;
    int i, j;
    for (i = 0; i * nt + tid < size; i++) {
        float* x = d_data + D * (i * nt + tid);
        float* out = d_output + D * (i * nt + tid);
        float k = to_multiply[i * nt + tid];
        for (j = 0; j < D; j++) {
            out[j] = x[j] * k;
        }
        //		d_output[i * nt + tid] = d_data[i * nt + tid] * to_multiply[i * nt
        //				+ tid];
    }
}

__global__ void exp_gpu(int size, float *d_data, float* d_output) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    int i;
    for (i = tid; i < size; i += tt) {
        d_output[i] = expf(d_data[i]);
    }
}

__global__ void multiply_matrix_gpu(int size, int D, float *d_data, float* d_output,
        float* to_multiply) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int nt = blockDim.x * gridDim.x;
    int i, j;

    SharedMemory<float> smem;
    float* sdata = smem.getPointer();
    const int tidib = threadIdx.x;
    float* myArray = sdata + tidib * D;

    //	for (int i = 0; i < D * D; i++) {
    //		printf("%f ", to_multiply[i]);
    //	}
    //	printf("\n");

    for (i = tid; i < size; i += nt) {
        float* x = d_vector_get(d_data, D, i);
        float* out = d_vector_get(d_output, D, i);
        d_matrix_times(to_multiply, x, myArray, D, D, D, 1);
        for (j = 0; j < D; j++) {
            out[j] = myArray[j];
        }
    }

    //	for (i = 0; i * nt + tid < size; i++) {
    //		float* x = d_data + D * (i * nt + tid);
    //		float* out = d_output + D * (i * nt + tid);
    //		d_matrix_times(to_multiply, x, out, D, D, D, 1);
    //		float k = to_multiply[i * nt + tid];
    //		for (j = 0; j < D; j++) {
    //			out[j] = x[j] * k;
    //		}
    //		d_output[i * nt + tid] = d_data[i * nt + tid] * to_multiply[i * nt
    //				+ tid];
    //	}
}

__global__ void set_gpu(int N, float* d_array, float value) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += tt) {
        d_array[i] = value;
    }
}

void add(int size, float* d_array, float* d_output, float to_add, int nb, int nt) {

    add_gpu<<<nb,nt>>>(size, d_array, d_output, to_add);

    cudaThreadSynchronize();
}

void add(int size, float* d_array, float* d_output, float* d_to_add, int nb, int nt) {

    add_gpu<<<nb,nt>>>(size, d_array, d_output, d_to_add);

    cudaThreadSynchronize();
}

void multiply(int size, float* d_array, float* d_output, float to_multiply, int nb, int nt) {
    multiply_gpu<<<nb,nt>>>(size, d_array, d_output, to_multiply);

    cudaThreadSynchronize();
}

void multiply(int size, float* d_array, float* d_output, float* d_to_multiply, int nb, int nt) {
    multiply_gpu<<<nb,nt>>>(size, d_array, d_output, d_to_multiply);

    cudaThreadSynchronize();

}

void multiply(int size, int D, float* d_array, float* d_output, float* d_to_multiply, int nb,
        int nt) {
    multiply_gpu<<<nb,nt, nt * D * D * sizeof(float)>>>(size, D, d_array, d_output, d_to_multiply);

    cudaThreadSynchronize();
}

void exp(int size, float* d_array, float* d_output, int nb, int nt) {
    exp_gpu<<<nb,nt>>>(size, d_array, d_output);

    cudaThreadSynchronize();
}

void multiply_matrix(int size, int D, float* d_array, float* d_output, float* d_to_multiply,
        int nb, int nt) {
    multiply_matrix_gpu<<<nb,nt,D * nt * sizeof(float)>>>(size, D, d_array, d_output, d_to_multiply);

    cudaThreadSynchronize();
}

void set(int size, float* d_array, float value, int nb, int nt) {
    set_gpu<<<nb,nt>>>(size, d_array, value);

    cudaThreadSynchronize();
}

