#include "xorshift.h"
#include "reduce.h"
#include "square.h"
#include "kiss.h"
#include <stdio.h>
#include "rng_shared.ch"
#include "func.h"

__device__ unsigned int* d_seeds_xorshift;

int nb_xorshift;

int nt_xorshift;

//__global__ void seedGPU(unsigned int* d_seeds) {
//	const int tt = blockDim.x * gridDim.x;
//	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	d_seeds[tid * 4] = tid;
//	d_seeds[tid * 4 + 1] = tt + tid;
//	d_seeds[tid * 4 + 2] = 2 * tt + tid;
//	d_seeds[tid * 4 + 3] = 3 * tt + tid;
//}

void killXS() {
    cudaFree(d_seeds_xorshift);
}

__global__ void burn(int N, unsigned int* d_seeds) {
    const int tt = blockDim.x * gridDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int x, y, z, w, tmp;

    x = d_seeds[tid * 4];
    y = d_seeds[tid * 4 + 1];
    z = d_seeds[tid * 4 + 2];
    w = d_seeds[tid * 4 + 3];

    int i;
    for (i = tid; i < N; i += tt) {
        tmp = (x ^ (x << 15));
        x = y;
        y = z;
        z = w;
        w = (w ^ (w >> 21)) ^ (tmp ^ (tmp >> 4));
    }

    d_seeds[tid * 4] = x;
    d_seeds[tid * 4 + 1] = y;
    d_seeds[tid * 4 + 2] = z;
    d_seeds[tid * 4 + 3] = w;
}

void seedXS(int n_burn, int nba, int nta) {
    nb_xorshift = nba;
    nt_xorshift = nta;
    int tt = nb_xorshift * nt_xorshift;
    cudaMalloc((void **) &d_seeds_xorshift, tt * 4 * sizeof(unsigned int));
    unsigned int* seeds = (unsigned int*) malloc(tt * 4 * sizeof(unsigned int));

    KISS_reset();
    KISS_burn(2048);

    for (int i = 0; i < tt * 4; i++) {
        seeds[i] = KISS();
    }

    cudaMemcpy(d_seeds_xorshift, seeds, tt * 4 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    free(seeds);

    burn<<<nb_xorshift, nt_xorshift>>>(n_burn, d_seeds_xorshift);
    cudaThreadSynchronize();
}

__global__ void randomUI_XS(unsigned int *d_random, int N, unsigned int* d_seeds) {
    const int tt = blockDim.x * gridDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int x, y, z, w, tmp;

    x = d_seeds[tid * 4];
    y = d_seeds[tid * 4 + 1];
    z = d_seeds[tid * 4 + 2];
    w = d_seeds[tid * 4 + 3];

    int i;
    for (i = tid; i < N; i += tt) {
        tmp = (x ^ (x << 15));
        x = y;
        y = z;
        z = w;
        w = (w ^ (w >> 21)) ^ (tmp ^ (tmp >> 4));

        d_random[i] = w;
    }

    d_seeds[tid * 4] = x;
    d_seeds[tid * 4 + 1] = y;
    d_seeds[tid * 4 + 2] = z;
    d_seeds[tid * 4 + 3] = w;

}

__global__ void randomIK_XS(int* d_random, int N, int k, unsigned int* d_seeds) {
    const int tt = blockDim.x * gridDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int x, y, z, w, tmp;

    x = d_seeds[tid * 4];
    y = d_seeds[tid * 4 + 1];
    z = d_seeds[tid * 4 + 2];
    w = d_seeds[tid * 4 + 3];

    int i;
    for (i = tid; i < N; i += tt) {
        tmp = (x ^ (x << 15));
        x = y;
        y = z;
        z = w;
        w = (w ^ (w >> 21)) ^ (tmp ^ (tmp >> 4));

        // same as w % k if k is a power of 2.
        //		d_random[i] = w & (k - 1);
        d_random[i] = w % k;
    }

    d_seeds[tid * 4] = x;
    d_seeds[tid * 4 + 1] = y;
    d_seeds[tid * 4 + 2] = z;
    d_seeds[tid * 4 + 3] = w;
}

__global__ void randomF_XS(float* d_random, int N, unsigned int* d_seeds) {
    const int tt = blockDim.x * gridDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int x, y, z, w, tmp;

    x = d_seeds[tid * 4];
    y = d_seeds[tid * 4 + 1];
    z = d_seeds[tid * 4 + 2];
    w = d_seeds[tid * 4 + 3];

    int i;
    for (i = tid; i < N; i += tt) {
        tmp = (x ^ (x << 15));
        x = y;
        y = z;
        z = w;
        w = (w ^ (w >> 21)) ^ (tmp ^ (tmp >> 4));

        d_random[i] = ((float) w) / 4294967295.0f;
    }

    d_seeds[tid * 4] = x;
    d_seeds[tid * 4 + 1] = y;
    d_seeds[tid * 4 + 2] = z;
    d_seeds[tid * 4 + 3] = w;

}

void populate_rand_XS(float* array, int N) {
    int tt = nb_xorshift * nt_xorshift;
    int M = max(N, tt);
    float *d_Rand;

    cudaMalloc((void **) &d_Rand, M * sizeof(float));

    cudaThreadSynchronize();

    randomF_XS<<<nb_xorshift, nt_xorshift>>>(d_Rand, M, d_seeds_xorshift);

    cudaThreadSynchronize();
    cudaMemcpy(array, d_Rand, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_Rand);
}

// populates a length N array of floats on host with N(0,1) numbers
void populate_randn_XS(float* array, int N) {
    int tt = nb_xorshift * nt_xorshift;
    int M = max(tt * 2, N);
    float *d_Rand;

    cudaMalloc((void **) &d_Rand, M * sizeof(float));

    randomF_XS<<<nb_xorshift, nt_xorshift>>>(d_Rand, M, d_seeds_xorshift);

    cudaThreadSynchronize();

    BoxMullerGPU<<<nb_xorshift, nt_xorshift>>>(d_Rand, M);

    cudaMemcpy(array, d_Rand, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Rand);
}

void populate_randIK_XS(int* h_array, int N, int k) {
    int tt = nb_xorshift * nt_xorshift;
    int M = max(N, tt);
    int *d_Rand;
    cudaMalloc((void **) &d_Rand, M * sizeof(int));

    randomIK_XS<<<nb_xorshift, nt_xorshift>>>(d_Rand, M, k, d_seeds_xorshift);

    cudaThreadSynchronize();
    cudaMemcpy(h_array, d_Rand, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_Rand);

}

void populate_randIK_XS_d(int* d_array, int N, int k) {
    int tt = nb_xorshift * nt_xorshift;
    if (N < tt) {
        int *d_Rand;
        cudaMalloc((void **) &d_Rand, tt * sizeof(int));

        randomIK_XS<<<nb_xorshift, nt_xorshift>>>(d_Rand, tt, k, d_seeds_xorshift);

        cudaMemcpy(d_array, d_Rand, N * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaFree(d_Rand);
    } else {
        randomIK_XS<<<nb_xorshift, nt_xorshift>>>(d_array, N, k, d_seeds_xorshift);
        cudaThreadSynchronize();
    }
}

void populate_rand_XS_d(float* d_array, int N) {
    int tt = nb_xorshift * nt_xorshift;
    if (N < tt) {
        float* d_Rand;
        cudaMalloc((void**) &d_Rand, tt * sizeof(float));

        randomF_XS<<<nb_xorshift, nt_xorshift>>>(d_Rand, tt, d_seeds_xorshift);

        cudaMemcpy(d_array, d_Rand, N * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(d_Rand);
    } else {
        randomF_XS<<<nb_xorshift, nt_xorshift>>>(d_array, N, d_seeds_xorshift);
        cudaThreadSynchronize();
    }
}

void populate_randn_XS_d(float* d_array, int N) {
    int tt = nb_xorshift * nt_xorshift;

    if (N < tt * 2) {

        float* temp;
        cudaMalloc((void**) &temp, tt * 2 * sizeof(float));

        randomF_XS<<<nb_xorshift, nt_xorshift>>>(temp, tt * 2, d_seeds_xorshift);

        cudaThreadSynchronize();

        BoxMullerGPU<<<nb_xorshift, nt_xorshift>>>(temp, tt * 2);

        cudaThreadSynchronize();

        cudaMemcpy(d_array, temp, N * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaFree(temp);

    } else {

        randomF_XS<<<nb_xorshift, nt_xorshift>>>(d_array, N, d_seeds_xorshift);

        cudaThreadSynchronize();

        BoxMullerGPU<<<nb_xorshift, nt_xorshift>>>(d_array, N);

        cudaThreadSynchronize();
    }
}

void populate_randUI_XS(unsigned int* array, int N) {
    int tt = nb_xorshift * nt_xorshift;
    int M = max(N, tt);
    unsigned int *d_Rand;

    cudaMalloc((void **) &d_Rand, M * sizeof(unsigned int));

    cudaThreadSynchronize();

    randomUI_XS<<<nb_xorshift, nt_xorshift>>>(d_Rand, M, d_seeds_xorshift);

    cudaThreadSynchronize();
    cudaMemcpy(array, d_Rand, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_Rand);
}

void populate_randUI_XS_d(unsigned int* d_array, int N) {
    int tt = nb_xorshift * nt_xorshift;
    if (N < tt) {
        unsigned int* d_Rand;
        cudaMalloc((void**) &d_Rand, tt * sizeof(unsigned int));

        randomUI_XS<<<nb_xorshift, nt_xorshift>>>(d_Rand, tt, d_seeds_xorshift);

        cudaMemcpy(d_array, d_Rand, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
        cudaFree(d_Rand);
    } else {
        randomUI_XS<<<nb_xorshift, nt_xorshift>>>(d_array, N, d_seeds_xorshift);
        cudaThreadSynchronize();
    }
}

//void compute_mean_and_variance(float* d_array, int N, float* mean, float* var) {
//	float h_sum;
//	float h_ss;
//
//	reduce(N, d_array, h_sum, nb, nt);
//
//	float* d_array2;
//	cudaMalloc((void **) &d_array2, N * sizeof(float));
//
//	square(N, d_array, d_array2, nb, nt);
//	reduce(N, d_array2, h_ss, nb, nt);
//	//	reduce_ss(N, d_array, &h_ss, nb, nt);
//
//	*mean = h_sum / N;
//	*var = h_ss / N - (*mean) * (*mean);
//
//	cudaFree(d_array2);
//
//}

void populate_rand_XS_REF_d(float* d_array, int N) {
    float* array = (float*) malloc(N * sizeof(float));
    populate_rand_XS_REF(array, N);
    cudaMemcpy(d_array, array, N * sizeof(float), cudaMemcpyHostToDevice);
    free(array);
}

void populate_randn_XS_REF_d(float* d_array, int N) {
    float* array = (float*) malloc(N * sizeof(float));
    populate_randn_XS_REF(array, N);
    cudaMemcpy(d_array, array, N * sizeof(float), cudaMemcpyHostToDevice);
    free(array);
}

void populate_randIK_XS_REF_d(int* d_array, int N, int k) {
    int* array = (int*) malloc(N * sizeof(int));
    populate_randIK_XS_REF(array, N, k);
    cudaMemcpy(d_array, array, N * sizeof(int), cudaMemcpyHostToDevice);
    free(array);
}

void populate_randUI_XS_REF_d(unsigned int* d_array, int N) {

    unsigned int* array = (unsigned int*) malloc(N * sizeof(unsigned int));
    populate_randUI_XS_REF(array, N);
    cudaMemcpy(d_array, array, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    free(array);
}
