/*
 * MRG2.cu
 *
 *  Created on: 24-Mar-2009
 *      Author: Owner
 */

#include "matrix.h"
#include "rng_shared.ch"

int nb_MRG;

int nt_MRG;

#define a_MRG 1403580UL
#define b_MRG 810728UL
#define c_MRG 527612UL
#define d_MRG 1370589UL

#define m1_MRG 4294967087UL
#define m2_MRG 4294944443UL

__device__ unsigned long* d_seeds_MRG;

void seed_MRG(int nba, int nta, unsigned long* seeds) {

    nb_MRG = nba;
    nt_MRG = nta;
    int tt = nb_MRG * nt_MRG;
    int logtt = (int) log2f((float) tt);
    int k = 190 - logtt; // period is ~ 2^191

    unsigned long mbmodm1 = m1_MRG - b_MRG;
    unsigned long mdmodm2 = m2_MRG - d_MRG;

    unsigned long E[9] = { 0, a_MRG, mbmodm1, 1, 0, 0, 0, 1, 0 };
    unsigned long F[9] = { c_MRG, 0, mdmodm2, 1, 0, 0, 0, 1, 0 };

    unsigned long temp[9];

    unsigned long E_k[9];
    unsigned long F_k[9];

    matrix_copy(E, E_k, 3, 3);
    matrix_copy(F, F_k, 3, 3);

    for (int i = 0; i < k; i++) {
        matrix_times_mod(E_k, E_k, temp, 3, 3, 3, 3, m1_MRG);
        matrix_copy(temp, E_k, 3, 3);

        matrix_times_mod(F_k, F_k, temp, 3, 3, 3, 3, m2_MRG);
        matrix_copy(temp, F_k, 3, 3);
    }

    cudaMalloc((void**) &d_seeds_MRG, tt * 6 * sizeof(unsigned long));

    unsigned long* hd_seeds = (unsigned long*) malloc(tt * 6 * sizeof(unsigned long));

    unsigned long y1[3] = { seeds[0], seeds[1], seeds[2] };
    unsigned long y2[3] = { seeds[3], seeds[4], seeds[5] };

    unsigned long y1_n[3];
    unsigned long y2_n[3];

    hd_seeds[0] = y1[0];
    hd_seeds[1] = y1[1];
    hd_seeds[2] = y1[2];
    hd_seeds[3] = y2[0];
    hd_seeds[4] = y2[1];
    hd_seeds[5] = y2[2];

    for (int i = 1; i < tt; i++) {
        matrix_times_mod(E_k, y1, y1_n, 3, 3, 3, 1, m1_MRG);
        matrix_times_mod(F_k, y2, y2_n, 3, 3, 3, 1, m2_MRG);

        y1[0] = y1_n[0];
        y1[1] = y1_n[1];
        y1[2] = y1_n[2];

        y2[0] = y2_n[0];
        y2[1] = y2_n[1];
        y2[2] = y2_n[2];

        hd_seeds[i * 6] = y1[0];
        hd_seeds[i * 6 + 1] = y1[1];
        hd_seeds[i * 6 + 2] = y1[2];
        hd_seeds[i * 6 + 3] = y2[0];
        hd_seeds[i * 6 + 4] = y2[1];
        hd_seeds[i * 6 + 5] = y2[2];
    }

    cudaMemcpy(d_seeds_MRG, hd_seeds, tt * 6 * sizeof(unsigned long), cudaMemcpyHostToDevice);

    //    for (int i = 0; i < 6; i++) {
    //        h_seeds_MRG[i] = hd_seeds[tt * 6 - 6 + i];
    //    }

    free(hd_seeds);

}

void kill_MRG() {
    cudaFree(d_seeds_MRG);
}

__device__ unsigned long mymod(unsigned long x, unsigned long m) {
    if (x > m) {
        return x % m;
    } else {
        return x;
    }
}

__global__ void randomUI_MRG(unsigned int* d_array, int N, unsigned long* d_seeds) {
    const int tt = blockDim.x * gridDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned long y1[3] = { d_seeds[tid * 6], d_seeds[tid * 6 + 1], d_seeds[tid * 6 + 2] };
    unsigned long y2[3] = { d_seeds[tid * 6 + 3], d_seeds[tid * 6 + 4], d_seeds[tid * 6 + 5] };

    unsigned long y1_n;
    unsigned long y2_n;
    unsigned long z_n;
    unsigned long t1;
    unsigned long t2;

    for (int i = tid; i < N; i += tt) {
        t1 = (a_MRG * y1[1]) % m1_MRG;
        t2 = mymod(b_MRG * y1[2], m1_MRG);
        if (t2 < t1) {
            y1_n = t1 - t2;
        } else {
            y1_n = t1 + m1_MRG - t2;
        }

        t1 = (c_MRG * y2[0]) % m2_MRG;
        t2 = mymod(d_MRG * y2[2], m2_MRG);
        if (t2 < t1) {
            y2_n = t1 - t2;
        } else {
            y2_n = t1 + m2_MRG - t2;
        }

        y1[2] = y1[1];
        y1[1] = y1[0];
        y1[0] = y1_n;

        y2[2] = y2[1];
        y2[1] = y2[0];
        y2[0] = y2_n;

        if (y1_n > y2_n) {
            z_n = y1_n - y2_n;
        } else {
            z_n = y1_n + m1_MRG - y2_n;
        }
        if (z_n > 0) {
            d_array[i] = z_n;
        } else {
            d_array[i] = m1_MRG;
        }
    }

    d_seeds[tid * 6] = y1[0];
    d_seeds[tid * 6 + 1] = y1[1];
    d_seeds[tid * 6 + 2] = y1[2];
    d_seeds[tid * 6 + 3] = y2[0];
    d_seeds[tid * 6 + 4] = y2[1];
    d_seeds[tid * 6 + 5] = y2[2];

}

__global__ void randomF_MRG(float* d_array, int N, unsigned long* d_seeds) {
    const int tt = blockDim.x * gridDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned long y1[3] = { d_seeds[tid * 6], d_seeds[tid * 6 + 1], d_seeds[tid * 6 + 2] };
    unsigned long y2[3] = { d_seeds[tid * 6 + 3], d_seeds[tid * 6 + 4], d_seeds[tid * 6 + 5] };

    unsigned long y1_n;
    unsigned long y2_n;
    unsigned long z_n;
    unsigned long t1;
    unsigned long t2;

    for (int i = tid; i < N; i += tt) {
        t1 = (a_MRG * y1[1]) % m1_MRG;
        t2 = mymod(b_MRG * y1[2], m1_MRG);
        if (t2 < t1) {
            y1_n = t1 - t2;
        } else {
            y1_n = t1 + m1_MRG - t2;
        }

        t1 = (c_MRG * y2[0]) % m2_MRG;
        t2 = mymod(d_MRG * y2[2], m2_MRG);
        if (t2 < t1) {
            y2_n = t1 - t2;
        } else {
            y2_n = t1 + m2_MRG - t2;
        }

        y1[2] = y1[1];
        y1[1] = y1[0];
        y1[0] = y1_n;

        y2[2] = y2[1];
        y2[1] = y2[0];
        y2[0] = y2_n;

        if (y1_n > y2_n) {
            z_n = y1_n - y2_n;
        } else {
            z_n = y1_n + m1_MRG - y2_n;
        }

        if (z_n > 0) {
            d_array[i] = ((float) z_n) / (m1_MRG + 1);
        } else {
            d_array[i] = ((float) m1_MRG) / (m1_MRG + 1);
        }
    }

    d_seeds[tid * 6] = y1[0];
    d_seeds[tid * 6 + 1] = y1[1];
    d_seeds[tid * 6 + 2] = y1[2];
    d_seeds[tid * 6 + 3] = y2[0];
    d_seeds[tid * 6 + 4] = y2[1];
    d_seeds[tid * 6 + 5] = y2[2];

}

__global__ void randomIK_MRG(int* d_array, int N, unsigned int k, unsigned long* d_seeds) {
    const int tt = blockDim.x * gridDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned long y1[3] = { d_seeds[tid * 6], d_seeds[tid * 6 + 1], d_seeds[tid * 6 + 2] };
    unsigned long y2[3] = { d_seeds[tid * 6 + 3], d_seeds[tid * 6 + 4], d_seeds[tid * 6 + 5] };

    unsigned long y1_n;
    unsigned long y2_n;
    unsigned long z_n;
    unsigned long t1;
    unsigned long t2;

    //    unsigned long mbmodm1 = m1_MRG - b_MRG;
    //    unsigned long mdmodm2 = m2_MRG - d_MRG;

    for (int i = tid; i < N; i += tt) {
        t1 = (a_MRG * y1[1]) % m1_MRG;
        t2 = mymod(b_MRG * y1[2], m1_MRG);
        if (t2 < t1) {
            y1_n = t1 - t2;
        } else {
            y1_n = t1 + m1_MRG - t2;
        }

        t1 = (c_MRG * y2[0]) % m2_MRG;
        t2 = mymod(d_MRG * y2[2], m2_MRG);
        if (t2 < t1) {
            y2_n = t1 - t2;
        } else {
            y2_n = t1 + m2_MRG - t2;
        }

        y1[2] = y1[1];
        y1[1] = y1[0];
        y1[0] = y1_n;

        y2[2] = y2[1];
        y2[1] = y2[0];
        y2[0] = y2_n;

        if (y1_n > y2_n) {
            z_n = y1_n - y2_n;
        } else {
            z_n = y1_n + m1_MRG - y2_n;
        }

        //        z_n = (y1_n + m1_MRG - y2_n) % m1_MRG;
        d_array[i] = z_n % k;

    }

    d_seeds[tid * 6] = y1[0];
    d_seeds[tid * 6 + 1] = y1[1];
    d_seeds[tid * 6 + 2] = y1[2];
    d_seeds[tid * 6 + 3] = y2[0];
    d_seeds[tid * 6 + 4] = y2[1];
    d_seeds[tid * 6 + 5] = y2[2];

}

void populate_rand_MRG(float* array, int N) {
    int tt = nb_MRG * nt_MRG;
    int M = max(N, tt);
    float *d_Rand;

    cudaMalloc((void **) &d_Rand, M * sizeof(float));

    cudaThreadSynchronize();

    randomF_MRG<<<nb_MRG, nt_MRG>>>(d_Rand, M, d_seeds_MRG);

    cudaThreadSynchronize();
    cudaMemcpy(array, d_Rand, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_Rand);
}

void populate_rand_MRG_d(float* d_array, int N) {
    int tt = nb_MRG * nt_MRG;
    if (N < tt) {
        float* d_Rand;
        cudaMalloc((void**) &d_Rand, tt * sizeof(float));

        randomF_MRG<<<nb_MRG, nt_MRG>>>(d_Rand, tt, d_seeds_MRG);

        cudaMemcpy(d_array, d_Rand, N * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(d_Rand);
    } else {
        randomF_MRG<<<nb_MRG, nt_MRG>>>(d_array, N, d_seeds_MRG);
        cudaThreadSynchronize();
    }
}

// populates a length N array of floats on host with N(0,1) numbers
void populate_randn_MRG(float* array, int N) {
    int tt = nb_MRG * nt_MRG;
    int M = max(tt * 2, N);
    float *d_Rand;

    cudaMalloc((void **) &d_Rand, M * sizeof(float));

    randomF_MRG<<<nb_MRG, nt_MRG>>>(d_Rand, M, d_seeds_MRG);

    cudaThreadSynchronize();

    BoxMullerGPU<<<nb_MRG, nt_MRG>>>(d_Rand, M);

    cudaMemcpy(array, d_Rand, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_Rand);
}

void populate_randIK_MRG(int* h_array, int N, int k) {
    int tt = nb_MRG * nt_MRG;
    int M = max(N, tt);
    int *d_Rand;
    cudaMalloc((void **) &d_Rand, M * sizeof(int));

    randomIK_MRG<<<nb_MRG, nt_MRG>>>(d_Rand, M, k, d_seeds_MRG);

    cudaThreadSynchronize();
    cudaMemcpy(h_array, d_Rand, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_Rand);

}

void populate_randIK_MRG_d(int* d_array, int N, int k) {
    int tt = nb_MRG * nt_MRG;
    if (N < tt) {
        int *d_Rand;
        cudaMalloc((void **) &d_Rand, tt * sizeof(int));

        randomIK_MRG<<<nb_MRG, nt_MRG>>>(d_Rand, tt, k, d_seeds_MRG);

        cudaMemcpy(d_array, d_Rand, N * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaFree(d_Rand);
    } else {
        randomIK_MRG<<<nb_MRG, nt_MRG>>>(d_array, N, k, d_seeds_MRG);
        cudaThreadSynchronize();
    }
}

void populate_randn_MRG_d(float* d_array, int N) {
    int tt = nb_MRG * nt_MRG;

    if (N < tt * 2) {

        float* temp;
        cudaMalloc((void**) &temp, tt * 2 * sizeof(float));

        randomF_MRG<<<nb_MRG, nt_MRG>>>(temp, tt * 2, d_seeds_MRG);

        cudaThreadSynchronize();

        BoxMullerGPU<<<nb_MRG, nt_MRG>>>(temp, tt * 2);

        cudaThreadSynchronize();

        cudaMemcpy(d_array, temp, N * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaFree(temp);

    } else {

        randomF_MRG<<<nb_MRG, nt_MRG>>>(d_array, N, d_seeds_MRG);

        cudaThreadSynchronize();

        BoxMullerGPU<<<nb_MRG, nt_MRG>>>(d_array, N);

        cudaThreadSynchronize();
    }
}

void populate_randUI_MRG(unsigned int* array, int N) {
    int tt = nb_MRG * nt_MRG;
    int M = max(N, tt);
    unsigned int *d_Rand;

    cudaMalloc((void **) &d_Rand, M * sizeof(unsigned int));

    cudaThreadSynchronize();

    randomUI_MRG<<<nb_MRG, nt_MRG>>>(d_Rand, M, d_seeds_MRG);

    cudaThreadSynchronize();
    cudaMemcpy(array, d_Rand, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_Rand);
}

void populate_randUI_MRG_d(unsigned int* d_array, int N) {
    int tt = nb_MRG * nt_MRG;
    if (N < tt) {
        unsigned int* d_Rand;
        cudaMalloc((void**) &d_Rand, tt * sizeof(unsigned int));

        randomUI_MRG<<<nb_MRG, nt_MRG>>>(d_Rand, tt, d_seeds_MRG);

        cudaMemcpy(d_array, d_Rand, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
        cudaFree(d_Rand);
    } else {
        randomUI_MRG<<<nb_MRG, nt_MRG>>>(d_array, N, d_seeds_MRG);
        cudaThreadSynchronize();
    }
}
