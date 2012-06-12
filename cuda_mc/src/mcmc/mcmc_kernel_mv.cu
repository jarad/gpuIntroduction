/*
 * mcmc_kernel_mv.cu
 *
 *  Created on: 24-Feb-2009
 *      Author: alee
 */

#include "temper.ch"
#include "matrix.ch"
#include "matrix.h"
#include <stdio.h>
#include "sharedmem.cuh"
#include "test_functions.h"
#include "rng.h"

__constant__ float args_p[NUM_AP];

template<int D>
__global__ void FUNC( metropolis_rw_gpu, TYPE)(int N, float* d_array_init, float* d_array_step,
        float* d_array_uniform, float* d_array_out, int log) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int nt = blockDim.x * gridDim.x;
    int j;
    float* x;
    float* w;
    float ratio;
    SharedMemory<float> smem;
    float* sdata = smem.getPointer();
    float* y = sdata + D * threadIdx.x;

    x = d_vector_get(d_array_init, D, tid);

    for (j = tid; j < N; j += nt) {
        w = d_vector_get(d_array_step, D, j);
        d_vector_add(x, w, y, D);
        // Metropolis so q(y,x) = q(x,y)
        if (log == 0) {
            ratio = TARGET<D> (y, args_p) / TARGET<D> (x, args_p);
        } else {
            ratio = expf(LOG_TARGET<D> (y, args_p) - LOG_TARGET<D> (x, args_p));
        }
        if (d_array_uniform[j] < ratio) {
            d_vector_set(x, y, D);
        }
        d_vector_set(d_vector_get(d_array_out, D, j), x, D);
    }
}

template <int D>
void FUNC( metropolis_rw, TYPE)(int N, float* d_array_init, float sigma, float* d_array_out,
        float* h_args_p, int log, int nb, int nt) {
    cudaMemcpyToSymbol(args_p, h_args_p, NUM_AP * sizeof(float));

    float* d_array_uniform;
    cudaMalloc((void **) &d_array_uniform, N * sizeof(float));
    populate_rand_d(d_array_uniform, N);

    float* d_array_step;
    cudaMalloc((void **) &d_array_step, N * D * sizeof(float));
    populate_randn_d(d_array_step, N * D);
    if (sigma != 1.0) {
        multiply(N * D, d_array_step, d_array_step, sigma, nb, nt);
    }

    FUNC(metropolis_rw_gpu, TYPE) < D> <<<nb,nt,D*nt*sizeof(float)>>>(N, d_array_init, d_array_step, d_array_uniform, d_array_out, log);

    cudaFree(d_array_uniform);
    cudaFree(d_array_step);
}

template <int D>
__global__ void FUNC( metropolis_rwpop_step, TYPE)(float* d_array_init, float* d_array_step,
        float* d_array_uniform, float* d_temps, float* d_array_out, int log) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float* w;
    float* x;
    float t, ratio;
    SharedMemory<float> smem;
    float* sdata = smem.getPointer();
    float* y = sdata + D * threadIdx.x;

    t = d_temps[tid];
    x = d_vector_get(d_array_init, D, tid);
    w = d_vector_get(d_array_step, D, tid);
    d_vector_add(x, w, y, D);

    // Metropolis so q(y,x) = q(x,y)
    if (log == 0) {
        ratio = temper(TARGET<D> (y, args_p), t) / temper(TARGET<D> (x, args_p), t);
    } else {
        ratio = expf(LOG_TARGET<D> (y, args_p) * t - LOG_TARGET<D> (x, args_p) * t);
    }
    if (d_array_uniform[tid] < ratio) {
        d_vector_set(d_vector_get(d_array_out, D, tid), y, D);
    } else {
        d_vector_set(d_vector_get(d_array_out, D, tid), x, D);
    }
}

template <int D>
__global__ void FUNC( metropolis_rwpop_step2, TYPE)(float* d_array_init, float* d_array_step,
        float* d_array_uniform, float* d_temps, float* d_array_out, int log, float* densities) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float* w;
    float* x;
    float t, ratio;
    SharedMemory<float> smem;
    float* sdata = smem.getPointer();
    float* y = sdata + D * threadIdx.x;

    t = d_temps[tid];
    x = d_vector_get(d_array_init, D, tid);
    w = d_vector_get(d_array_step, D, tid);
    d_vector_add(x, w, y, D);

    // Metropolis so q(y,x) = q(x,y)
    float nv;

    if (log == 0) {
        nv = TARGET<D> (y, args_p);
        ratio = temper(nv, t) / temper(densities[tid], t);
    } else {
        nv = LOG_TARGET<D> (y, args_p);
        ratio = expf(nv * t - densities[tid] * t);
    }
    if (d_array_uniform[tid] < ratio) {
        densities[tid] = nv;
        d_vector_set(d_vector_get(d_array_out, D, tid), y, D);
    } else {
        d_vector_set(d_vector_get(d_array_out, D, tid), x, D);
    }
}

template <int D>
__global__ void FUNC( metropolis_rwpop_init, TYPE)(float* d_array_init, int log, float* densities) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float* x = d_vector_get(d_array_init, D, tid);
    if (log == 0) {
        densities[tid] = TARGET<D> (x, args_p);
    } else {
        densities[tid] =  LOG_TARGET<D> (x, args_p);
    }
}

template<int D>
__global__ void FUNC( metropolis_rwpop_exchange, TYPE)(float* d_array_values, int type, float* d_temps,
        float* d_array_uniform, int log) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;

    //	if ((type == 1 && tid % 2 == 0) || (type == 0 && tid % 2 == 1)) {
    if (tid % 2 == type) {

        int otid = (tid + 1) % tt;

        float* x = d_vector_get(d_array_values, D, tid);
        float* y = d_vector_get(d_array_values, D, otid);

        float t = d_temps[tid];
        float t2 = d_temps[otid];

        float ratio;
        if (log) {
            float ty = LOG_TARGET<D> (y, args_p);
            float tx = LOG_TARGET<D> (x, args_p);
            ratio = expf(ty * (t - t2) + tx * (t2 - t));
        } else {
            ratio = temper(TARGET<D> (y, args_p), t) / temper(TARGET<D> (y, args_p), t2) * temper(
                                TARGET<D> (x, args_p), t2) / temper(TARGET<D> (x, args_p), t);
        }

        if (d_array_uniform[tid] < ratio) {
            d_vector_swap(x, y, D);
        }
    }
}

template<int D>
__global__ void FUNC( metropolis_rwpop_exchange2, TYPE)(float* d_array_values, int type, float* d_temps,
        float* d_array_uniform, int log, float* densities) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tt = blockDim.x * gridDim.x;

    //  if ((type == 1 && tid % 2 == 0) || (type == 0 && tid % 2 == 1)) {
    if (tid % 2 == type) {

        int otid = (tid + 1) % tt;

        float* x = d_vector_get(d_array_values, D, tid);
        float* y = d_vector_get(d_array_values, D, otid);

        float t = d_temps[tid];
        float t2 = d_temps[otid];

        float ratio;
        float ty = densities[otid];
        float tx = densities[tid];
        if (log) {
            ratio = expf(ty * (t - t2) + tx * (t2 - t));
        } else {
            ratio = temper(ty, t - t2) * temper(tx, t2 - t);
        }

        if (d_array_uniform[tid] < ratio) {
            densities[tid] = ty;
            densities[otid] = tx;
            d_vector_swap(x, y, D);
        }
    }
}

template<int D>
void FUNC( metropolis_rwpop, TYPE)(int N, float* d_array_init, float sigma, float* h_args_p,
        float* d_temps, float* d_array_out, int log, int nb, int nt) {
    cudaMemcpyToSymbol(args_p, h_args_p, NUM_AP * sizeof(float));

    int tt = nb * nt;
    int numSteps = N / tt;

    int* array_types = (int*) malloc(numSteps * sizeof(int));

    populate_randIK(array_types, numSteps, 2);

    float* d_array_step;
    cudaMalloc((void **) &d_array_step, N * D * sizeof(float));

    populate_randn_d(d_array_step, N * D);

    if (sigma != 1.0) {
        multiply(N * D, d_array_step, d_array_step, sigma, nb, nt);
    }

    float* d_array_uniform1;
    float* d_array_uniform2;
    cudaMalloc((void **) &d_array_uniform1, N * sizeof(float));
    cudaMalloc((void **) &d_array_uniform2, N * sizeof(float));
    populate_rand_d(d_array_uniform1, N);
    populate_rand_d(d_array_uniform2, N);

    float* du1 = d_array_uniform1;
    float* du2 = d_array_uniform2;
    float* ds = d_array_step;

    for (int i = 0; i < numSteps; i++) {

        //		printf("on step %d\n", i);

        FUNC(metropolis_rwpop_step, TYPE)<D><<<nb,nt,D*nt*sizeof(float)>>>(d_array_init, ds, du1, d_temps, d_array_out, log);
        cudaThreadSynchronize();

        FUNC(metropolis_rwpop_exchange, TYPE)<D><<<nb,nt>>>(d_array_out, array_types[i], d_temps, du2, log);
        cudaThreadSynchronize();

        d_array_init = d_array_out;
        ds += tt * D;
        du1 += tt;
        d_array_out += tt * D;
        du2 += tt;

    }

    cudaFree(d_array_uniform1);
    cudaFree(d_array_uniform2);
    cudaFree(d_array_step);

    free(array_types);

}

template<int D>
void FUNC( metropolis_rwpop_marginal, TYPE)(int N, float* d_array_init, float sigma, float* h_args_p,
        float* d_temps, float* d_array_out, int log, int nb, int nt) {
    cudaMemcpyToSymbol(args_p, h_args_p, NUM_AP * sizeof(float));
    int tt = nb * nt;
    int numSteps = N / tt;

    int* array_types = (int*) malloc(numSteps * sizeof(int));

    populate_randIK(array_types, numSteps, 2);

    float* d_array_step;
    cudaMalloc((void **) &d_array_step, N * D * sizeof(float));
    populate_randn_d(d_array_step, N * D);
    if (sigma != 1.0) {
        multiply(N * D, d_array_step, d_array_step, sigma, nb, nt);
    }
    float* d_array_uniform1;
    float* d_array_uniform2;
    cudaMalloc((void **) &d_array_uniform1, N * sizeof(float));
    cudaMalloc((void **) &d_array_uniform2, N * sizeof(float));
    populate_rand_d(d_array_uniform1, N);
    populate_rand_d(d_array_uniform2, N);

    float* du1 = d_array_uniform1;
    float* du2 = d_array_uniform2;
    float* ds = d_array_step;

    float* d_array_temp;
    cudaMalloc((void**) &d_array_temp, tt * D * sizeof(float));

    float* densities;
    cudaMalloc((void**) &densities, tt * sizeof(float));

    FUNC( metropolis_rwpop_init, TYPE)<D><<<nb,nt>>>(d_array_init, log, densities);
    cudaThreadSynchronize();

    for (int i = 0; i < numSteps; i++) {
        //		printf("Time %d:\n", i);

        FUNC(metropolis_rwpop_step2, TYPE)<D><<<nb,nt,D*nt*sizeof(float)>>>(d_array_init, ds + i * tt * D, du1 + i * tt, d_temps, d_array_temp, log, densities);
        cudaThreadSynchronize();

        FUNC(metropolis_rwpop_exchange2, TYPE)<D><<<nb,nt>>>(d_array_temp, array_types[i], d_temps,
                du2 + i * tt, log, densities);

        cudaMemcpy(d_array_init, d_array_temp, tt * D * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaMemcpy(vector_get(d_array_out, D, i), vector_get(d_array_temp, D, tt - 1), D
                * sizeof(float), cudaMemcpyDeviceToDevice);

    }

    cudaFree(d_array_uniform1);
    cudaFree(d_array_uniform2);
    cudaFree(d_array_step);
    cudaFree(d_array_temp);
    cudaFree(densities);

    free(array_types);

}
