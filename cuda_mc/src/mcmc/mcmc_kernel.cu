/*
 * mcmc_kernel.cu
 *
 *  Created on: 04-Feb-2009
 *      Author: alee
 */

#include "temper.ch"
#include "rng.h"
#include <stdio.h>
#include "test_functions.h"

__constant__ float args_p[NUM_AP];

//
__global__ void FUNC( metropolis_rw, TYPE)(
int size, float* d_array_init, float* d_array_step, float* d_array_uniform, float* d_array_out, int log) {
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int nt = blockDim.x * gridDim.x;
	int j;
	float w, x, y, ratio;

	x = d_array_init[tid];

	for (j = tid; j < size; j += nt) {
		w = d_array_step[j];
		y = x + w;
		// Metropolis so q(y,x) = q(x,y)
		if (log) {
			ratio = expf(LOG_TARGET(y, args_p) - LOG_TARGET(x, args_p));
		} else {
			ratio = TARGET(y, args_p) / TARGET(x, args_p);
		}
		if (d_array_uniform[j] < ratio) {
			x = y;
		}
		d_array_out[j] = x;
	}
}

void FUNC( metropolis_rw, TYPE)(
int N, float* d_array_init, float sigma, float* d_array_out, float* h_args_p, int log, int nb, int nt) {
	cudaMemcpyToSymbol(args_p, h_args_p, NUM_AP * sizeof(float));

	float* d_array_uniform;
	cudaMalloc((void **) &d_array_uniform, N * sizeof(float));
	populate_rand_d(d_array_uniform, N);

	float* d_array_step;
	cudaMalloc((void **) &d_array_step, N * sizeof(float));
	populate_randn_d(d_array_step, N);
	if (sigma != 1.0) {
		multiply(N, d_array_step, d_array_step, sigma, nb, nt);
	}

	FUNC(metropolis_rw, TYPE)<<<nb,nt>>>(N, d_array_init, d_array_step, d_array_uniform, d_array_out, log);

	cudaFree(d_array_uniform);
	cudaFree(d_array_step);
}

//__global__ void FUNC( metropolis_rw_step, TYPE)(
//float* d_array_init, float* d_array_step, float* d_array_uniform, float* d_array_out) {
//	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	float w, x, y, ratio;
//
//	x = d_array_init[tid];
//	w = d_array_step[tid];
//	y = x + w;
//	// Metropolis so q(y,x) = q(x,y)
//	ratio = TARGET(y, args_p) / TARGET(x, args_p);
//	if (d_array_uniform[tid] < ratio) {
//		d_array_out[tid] = y;
//	} else {
//		d_array_out[tid] = x;
//	}
//
//}
//
//void FUNC( metropolis_rw_steps, TYPE)(
//int N, float* d_array_init, float sigma, float* d_array_out, float* h_args_p, int nb, int nt) {
//	cudaMemcpyToSymbol(args_p, h_args_p, NUM_AP * sizeof(float));
//	int tt = nb * nt;
//	int numSteps = N / tt;
//
//	float* d_array_uniform;
//	cudaMalloc((void **) &d_array_uniform, N * sizeof(float));
//	populate_rand_d(d_array_uniform, N);
//
//	float* d_array_step;
//	cudaMalloc((void **) &d_array_step, N * sizeof(float));
//	populate_randn_d(d_array_step, N);
//	if (sigma != 1.0) {
//		multiply(N, d_array_step, d_array_step, sigma, nb, nt);
//	}
//
//	for (int i = 0; i < numSteps; i++) {
//		FUNC(metropolis_rw_step, TYPE)<<<nb,nt>>>(d_array_init, d_array_step, d_array_uniform, d_array_out);
//		cudaThreadSynchronize();
//		d_array_init = d_array_out;
//		d_array_step += tt;
//		d_array_uniform += tt;
//		d_array_out += tt;
//	}
//
//	cudaFree(d_array_uniform);
//	cudaFree(d_array_step);
//
//}

__global__ void FUNC( metropolis_rwpop_step, TYPE)(
float* d_array_init, float* d_array_step, float* d_array_uniform, float* d_temps, float* d_array_out, int log) {
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	float w, x, y, t, ratio;

	t = d_temps[tid];
	x = d_array_init[tid];
	w = d_array_step[tid];
	y = x + w;
	// Metropolis so q(y,x) = q(x,y)
	if (log) {
		ratio = expf(t * (LOG_TARGET(y, args_p) - LOG_TARGET(x, args_p)));
	} else {
		ratio = temper(TARGET(y, args_p), t) / temper(TARGET(x, args_p), t);
	}
	if (d_array_uniform[tid] < ratio) {
		d_array_out[tid] = y;
	} else {
		d_array_out[tid] = x;
	}
}

__global__ void FUNC( metropolis_rwpop_exchange, TYPE)(
float* d_array_values, int type, float* d_temps,
float* d_array_uniform, int log) {
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int tt = blockDim.x * gridDim.x;

	if (tid % 2 == type && tid != tt - 1) {

		int otid = tid + 1;

		float x = d_array_values[tid];
		float y = d_array_values[otid];

		float t = d_temps[tid];
		float t2 = d_temps[otid];

		float ratio;
		if (log) {
			float ty = LOG_TARGET(y, args_p);
			float tx = LOG_TARGET(x, args_p);
			ratio = expf(ty * (t - t2) + tx * (t2 - t));
		} else {
			float ty = TARGET(y, args_p);
			float tx = TARGET(x, args_p);
			ratio = temper(ty, t - 2) * temper(tx, t2 - t);
//			ratio = temper(TARGET(y, args_p), t)
//			/ temper(TARGET(y, args_p), t2)
//			* temper(TARGET(x, args_p), t2)
//			/ temper(TARGET(x, args_p), t);
		}

		if (d_array_uniform[tid] < ratio) {
			d_array_values[tid] = y;
			d_array_values[otid] = x;
		}
	}
}

void FUNC( metropolis_rwpop, TYPE)(
int N, float* d_array_init, float sigma, float* h_args_p, float* d_temps, float* d_array_out, int log, int nb, int nt) {

	cudaMemcpyToSymbol(args_p, h_args_p, NUM_AP * sizeof(float));
	int tt = nb * nt;
	int numSteps = N / tt;

	int* array_types = (int*) malloc(numSteps * sizeof(int));
	populate_randIK(array_types, numSteps, 2);

	float* d_array_step;
	cudaMalloc((void **) &d_array_step, N * sizeof(float));
	populate_randn_d(d_array_step, N);
	if (sigma != 1.0) {
		multiply(N, d_array_step, d_array_step, sigma, nb, nt);
	}

	float* d_array_uniform1;
	float* d_array_uniform2;
	cudaMalloc((void **) &d_array_uniform1, N * sizeof(float));
	cudaMalloc((void **) &d_array_uniform2, N * sizeof(float));
	populate_rand_d(d_array_uniform1, N);
	populate_rand_d(d_array_uniform2, N);

	float* du1_orig = d_array_uniform1;
	float* du2_orig = d_array_uniform2;
	float* ds_orig = d_array_step;

	for (int i = 0; i < numSteps; i++) {

		FUNC(metropolis_rwpop_step, TYPE)<<<nb,nt>>>(d_array_init, d_array_step + i*tt, d_array_uniform1 + i*tt,
		d_temps, d_array_out + i*tt, log);
		cudaThreadSynchronize();

		FUNC(metropolis_rwpop_exchange, TYPE)<<<nb,nt>>>(d_array_out + i*tt, array_types[i], d_temps,
		d_array_uniform2 + i * tt, log);
		cudaThreadSynchronize();

		cudaMemcpy(d_array_init, d_array_out + i * tt, tt * sizeof(float), cudaMemcpyDeviceToDevice);

	}

	cudaFree(du1_orig);
	cudaFree(du2_orig);
	cudaFree(ds_orig);

	free(array_types);

}
