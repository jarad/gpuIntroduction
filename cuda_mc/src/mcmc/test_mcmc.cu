/*
 * test_mcmc.cu
 *
 *  Created on: 04-Feb-2009
 *      Author: alee
 */

#include <stdio.h>
#include <cuda.h>
#include <cutil.h>
#include "rng.h"
#include "reduce.h"
#include "mcmc_gauss.h"
#include "mcmc_gauss_mv.h"
#include "mix_gauss.h"
#include "gauss.h"
#include "mcmc_mix_gauss.h"
#include "mcmc_mix_gauss_mu.h"
#include "output.h"
#include "matrix.h"
#include "test_functions.h"

void tmm_pop(long N, int k, float* d_array_init, float* temps, float* h_args_p,
		int nb, int nt) {

	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	float* h_sum = (float*) malloc(k * sizeof(float));
	float* result = (float*) malloc(k * sizeof(float));

	int numChains = nb * nt;
	int M = N / numChains;

	const int dataLimit = 134217728;
	int numIterations = 1;
	int iterationSize = N;
	long dataNeeded = ((long) N) * k * sizeof(float);
	if (dataNeeded > dataLimit) {
		numIterations = dataNeeded / dataLimit;
		iterationSize = N / numIterations;
	}

	int marginalIncrement = M / numIterations * k;

	float* d_array_out;
	cudaMalloc((void **) &d_array_out, M * k * sizeof(float));

	float* array_out = (float*) malloc(M * k * sizeof(float));

	float* d_temps;
	cudaMalloc((void **) &d_temps, numChains * sizeof(float));
	cudaMemcpy(d_temps, temps, numChains * sizeof(float),
			cudaMemcpyHostToDevice);

	kill_rng();
	seed_rng(32768);

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);

	for (int i = 0; i < numIterations; i++) {
		metropolis_rwpop_marginal_mgumu_mv(iterationSize, k, d_array_init, 1.0,
				h_args_p, d_temps, d_array_out + i * marginalIncrement, 1, nb,
				nt);
	}

	cudaThreadSynchronize();

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	cudaMemcpy(array_out, d_array_out, M * k * sizeof(float),
			cudaMemcpyDeviceToHost);

	for (int i = 0; i < k; i++) {
		h_sum[i] = 0;
	}
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < k; j++) {
			h_sum[j] += vector_get(array_out, k, i)[j];
		}
	}

	for (int j = 0; j < k; j++) {
		result[j] = h_sum[j] / M;
	}

	printf("POP RESULT = (%f,%f,%f,%f)\n", result[0], result[1], result[2],
			result[3]);
	char* fname = "real_pop_00000.txt";
	if (numChains == 1) {
		fname = "real_pop_00001.txt";
	} else if (numChains == 2) {
		fname = "real_pop_00002.txt";
	} else if (numChains == 4) {
		fname = "real_pop_00004.txt";
	} else if (numChains == 8) {
		fname = "real_pop_00008.txt";
	} else if (numChains == 32) {
		fname = "real_pop_00032.txt";
	} else if (numChains == 128) {
		fname = "real_pop_00128.txt";
	} else if (numChains == 512) {
		fname = "real_pop_00512.txt";
	} else if (numChains == 2048) {
		fname = "real_pop_02048.txt";
	} else if (numChains == 8192) {
		fname = "real_pop_08192.txt";
	} else if (numChains == 32768) {
		fname = "real_pop_32768.txt";
	}
	to_file(array_out, M, k, fname);

	free(array_out);
	cudaFree(d_array_out);
	cudaFree(d_temps);

	free(h_sum);
	free(result);

}

void tmm_pop_ref(int N, int k, float* array_init, float* temps,
		float* h_args_p, int numChains) {
	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	float* h_sum = (float*) malloc(k * sizeof(float));
	float* result = (float*) malloc(k * sizeof(float));

	int M = N / numChains;

	const int dataLimit = 134217728;
	int numIterations = 1;
	int iterationSize = N;
	long dataNeeded = ((long) N) * k * sizeof(float);
	if (dataNeeded > dataLimit) {
		numIterations = dataNeeded / dataLimit;
		iterationSize = N / numIterations;
	}

	int marginalIncrement = M / numIterations * k;

	float* array_out = (float*) malloc(M * k * sizeof(float));

	kill_rng();
	seed_rng(32768);

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);

	for (int i = 0; i < numIterations; i++) {
		metropolis_rwpop_marginal_ref_mgumu_mv(iterationSize, k, array_init,
				1.0, h_args_p, temps, array_out + i * marginalIncrement, 1,
				numChains);
	}

	cudaThreadSynchronize();

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	for (int i = 0; i < k; i++) {
		h_sum[i] = 0;
	}
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < k; j++) {
			h_sum[j] += vector_get(array_out, k, i)[j];
		}
	}

	for (int j = 0; j < k; j++) {
		result[j] = h_sum[j] / M;
	}

	printf("POP REF RESULT = (%f,%f,%f,%f)\n", result[0], result[1], result[2],
			result[3]);

	to_file(array_out, M, k, "ref_pop.txt");

	free(array_out);

	free(h_sum);
	free(result);
}

void test_mixture_mus(int M, int nb, int nt) {
	const int k = 4;
	long numChains = nb * nt;

	long N = numChains * M;

	printf("No.Chains = %ld\n", numChains);

	const int L = 100;
	float sigma = 0.55f;
	float mus[k];
	mus[0] = -3;
	mus[1] = 0;
	mus[2] = 3;
	mus[3] = 6;

	kill_rng();
	seed_rng(32768, 32, 128);

	float data_array[L];
	generate_mix_data(k, sigma, mus, data_array, L);

	float c1, c2;
	compute_ci1_ci2(sigma, 1.0f / k, c1, c2);

	float h_args_p[L + 5];
	h_args_p[0] = L;

	for (int i = 0; i < L; i++) {
		h_args_p[i + 1] = data_array[i];
	}

	h_args_p[L + 1] = c1;
	h_args_p[L + 2] = c2;
	h_args_p[L + 3] = -10;
	h_args_p[L + 4] = 10;

	float* array_init = (float*) malloc(numChains * k * sizeof(float));

	float* d_array_init;

	cudaMalloc((void **) &d_array_init, numChains * k * sizeof(float));

	populate_rand_d(d_array_init, numChains * k);
	cudaThreadSynchronize();
	multiply(numChains * k, d_array_init, d_array_init, 20, nb, nt);
	cudaThreadSynchronize();
	add(numChains * k, d_array_init, d_array_init, -10, nb, nt);
	cudaThreadSynchronize();
	cudaMemcpy(array_init, d_array_init, numChains * k * sizeof(float),
			cudaMemcpyDeviceToHost);

	float* temps = (float*) malloc(numChains * sizeof(float));
	for (int i = 0; i < numChains; i++) {
		float j = ((float) i + 1);
		temps[i] = j * j / (numChains * numChains);
	}

	tmm_pop(N, k, d_array_init, temps, h_args_p, nb, nt);

	tmm_pop_ref(N, k, array_init, temps, h_args_p, numChains);

	free(temps);
	free(array_init);
	cudaFree(d_array_init);
}

void test_gauss_mv(int N) {
	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	const int D = 2;
	int nb = 32;
	int nt = 128;
	int tt = nb * nt;

	float h_args_p[1 + D * D + D];
	float cov_p[D * D];
	matrix_set(cov_p, D, D, 0, 0, 1.0f);
	matrix_set(cov_p, D, D, 0, 1, 0.5f);
	matrix_set(cov_p, D, D, 1, 0, 0.5f);
	matrix_set(cov_p, D, D, 1, 1, 2.0f);

	float sigma = 1.0;

	compute_c1_c2(cov_p, D, h_args_p[0], h_args_p + 1);
	h_args_p[5] = 1;
	h_args_p[6] = 1;

	float* d_array_out;
	cudaMalloc((void **) &d_array_out, N * D * sizeof(float));

	float* d_array_init;
	cudaMalloc((void **) &d_array_init, tt * D * sizeof(float));
	populate_randn_d(d_array_init, tt * D);
	cudaThreadSynchronize();

	float h_sum[2];
	float result[2];

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);

	metropolis_rw_n_mv(N, D, d_array_init, sigma, d_array_out, h_args_p, 0, 32,
			128);

	cudaThreadSynchronize();

	reduce(N, D, d_array_out, h_sum, 32, 128);

	cudaThreadSynchronize();

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	result[0] = h_sum[0] / N;
	result[1] = h_sum[1] / N;
	printf("RESULT = (%f,%f)\n", result[0], result[1]);

	cudaFree(d_array_init);
	cudaFree(d_array_out);

	N = N / 32;

	float* h_array_init = (float*) malloc(D * sizeof(float));
	float* h_array_out = (float*) malloc(N * D * sizeof(float));

	for (int i = 0; i < D; i++) {
		h_array_init[i] = 0;
	}

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);

	metropolis_rw_ref_n_mv(N, D, h_array_init, sigma, h_array_out, h_args_p, 0);

	h_sum[0] = 0;
	h_sum[1] = 0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			h_sum[j] += vector_get(h_array_out, D, i)[j];
		}
	}

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	result[0] = h_sum[0] / N;
	result[1] = h_sum[1] / N;
	printf("HOST RESULT = (%f,%f)\n", result[0], result[1]);

	free(h_array_init);
	free(h_array_out);
}

void test_gauss_pop_mv(int N) {

	//	const int N = 8388608;
	const int D = 2;
	//	int N = 1048576;

	int nb = 32;
	int nt = 128;
	int tt = nb * nt;

	int M = N / tt;

	float h_args_p[1 + D * D + D];
	float cov_p[D * D];
	matrix_set(cov_p, D, D, 0, 0, 1.0f);
	matrix_set(cov_p, D, D, 0, 1, 0.5f);
	matrix_set(cov_p, D, D, 1, 0, 0.5f);
	matrix_set(cov_p, D, D, 1, 1, 2.0f);

	float sigma = 1.0;

	compute_c1_c2(cov_p, D, h_args_p[0], h_args_p + 1);
	h_args_p[5] = 14;
	h_args_p[6] = 14;

	float* d_array_out;
	cudaMalloc((void **) &d_array_out, N * D * sizeof(float));

	float* d_array_init;
	cudaMalloc((void **) &d_array_init, tt * D * sizeof(float));
	populate_randn_d(d_array_init, tt * D);
	cudaThreadSynchronize();
	float* dinit_orig = d_array_init;

	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	float* d_temps;
	cudaMalloc((void **) &d_temps, tt * sizeof(float));

	float* temps = (float*) malloc(tt * sizeof(float));
	for (int i = 0; i < tt; i++) {
		temps[i] = 1.0f / tt * (i + 1);
	}
	cudaMemcpy(d_temps, temps, tt * sizeof(float), cudaMemcpyHostToDevice);
	free(temps);

	float h_sum[2];
	h_sum[0] = 0;
	h_sum[1] = 0;

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);
	int i;

	for (i = 0; i < 10; i++) {
		float micro_sum[2];
		micro_sum[0] = 0;
		micro_sum[1] = 0;

		if (i != 0) {
			d_array_init = d_array_out + N * D - tt * D;
		}

		metropolis_rwpop_n_mv(N, D, d_array_init, sigma, h_args_p, d_temps,
				d_array_out, 0, nb, nt);

		cudaThreadSynchronize();

		float* h_array = (float*) malloc(N * D * sizeof(float));
		cudaMemcpy(h_array, d_array_out, N * D * sizeof(float),
				cudaMemcpyDeviceToHost);

		int j;
		for (j = (tt - 1) * D; j < N * D; j += tt * D) {
			micro_sum[0] += h_array[j];
			micro_sum[1] += h_array[j + 1];
		}

		free(h_array);

		cudaThreadSynchronize();

		micro_sum[0] /= M;
		micro_sum[1] /= M;

		h_sum[0] = h_sum[0] * ((float) i) / ((float) i + 1) + micro_sum[0] / (i
				+ 1);
		h_sum[1] = h_sum[1] * ((float) i) / ((float) i + 1) + micro_sum[1] / (i
				+ 1);
		printf("(CURRENT) RESULT = (%f,%f)\n", micro_sum[0], micro_sum[1]);

	}

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	printf("FINAL RESULT = (%f,%f)\n", h_sum[0], h_sum[1]);

	cudaFree(d_temps);
	cudaFree(d_array_out);
	cudaFree(dinit_orig);

}

void test_gauss() {
	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	int N = 8388608;
	int nb = 32;
	int nt = 128;

	float h_args_p[3];
	compute_c1_c2(1.0f, h_args_p[0], h_args_p[1]);
	h_args_p[2] = 10;

	float* d_array_init;
	float* d_array_out;
	cudaMalloc((void **) &d_array_init, nb * nt * sizeof(float));
	cudaMalloc((void **) &d_array_out, N * sizeof(float));
	populate_randn_d(d_array_init, nb * nt);
	cudaThreadSynchronize();

	float h_sum, result;

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);

	metropolis_rw_n(N, d_array_init, 1.0f, d_array_out, h_args_p, 0, nb, nt);

	cudaThreadSynchronize();

	reduce(N, d_array_out, h_sum, nb, nt);

	cudaThreadSynchronize();

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	result = h_sum / N;
	printf("SEPARATE ARRAY RESULT = %f\n", result);

	cudaFree(d_array_init);
	cudaFree(d_array_out);

	float* h_array_out = (float*) malloc(N * sizeof(float));

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);

	metropolis_rw_ref_n(N, 0, 1.0, h_array_out, h_args_p, 0);

	h_sum = 0;
	for (int i = 0; i < N; i++) {
		h_sum += h_array_out[i];
	}

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	result = h_sum / N;
	printf("HOST RESULT = %f\n", result);

	free(h_array_out);
}

void test_gauss_pop_host() {
	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	//	int N = 8388608;
	int N = 262144;
	int NT = 256;
	int M = N / NT;

	float h_args_p[3];
	compute_c1_c2(1.0f, h_args_p[0], h_args_p[1]);
	h_args_p[2] = 10;

	float* h_array_out = (float*) malloc(N * sizeof(float));

	float* init = (float*) malloc(NT * sizeof(float));
	populate_randn(init, NT);

	float* temps = (float*) malloc(NT * sizeof(float));
	for (int i = 0; i < NT; i++) {
		temps[i] = 1.0f / NT * (i + 1);
	}

	float h_sum, result;
	h_sum = 0;

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);
	int i;
	for (i = 0; i < 10; i++) {
		float micro_sum = 0;

		metropolis_rwpop_ref_n(N, init, 1.0, h_args_p, temps, h_array_out, 0,
				NT);

		int j;
		for (j = NT - 1; j < N; j += NT) {
			micro_sum += h_array_out[j];
		}

		micro_sum /= M;

		h_sum = h_sum * ((float) i) / ((float) i + 1) + micro_sum / (i + 1);
		result = h_sum;
		printf("(CURRENT) RESULT = %f\n", micro_sum);

	}

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	printf("FINAL RESULT = %f\n", result);

	free(h_array_out);
	free(temps);
	free(init);

}

void test_gauss_pop() {
	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	int N = 8388608;
	//	int N = 1048576;

	int nb = 32;
	int nt = 32;
	int tt = nb * nt;

	int M = N / tt;

	float h_args_p[3];
	compute_c1_c2(1.0f, h_args_p[0], h_args_p[1]);
	h_args_p[2] = 10;

	float* d_array_out;
	float* d_temps;
	float* d_array_init;
	cudaMalloc((void **) &d_array_init, tt * sizeof(float));
	cudaMalloc((void **) &d_array_out, N * sizeof(float));
	cudaMalloc((void **) &d_temps, tt * sizeof(float));
	float* dinit_orig = d_array_init;

	float* temps = (float*) malloc(tt * sizeof(float));
	for (int i = 0; i < tt; i++) {
		temps[i] = 1.0f / tt * (i + 1);
	}
	cudaMemcpy(d_temps, temps, tt * sizeof(float), cudaMemcpyHostToDevice);
	free(temps);

	populate_randn_d(d_array_init, tt);

	float h_sum, result;
	h_sum = 0;

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);
	int i;

	for (i = 0; i < 10; i++) {
		float micro_sum = 0;

		if (i != 0) {
			d_array_init = d_array_out + N - tt;
		}

		metropolis_rwpop_n(N, d_array_init, 1.0, h_args_p, d_temps,
				d_array_out, 1, nb, nt);

		cudaThreadSynchronize();

		float* h_array = (float*) malloc(N * sizeof(float));
		cudaMemcpy(h_array, d_array_out, N * sizeof(float),
				cudaMemcpyDeviceToHost);

		int j;
		for (j = tt - 1; j < N; j += tt) {
			micro_sum += h_array[j];
		}

		free(h_array);

		cudaThreadSynchronize();

		micro_sum /= M;

		h_sum = h_sum * ((float) i) / ((float) i + 1) + micro_sum / (i + 1);
		result = h_sum;
		printf("(CURRENT) RESULT = %f\n", micro_sum);

	}

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	printf("FINAL RESULT = %f\n", result);

	cudaFree(d_temps);
	cudaFree(d_array_out);
	cudaFree(dinit_orig);
}

void test_mix_gauss(int N, int numIterations, float* h_args_p) {
	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	float h_sum, result;
	h_sum = 0;

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);

	float init = 0.0;
	float* h_array = (float*) malloc(N * numIterations * sizeof(float));

	int i;
	for (i = 0; i < numIterations; i++) {

		float micro_sum = 0;

		metropolis_rw_ref_mn(N, init, 1.0, h_array + N * i, h_args_p, 1);

		int j;
		for (j = 0; j < N; j++) {
			micro_sum += h_array[i * N + j];
		}

		init = h_array[i * N + N - 1];

		micro_sum /= N;

		h_sum = h_sum * ((float) i) / ((float) i + 1) + micro_sum / (i + 1);
		result = h_sum;
		printf("(CURRENT) RESULT = %f\n", micro_sum);

	}

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	result = h_sum;

	printf("FINAL RESULT = %f\n", result);

	to_file(h_array, N * numIterations, 1, "mix_gauss.txt");

	free(h_array);

}

void test_mix_gauss_pop(int N, int numIterations, float* h_args_p,
		float* temps, int nb, int nt) {

	printf("MIX GAUSS POP\n");
	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	int tt = nb * nt;

	int M = N / tt;

	float* d_array_out;
	float* d_temps;
	float* d_array_init;
	cudaMalloc((void **) &d_array_init, tt * sizeof(float));
	cudaMalloc((void **) &d_array_out, N * sizeof(float));
	cudaMalloc((void **) &d_temps, tt * sizeof(float));
	float* dinit_orig = d_array_init;

	float* h_full_array = (float*) malloc(numIterations * M * sizeof(float));

	cudaMemcpy(d_temps, temps, tt * sizeof(float), cudaMemcpyHostToDevice);

	populate_randn_d(d_array_init, tt);

	float h_sum, result;
	h_sum = 0;

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);
	int i;
	for (i = 0; i < numIterations; i++) {

		float micro_sum = 0;

		if (i != 0) {
			d_array_init = d_array_out + N - tt;
		}

		metropolis_rwpop_mn(N, d_array_init, 1.0, h_args_p, d_temps,
				d_array_out, 1, nb, nt);

		cudaThreadSynchronize();

		float* h_array = (float*) malloc(N * sizeof(float));
		cudaMemcpy(h_array, d_array_out, N * sizeof(float),
				cudaMemcpyDeviceToHost);

		int j;

		for (j = 0; j < M; j++) {
			h_full_array[i * M + j] = h_array[j * tt - 1];
			micro_sum += h_array[j * tt - 1];
		}

		free(h_array);

		micro_sum /= M;

		h_sum = h_sum * ((float) i) / ((float) i + 1) + micro_sum / (i + 1);
		result = h_sum;
		printf("(CURRENT) RESULT = %f\n", micro_sum);

	}

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	result = h_sum;

	printf("FINAL RESULT = %f\n", result);

	to_file(h_full_array, M * numIterations, 1, "mix_gauss_pop.txt");

	cudaFree(d_array_out);
	cudaFree(d_temps);
	cudaFree(dinit_orig);
	free(h_full_array);
}

void test_mix_gauss_pop_host(int N, int numIterations, float* h_args_p,
		float* temps, int nb, int nt) {
	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	int NT = nb * nt;
	int M = N / NT;

	float* h_array_out = (float*) malloc(N * sizeof(float));
	float* h_full_array = (float*) malloc(numIterations * M * sizeof(float));

	float* init = (float*) malloc(NT * sizeof(float));
	populate_randn(init, NT);

	float* init_orig = init;

	float h_sum = 0;
	float result = 0;

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);
	int i;
	for (i = 0; i < numIterations; i++) {
		float micro_sum = 0;

		if (i != 0) {
			init = h_array_out + N - NT;
		}

		metropolis_rwpop_ref_mn(N, init, 1.0, h_args_p, temps, h_array_out, 1,
				NT);

		cudaThreadSynchronize();

		int j;

		for (j = 0; j < M; j++) {
			h_full_array[i * M + j] = h_array_out[j * NT - 1];
			micro_sum += h_array_out[j * NT - 1];
		}

		micro_sum /= M;

		h_sum = h_sum * ((float) i) / ((float) i + 1) + micro_sum / (i + 1);
		result = h_sum;
		printf("(CURRENT) RESULT = %f\n", micro_sum);

	}

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	to_file(h_full_array, M * numIterations, 1, "mix_gauss_pop_ref.txt");

	printf("FINAL RESULT = %f\n", result);

	free(h_array_out);
	free(h_full_array);
	free(init_orig);

}

void test_mix_gauss(int N, int numIterations, float sigma, float mu1,
		float mu2, int nb, int nt) {
	const int k = 2;
	float w = 1.0 / k;

	float h_args_p[1 + 3 * k];
	h_args_p[0] = k;

	h_args_p[1] = mu1;
	h_args_p[2] = mu2;
	float* c1s = h_args_p + 1 + k;
	float* c2s = h_args_p + 1 + 2 * k;

	compute_ci1_ci2(sigma, w, c1s[0], c2s[0]);
	compute_ci1_ci2(sigma, w, c1s[1], c2s[1]);

	int tt = nb * nt;

	float* temps = (float*) malloc(tt * sizeof(float));
	for (int i = 0; i < tt; i++) {
		float j = (float) i + 1;
		temps[i] = (j * j) / (tt * tt);
	}

	test_mix_gauss_pop(N, numIterations, h_args_p, temps, nb, nt);
	test_mix_gauss_pop_host(N, numIterations, h_args_p, temps, nb, nt);
	test_mix_gauss(N, numIterations, h_args_p);

	free(temps);

}

int main(int argc, char **argv) {

	seed_rng(32768);

	// test_gauss_mv(8388608);
	// test_gauss_pop_mv(1048576);

	// int M = 8192;
	// int M = 32768;
	int M = 8192;
	// int M = 1048576;

	// test_mixture_mus(M, 1, 1); // 1
	// test_mixture_mus(M, 2, 1); // 2
	// test_mixture_mus(M, 4, 1); // 4
	test_mixture_mus(M, 8, 1); // 8
	test_mixture_mus(M, 16, 2); // 32
	test_mixture_mus(M, 64, 2); // 128
	test_mixture_mus(M, 32, 16); // 512
	test_mixture_mus(M, 32, 64); // 2048
	test_mixture_mus(M, 128, 64); // 8192
	test_mixture_mus(M, 256, 128); // 32768
	// test_mixture_mus(M, 512, 128); // 65536
	test_mixture_mus(M, 512, 256); // 131072

	//	test_gauss_mv(1048576);
	//	test_gauss_pop_mv(1048576);

	//	test_gauss();
	//	test_mix_gauss();

	//	test_gauss_pop();
	//	test_gauss_pop_host();

	// mix pop
	//	int N = 8388608;
	//	int N = 262144;
	//  int N = 1048576;
	//  float sigma = 1.0;
	//  float mu1 = 0;
	//  float mu2 = 10;
	//  int nb = 1;
	//  int nt = 32;
	//  int numIterations = 5;
	//
	//  test_mix_gauss(N, numIterations, sigma, mu1, mu2, nb, nt);

	// mix pop host
	//	test_mix_gauss_pop_host();

	kill_rng();

}
