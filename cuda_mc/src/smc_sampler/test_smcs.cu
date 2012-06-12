/*
 * test_smcs.cu
 *
 *  Created on: 17-Mar-2009
 *      Author: Owner
 */

#include <stdio.h>
#include "scan.h"
#include <cutil.h>
#include "rng.h"
#include "gauss.h"
#include "output.h"
#include "test_functions.h"
#include "smcs_mix_gauss_mu.h"
#include "smcs_gauss_gauss.h"
#include "mix_gauss.h"
#include "matrix.h"

void testMG(int N, int D, int T, int numSteps, float* d_array_init,
		float* temps, float* h_args_t1, float* h_args_t2, int nb, int nt) {
	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	float* d_array_out;
	cudaMalloc((void **) &d_array_out, N * T * D * sizeof(float));

	float* h_array_out = (float*) malloc(N * T * D * sizeof(float));
	float* h_sum = (float*) malloc(D * sizeof(float));
	float* result = (float*) malloc(D * sizeof(float));

	float ll;

	float* w = (float*) malloc(N * sizeof(float));

	float* d_temps;
	cudaMalloc((void **) &d_temps, T * sizeof(float));

	cudaMemcpy(d_temps, temps, T * sizeof(float), cudaMemcpyHostToDevice);

	float* d_w;
	cudaMalloc((void**) &d_w, N * sizeof(float));

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);

	smcs_mgumu_mv(d_array_init, d_array_out, d_w, N, D, T, numSteps, NULL,
			h_args_t1, h_args_t2, d_temps, &ll, nb, nt);

	cudaThreadSynchronize();

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	printf("ll = %f\n", ll);

	cudaMemcpy(h_array_out, d_array_out, N * T * D * sizeof(float),
			cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();

	cudaMemcpy(w, d_w, N * sizeof(float), cudaMemcpyDeviceToHost);

	to_file(h_array_out, N * T, D, "smcs_mgmu.txt");
	to_file(w, N, 1, "smcs_mgmu_w.txt");

	for (int i = 0; i < D; i++) {
		h_sum[i] = 0;
	}
	float* final_results = h_array_out + N * (T - 1) * D;
	float sumw = 0;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			h_sum[j] += w[i] * vector_get(final_results, D, i)[j];
		}
		sumw += w[i];
	}

	for (int j = 0; j < D; j++) {
		result[j] = h_sum[j] / sumw;
	}

	printf("SMCS RESULT = (%f,%f,%f,%f)\n", result[0], result[1], result[2],
			result[3]);

	free(h_sum);
	free(result);

	cudaFree(d_array_out);
	cudaFree(d_w);
	cudaFree(d_temps);
	free(h_array_out);
	free(w);
}

void testMG_forget(int N, int D, int T, int numSteps, float* d_array_init,
		float* temps, float* h_args_t1, float* h_args_t2, int nb, int nt) {
	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	float* d_array_out;

	cudaMalloc((void **) &d_array_out, N * D * sizeof(float));

	float* h_array_out = (float*) malloc(N * D * sizeof(float));

	float* h_sum = (float*) malloc(D * sizeof(float));
	float* result = (float*) malloc(D * sizeof(float));

	float ll;

	float* w = (float*) malloc(N * sizeof(float));

	float* d_temps;
	cudaMalloc((void **) &d_temps, T * sizeof(float));

	cudaMemcpy(d_temps, temps, T * sizeof(float), cudaMemcpyHostToDevice);

	float* d_w;
	cudaMalloc((void**) &d_w, N * sizeof(float));

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);

	smcs_forget_mgumu_mv(d_array_init, d_array_out, d_w, N, D, T, numSteps,
			NULL, h_args_t1, h_args_t2, d_temps, &ll, nb, nt);

	cudaThreadSynchronize();

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	printf("ll = %f\n", ll);

	cudaMemcpy(h_array_out, d_array_out, N * D * sizeof(float),
			cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();

	cudaMemcpy(w, d_w, N * sizeof(float), cudaMemcpyDeviceToHost);

	char* fnamed = "smcs_mgmu_forget_000000.txt";
	char* fnamew = "smcs_mgmu_forget_w_000000.txt";

	if (N == 1024) {
		fnamed = "smcs_mgmu_forget_001024.txt";
		fnamew = "smcs_mgmu_forget_w_001024.txt";
	} else if (N == 2048) {
		fnamed = "smcs_mgmu_forget_002048.txt";
		fnamew = "smcs_mgmu_forget_w_002048.txt";
	} else if (N == 4096) {
		fnamed = "smcs_mgmu_forget_004096.txt";
		fnamew = "smcs_mgmu_forget_w_004096.txt";
	} else if (N == 8192) {
		fnamed = "smcs_mgmu_forget_008192.txt";
		fnamew = "smcs_mgmu_forget_w_008192.txt";
	} else if (N == 16384) {
		fnamed = "smcs_mgmu_forget_016384.txt";
		fnamew = "smcs_mgmu_forget_w_016384.txt";
	} else if (N == 32768) {
		fnamed = "smcs_mgmu_forget_032768.txt";
		fnamew = "smcs_mgmu_forget_w_032768.txt";
	} else if (N == 65536) {
		fnamed = "smcs_mgmu_forget_065536.txt";
		fnamew = "smcs_mgmu_forget_w_065536.txt";
	} else if (N == 131072) {
		fnamed = "smcs_mgmu_forget_131072.txt";
		fnamew = "smcs_mgmu_forget_w_131072.txt";
	}

	to_file(h_array_out, N, D, fnamed);
	to_file(w, N, 1, fnamew);

	for (int i = 0; i < D; i++) {
		h_sum[i] = 0;
	}
	float* final_results = h_array_out;
	float sumw = 0;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			h_sum[j] += w[i] * vector_get(final_results, D, i)[j];
		}
		sumw += w[i];
	}

	for (int j = 0; j < D; j++) {
		result[j] = h_sum[j] / sumw;
	}

	printf("SMCS RESULT = (%f,%f,%f,%f)\n", result[0], result[1], result[2],
			result[3]);

	cudaFree(d_array_out);
	free(h_array_out);
	cudaFree(d_w);
	free(w);
	cudaFree(d_temps);

	free(h_sum);
	free(result);
}

void testMG_host(int N, int D, int T, int numSteps, float* h_array_init,
		float* temps, float* h_args_t1, float* h_args_t2) {
	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	float* h_array_out = (float*) malloc(N * T * D * sizeof(float));

	populate_rand(h_array_init, N * D);

	for (int i = 0; i < N * D; i++) {
		//		h_array_init[i] = -10 + h_array_init[i]*20;
		if (h_array_init[i] > 10) {
			printf("%f\n", h_array_init[i]);
		}
	}

	float* h_sum = (float*) malloc(D * sizeof(float));
	float* result = (float*) malloc(D * sizeof(float));

	float ll;

	float* w = (float*) malloc(N * sizeof(float));

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);

	smcs_ref_mgumu_mv(h_array_init, h_array_out, w, N, D, T, numSteps, NULL,
			h_args_t1, h_args_t2, temps, &ll);

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	printf("ll = %f\n", ll);

	to_file(h_array_out, N * T, D, "smcs_mgmu_ref.txt");
	to_file(w, N, 1, "smcs_mgmu_ref_w.txt");

	for (int i = 0; i < D; i++) {
		h_sum[i] = 0;
	}
	float* final_results = h_array_out + N * (T - 1) * D;
	float sumw = 0;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			h_sum[j] += w[i] * vector_get(final_results, D, i)[j];
		}
		sumw += w[i];
	}

	for (int j = 0; j < D; j++) {
		result[j] = h_sum[j] / sumw;
	}

	printf("SMCS RESULT = (%f,%f,%f,%f)\n", result[0], result[1], result[2],
			result[3]);

	free(h_array_out);
	free(w);
	free(h_sum);
	free(result);
}

void testMG(int N, int nb, int nt) {
	const int k = 4;

	printf("N = %d\n", N);

	int T = 200;
	int numSteps = 10;

	const int L = 100;
	float sigma = 0.55f;
	float mus[4];
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

	float h_args_t1[2] = { -10, 10 };

	float h_args_t2[L + 5];
	h_args_t2[0] = L;

	for (int i = 0; i < L; i++) {
		h_args_t2[i + 1] = data_array[i];
	}

	h_args_t2[L + 1] = c1;
	h_args_t2[L + 2] = c2;
	h_args_t2[L + 3] = -10;
	h_args_t2[L + 4] = 10;

	float* temps = (float*) malloc(T * sizeof(float));
	for (int i = 0; i < T; i++) {
		float j = ((float) i + 1);
		temps[i] = j * j / (T * T);
	}

	float* d_array_init;
	cudaMalloc((void **) &d_array_init, N * k * sizeof(float));
	populate_rand_d(d_array_init, N * k);
	multiply(N * k, d_array_init, d_array_init, 20, nb, nt);
	cudaThreadSynchronize();
	add(N * k, d_array_init, d_array_init, -10, nb, nt);
	cudaThreadSynchronize();
	float* array_init = (float*) malloc(N * k * sizeof(float));
	cudaMemcpy(array_init, d_array_init, N * k * sizeof(float),
			cudaMemcpyDeviceToHost);

	kill_rng();
	seed_rng(32768, 32, 128);

	testMG(N, k, T, numSteps, d_array_init, temps, h_args_t1, h_args_t2, nb, nt);

	kill_rng();
	seed_rng(32768, 32, 128);

	testMG_forget(N, k, T, numSteps, d_array_init, temps, h_args_t1, h_args_t2,
			nb, nt);

	kill_rng();
	seed_rng(32768, 32, 128);

	testMG_host(N, k, T, numSteps, array_init, temps, h_args_t1, h_args_t2);

	free(temps);
	free(array_init);
	cudaFree(d_array_init);
}

void testGG(int N, int D, int T, int numSteps, float* temps, float* h_args_1,
		float* h_args_2, int nb, int nt) {
	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	float* d_array_init;
	float* d_array_out;

	cudaMalloc((void **) &d_array_init, N * D * sizeof(float));
	cudaMalloc((void **) &d_array_out, N * T * D * sizeof(float));
	populate_randn_d(d_array_init, N * D);

	float* h_array_out = (float*) malloc(N * T * D * sizeof(float));
	float* h_array_init = (float*) malloc(N * D * sizeof(float));
	cudaMemcpy(h_array_init, d_array_init, N * D * sizeof(float),
			cudaMemcpyDeviceToHost);

	float* h_sum = (float*) malloc(D * sizeof(float));
	float* result = (float*) malloc(D * sizeof(float));

	float ll;

	float* w = (float*) malloc(N * sizeof(float));

	float* d_temps;
	cudaMalloc((void **) &d_temps, T * sizeof(float));

	cudaMemcpy(d_temps, temps, T * sizeof(float), cudaMemcpyHostToDevice);

	float* d_w;
	cudaMalloc((void**) &d_w, N * sizeof(float));

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);

	smcs_gg_mv(d_array_init, d_array_out, d_w, N, D, T, numSteps, NULL,
			h_args_1, h_args_2, d_temps, &ll, nb, nt);

	cudaThreadSynchronize();

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	printf("ll = %f\n", ll);

	cudaMemcpy(h_array_out, d_array_out, N * T * D * sizeof(float),
			cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();

	cudaMemcpy(w, d_w, N * sizeof(float), cudaMemcpyDeviceToHost);

	to_file(h_array_out, N * T, D, "smcs_gg.txt");
	to_file(w, N, 1, "smcs_gg_w.txt");

	for (int i = 0; i < D; i++) {
		h_sum[i] = 0;
	}

	float* final_results = h_array_out + N * (T - 1) * D;
	float sumw = 0;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			h_sum[j] += w[i] * vector_get(final_results, D, i)[j];
		}
		sumw += w[i];
	}

	for (int j = 0; j < D; j++) {
		result[j] = h_sum[j] / sumw;
	}

	printf("SMCS RESULT = (%f,%f)\n", result[0], result[1]);

	cudaFree(d_array_init);
	cudaFree(d_array_out);
	cudaFree(d_temps);
	free(h_array_out);
	cudaFree(d_w);
	free(w);

	free(h_sum);
	free(result);
}

void testGG_host(int N, int D, int T, int numSteps, float* temps,
		float* h_args_1, float* h_args_2) {
	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	float* d_array_init;

	cudaMalloc((void **) &d_array_init, N * D * sizeof(float));
	populate_randn_d(d_array_init, N * D);

	float* h_array_out = (float*) malloc(N * T * D * sizeof(float));
	float* h_array_init = (float*) malloc(N * D * sizeof(float));
	cudaMemcpy(h_array_init, d_array_init, N * D * sizeof(float),
			cudaMemcpyDeviceToHost);

	float* h_sum = (float*) malloc(D * sizeof(float));
	float* result = (float*) malloc(D * sizeof(float));

	float ll;

	float* w = (float*) malloc(N * sizeof(float));

	float* d_temps;
	cudaMalloc((void **) &d_temps, T * sizeof(float));

	cudaMemcpy(d_temps, temps, T * sizeof(float), cudaMemcpyHostToDevice);

	float* d_w;
	cudaMalloc((void**) &d_w, N * sizeof(float));

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);

	smcs_ref_gg_mv(h_array_init, h_array_out, w, N, D, T, numSteps, NULL,
			h_args_1, h_args_2, temps, &ll);

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	printf("ll = %f\n", ll);

	to_file(h_array_out, N * T, D, "smcs_gg_ref.txt");
	to_file(w, N, 1, "smcs_gg_ref_w.txt");

	for (int i = 0; i < D; i++) {
		h_sum[i] = 0;
	}

	float* final_results = h_array_out + N * (T - 1) * D;
	float sumw = 0;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			h_sum[j] += w[i] * vector_get(final_results, D, i)[j];
		}
		sumw += w[i];
	}

	for (int j = 0; j < D; j++) {
		result[j] = h_sum[j] / sumw;
	}

	printf("SMCS HOST RESULT = (%f,%f)\n", result[0], result[1]);

	cudaFree(d_array_init);
	free(h_array_out);
	free(h_array_init);
	free(w);

	free(h_sum);
	free(result);
}

void testGG_forget(int N, int D, int T, int numSteps, float* temps,
		float* h_args_1, float* h_args_2, int nb, int nt) {
	unsigned int hTimer;
	double time;
	cutCreateTimer(&hTimer);

	float* d_array_init;
	float* d_array_out;

	cudaMalloc((void **) &d_array_init, N * D * sizeof(float));
	cudaMalloc((void **) &d_array_out, N * D * sizeof(float));
	populate_randn_d(d_array_init, N * D);

	float* h_array_out = (float*) malloc(N * D * sizeof(float));
	float* h_array_init = (float*) malloc(N * D * sizeof(float));
	cudaMemcpy(h_array_init, d_array_init, N * D * sizeof(float),
			cudaMemcpyDeviceToHost);

	float* h_sum = (float*) malloc(D * sizeof(float));
	float* result = (float*) malloc(D * sizeof(float));

	float ll;

	float* w = (float*) malloc(N * sizeof(float));
	float* d_temps;
	cudaMalloc((void **) &d_temps, T * sizeof(float));

	cudaMemcpy(d_temps, temps, T * sizeof(float), cudaMemcpyHostToDevice);

	float* d_w;
	cudaMalloc((void**) &d_w, N * sizeof(float));

	cutResetTimer(hTimer);
	cutStartTimer(hTimer);

	smcs_forget_gg_mv(d_array_init, d_array_out, d_w, N, D, T, numSteps, NULL,
			h_args_1, h_args_2, d_temps, &ll, nb, nt);

	cudaThreadSynchronize();

	cutStopTimer(hTimer);
	time = cutGetTimerValue(hTimer);
	printf("Time = %f\n", time);

	printf("ll = %f\n", ll);

	cudaMemcpy(h_array_out, d_array_out, N * D * sizeof(float),
			cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();

	cudaMemcpy(w, d_w, N * sizeof(float), cudaMemcpyDeviceToHost);

	to_file(h_array_out, N, D, "smcs_gg_forget.txt");
	to_file(w, N, 1, "smcs_gg_forget_w.txt");

	for (int i = 0; i < D; i++) {
		h_sum[i] = 0;
	}

	float* final_results = h_array_out;
	float sumw = 0;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			h_sum[j] += w[i] * vector_get(final_results, D, i)[j];
		}
		sumw += w[i];
	}

	for (int j = 0; j < D; j++) {
		result[j] = h_sum[j] / sumw;
	}

	printf("SMCS RESULT = (%f,%f)\n", result[0], result[1]);

	cudaFree(d_array_init);
	cudaFree(d_array_out);
	cudaFree(d_temps);
	free(h_array_out);
	cudaFree(d_w);
	free(w);

	free(h_sum);
	free(result);
}

void testGG(int N) {

	int nb = 256;
	int nt = 64;
	//	int tt = nb * nt;

	int T = 200;
	int numSteps = 10;

	const int D = 2;
	float mu_1[D] = { 0, 0 };
	float cov_1[D * D] = { 1.0, 0.0, 0.0, 1.0 };

	float c2_1[D * D];
	float c1_1;

	compute_c1_c2(cov_1, D, c1_1, c2_1);

	float* h_args_1 = (float*) malloc((1 + D + D * D) * sizeof(float));
	h_args_1[0] = c1_1;
	h_args_1[1] = c2_1[0];
	h_args_1[2] = c2_1[1];
	h_args_1[3] = c2_1[2];
	h_args_1[4] = c2_1[3];
	h_args_1[5] = mu_1[0];
	h_args_1[6] = mu_1[1];

	float mu_2[D] = { 10, 10 };
	float cov_2[D * D] = { 1.0, 0.0, 0.0, 1.0 };

	float c2_2[D * D];
	float c1_2;

	compute_c1_c2(cov_2, D, c1_2, c2_2);

	float* h_args_2 = (float*) malloc((1 + D + D * D) * sizeof(float));
	h_args_2[0] = c1_2;
	h_args_2[1] = c2_2[0];
	h_args_2[2] = c2_2[1];
	h_args_2[3] = c2_2[2];
	h_args_2[4] = c2_2[3];
	h_args_2[5] = mu_2[0];
	h_args_2[6] = mu_2[1];

	float* temps = (float*) malloc(T * sizeof(float));
	for (int i = 0; i < T; i++) {
		temps[i] = ((float) i + 1) / T;
	}

	kill_rng();
	seed_rng(32768, 32, 128);

	testGG(N, D, T, numSteps, temps, h_args_1, h_args_2, nb, nt);

	kill_rng();
	seed_rng(32768, 32, 128);

	testGG_forget(N, D, T, numSteps, temps, h_args_1, h_args_2, nb, nt);

	kill_rng();
	seed_rng(32768, 32, 128);

	testGG_host(N, D, T, numSteps, temps, h_args_1, h_args_2);

	free(temps);
	free(h_args_1);
	free(h_args_2);
}

int main(int argc, char **argv) {
	seed_rng(32768, 32, 128);
	int N = 16384;
	scan_init(N);

	//    testGG(16384);
	//    testGG(32768);

	//    testMG(1024, 32, 32);
	//    testMG(8192, 128, 64);
	    testMG(N, 256, 64); // N = 16384
	//    testMG(32768, 512, 64);
	//    testMG(65536, 512, 64);
	//    testMG(131072, 512, 64);
	//	testMG(262144, 512, 64);

	kill_rng();
	scan_destroy();
}
