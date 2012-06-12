/*
 * mix_gauss_normal.inl
 *
 *  Created on: 26-Feb-2009
 *      Author: alee
 */

#include "mix_gauss.ch"
#include <float.h>

template <int k>
__device__ float mgu_pdf(float* x, int N, float* mus, float c1,
		float c2, float min, float max) {
	for (int i = 0; i < k; i++) {
		if (mus[i] < min || mus[i] > max) {
			return 0;
		}
	}
	return mix_gauss1_mus_pdf<k>(x, N, c1, c2, mus);
}

template <int k>
__device__ float mgu_mu_pdf(float* mus, float* args) {
	int N = (int) args[0];
	float* x = args + 1;
	float c1 = args[N + 1];
	float c2 = args[N + 2];
	float min = args[N + 3];
	float max = args[N + 4];
	return mgu_pdf<k>(x, N, mus, c1, c2, min, max);
}

template <int k>
__device__ float log_mgu_pdf(float* x, int N, float* mus, float c1,
		float c2, float min, float max) {
	for (int i = 0; i < k; i++) {
		if (mus[i] < min || mus[i] > max) {
			return -FLT_MAX;
		}
	}
	return log_mix_gauss1_mus_pdf<k>(x, N, c1, c2, mus);
}

template <int k>
__device__ float log_mgu_mu_pdf(float* mus, float* args) {
	int N = (int) args[0];
	float* x = args + 1;
	float c1 = args[N + 1];
	float c2 = args[N + 2];
	float min = args[N + 3];
	float max = args[N + 4];
	return log_mgu_pdf<k>(x, N, mus, c1, c2, min, max);
}
