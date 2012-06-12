/*
 * smcs_mix_gauss_mu.cu
 *
 *  Created on: 26-Feb-2009
 *      Author: alee
 */

#include "func.h"
#include "mix_gauss_uniform.ch"
#include "uniform.ch"

//#define TARGET1 uniform_pdf
#define LOG_TARGET1 log_uniform_pdf
//#define TARGET2 mgu_mu_pdf
#define LOG_TARGET2 log_mgu_mu_pdf
#define TYPE mgumu_mv
#define NUM_AT1 2
#define NUM_AT2 105

#include "smcs_kernel_mv.cu"

void FUNC(smcs, TYPE)(float* x_init, float* x, float* w, int N, int D, int T, int numSteps, float* cov_step, float* h_args_t1, float* h_args_t2, float* temps, float* ll, int nb, int nt) {
	switch (D) {
	case 1:
		FUNC( smcs, TYPE) < 1 > (x_init, x, w, N, T, numSteps, cov_step, h_args_t1, h_args_t2, temps, ll, nb, nt);
		break;
	case 4:
		FUNC( smcs, TYPE) < 4 > (x_init, x, w, N, T, numSteps, cov_step, h_args_t1, h_args_t2, temps, ll, nb, nt);
		break;
	default:
		break;
	}

}

void FUNC(smcs_forget, TYPE)(
		float* x_init, float* x, float* w, int N, int D,
		int T, int numSteps, float* cov_step, float* h_args_t1, float* h_args_t2,
		float* temps, float* ll, int nb, int nt) {
	switch (D) {
	case 1:
		FUNC( smcs_forget, TYPE) < 1 > (x_init, x, w, N, T, numSteps, cov_step, h_args_t1, h_args_t2, temps, ll, nb, nt);
		break;
	case 4:
		FUNC( smcs_forget, TYPE) < 4 > (x_init, x, w, N, T, numSteps, cov_step, h_args_t1, h_args_t2, temps, ll, nb, nt);
		break;
	default:
		break;
	}
}
