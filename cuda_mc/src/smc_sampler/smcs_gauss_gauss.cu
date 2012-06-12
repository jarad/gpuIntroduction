/*
 * smcs_gauss_gauss.cu
 *
 *  Created on: 17-Mar-2009
 *      Author: alee
 */

#include "func.h"
#include "gauss.ch"

//#define TARGET1 gauss_pdf
#define LOG_TARGET1 log_gauss_pdf
//#define TARGET2 gauss_pdf
#define LOG_TARGET2 log_gauss_pdf
#define TYPE gg_mv
#define NUM_AT1 100 // up to 9D
#define NUM_AT2 100 // up to 9D

#include "smcs_kernel_mv.cu"

void FUNC( smcs, TYPE)(float* x_init, float* x, float* w, int N, int D, int T, int numSteps,
		float* cov_step, float* h_args_t1, float* h_args_t2, float* temps,
		float* ll, int nb, int nt) {
	switch (D) {
	case 1:
		FUNC( smcs, TYPE) < 1
				> (x_init, x, w, N, T, numSteps, cov_step, h_args_t1, h_args_t2, temps, ll, nb, nt);
		break;
	case 2:
			FUNC( smcs, TYPE) < 2
					> (x_init, x, w, N, T, numSteps, cov_step, h_args_t1, h_args_t2, temps, ll, nb, nt);
			break;
	default:
		break;
	}

}

void FUNC( smcs_forget, TYPE)(float* x_init, float* x, float* w, int N, int D,
		int T, int numSteps, float* cov_step, float* h_args_t1, float* h_args_t2,
		float* temps, float* ll, int nb, int nt) {
	switch (D) {
	case 1:
		FUNC( smcs_forget, TYPE) < 1
				> (x_init, x, w, N, T, numSteps, cov_step, h_args_t1, h_args_t2, temps, ll, nb, nt);
		break;
	case 2:
		FUNC( smcs_forget, TYPE) < 2
						> (x_init, x, w, N, T, numSteps, cov_step, h_args_t1, h_args_t2, temps, ll, nb, nt);
		break;
	default:
		break;
	}
}
