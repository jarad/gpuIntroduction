/*
 * smc_mvlg.cu
 *
 *  Created on: 10-Mar-2009
 *      Author: alee
 */

#include "func.h"
#include "gauss.ch"

#define LIKELIHOOD gauss_unequal_pdf
//#define LOG_LIKELIHOOD log_gauss_unequal_pdf
#define TYPE mvlg

#define NUM_AL 100

#include "smc_kernel_mv.cu"

void FUNC( smc, TYPE)(float* x_init, float* x, float* w, float* y, int N, int Dx,
		int Dy, int T, float* h_args_l, float* scale_step, float* cov_step,
		float* ll, int nb, int nt) {
	switch (Dx) {
	case 2:
		switch (Dy) {
		case 2:
			FUNC( smc, TYPE) <2, 2> (x_init, x, w, y, N, T, h_args_l, scale_step, cov_step, ll, nb, nt);
			break;
		default:
			break;
		}
	case 3:
		switch (Dy) {
		case 5:
			FUNC( smc, TYPE) <3, 5> (x_init, x, w, y, N, T, h_args_l, scale_step, cov_step, ll, nb, nt);
			break;
		default:
			break;
		}
	default:
		break;
	}

}

void FUNC( smc_forget, TYPE)(float* x_init, float* x, float* w, float* y, int N, int Dx,
		int Dy, int T, float* h_args_l, float* scale_step, float* cov_step,
		float* ll, int nb, int nt) {
	switch (Dx) {
	case 2:
		switch (Dy) {
		case 2:
			FUNC( smc_forget, TYPE) <2, 2> (x_init, x, w, y, N, T, h_args_l, scale_step, cov_step, ll, nb, nt);
			break;
		default:
			break;
		}
	case 3:
		switch (Dy) {
		case 5:
			FUNC( smc_forget, TYPE) <3, 5> (x_init, x, w, y, N, T, h_args_l, scale_step, cov_step, ll, nb, nt);
			break;
		default:
			break;
		}
	default:
		break;
	}

}
