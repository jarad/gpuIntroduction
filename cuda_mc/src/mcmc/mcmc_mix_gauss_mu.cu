/*
 * mcmc_mix_gauss_mu.cu
 *
 *  Created on: 26-Feb-2009
 *      Author: alee
 */

#include "mcmc_mix_gauss_mu.h"

#include "func.h"
#include "mix_gauss_uniform.ch"

#define TARGET mgu_mu_pdf
#define LOG_TARGET log_mgu_mu_pdf
#define TYPE mgumu_mv
#define NUM_AP 105

#include "mcmc_kernel_mv.cu"

void FUNC(metropolis_rwpop, TYPE)(int N, int D, float* d_array_init, float sigma,
		float* h_args_p, float* d_temps, float* d_array_out, int log, int nb,
		int nt) {
	switch (D) {
	case 1:
		FUNC(metropolis_rwpop, TYPE)<1>(N, d_array_init, sigma, h_args_p, d_temps,
				d_array_out, log, nb, nt);
		break;
	case 2:
		FUNC(metropolis_rwpop, TYPE)<2>(N, d_array_init, sigma, h_args_p, d_temps,
				d_array_out, log, nb, nt);
		break;
	case 3:
		FUNC(metropolis_rwpop, TYPE)<3>(N, d_array_init, sigma, h_args_p, d_temps,
				d_array_out, log, nb, nt);
		break;
	case 4:
		FUNC(metropolis_rwpop, TYPE)<4>(N, d_array_init, sigma, h_args_p, d_temps,
				d_array_out, log, nb, nt);
		break;
	default:
		break;
	}
}

void FUNC(metropolis_rw, TYPE)(int N, int D, float* d_array_init, float sigma,
		float* d_array_out, float* h_args_p, int log, int nb, int nt) {
	switch (D) {
	case 1: FUNC(metropolis_rw, TYPE)<1>(N, d_array_init, sigma, d_array_out, h_args_p, log, nb, nt);
			break;
	case 2: FUNC(metropolis_rw, TYPE)<2>(N, d_array_init, sigma, d_array_out, h_args_p, log, nb, nt);
			break;
	case 3: FUNC(metropolis_rw, TYPE)<3>(N, d_array_init, sigma, d_array_out, h_args_p, log, nb, nt);
			break;
	case 4: FUNC(metropolis_rw, TYPE)<4>(N, d_array_init, sigma, d_array_out, h_args_p, log, nb, nt);
			break;
	default:
		break;
	}
}

void FUNC(metropolis_rwpop_marginal, TYPE)(int N, int D, float* d_array_init,
		float sigma, float* h_args_p, float* d_temps, float* d_array_out,
		int log, int nb, int nt) {
	switch (D) {
	case 1: FUNC(metropolis_rwpop_marginal, TYPE)<1>(N, d_array_init, sigma, h_args_p, d_temps,
			d_array_out, log, nb, nt);
			break;
	case 2: FUNC(metropolis_rwpop_marginal, TYPE)<2>(N, d_array_init, sigma, h_args_p, d_temps,
				d_array_out, log, nb, nt);
				break;
	case 3: FUNC(metropolis_rwpop_marginal, TYPE)<3>(N, d_array_init, sigma, h_args_p, d_temps,
				d_array_out, log, nb, nt);
				break;
	case 4: FUNC(metropolis_rwpop_marginal, TYPE)<4>(N, d_array_init, sigma, h_args_p, d_temps,
				d_array_out, log, nb, nt);
				break;
	default:
			break;
	}
}
