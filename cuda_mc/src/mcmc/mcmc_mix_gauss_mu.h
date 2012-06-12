/*
 * mcmc_mix_gauss_mu.h
 *
 *  Created on: 26-Feb-2009
 *      Author: alee
 */

#ifndef MCMC_MIX_GAUSS_MU_H_
#define MCMC_MIX_GAUSS_MU_H_

void metropolis_rw_mgumu_mv(int N, int D, float* d_array_init, float sigma,
		float* d_array_out, float* h_args_p, int log, int nb, int nt);

void metropolis_rwpop_mgumu_mv(int N, int D, float* d_array_init, float sigma,
		float* h_args_p, float* d_temps, float* d_array_out, int log, int nb,
		int nt);

void metropolis_rwpop_marginal_mgumu_mv(int N, int D, float* d_array_init,
		float sigma, float* h_args_p, float* d_temps, float* d_array_out,
		int log, int nb, int nt);

void metropolis_rw_ref_mgumu_mv(int N, int D, float* h_array_init, float sigma,
		float* h_array_out, float* h_args_p, int log);

void metropolis_rwpop_marginal_ref_mgumu_mv(int N, int D, float* array_init,
		float sigma, float* h_args_p, float* temps, float* array_out,
		int log, int numChains);

#endif /* MCMC_MIX_GAUSS_MU_H_ */
