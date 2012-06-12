/*
 * mcmc_gauss_mv.h
 *
 *  Created on: 24-Feb-2009
 *      Author: alee
 */

#ifndef MCMC_GAUSS_MV_H_
#define MCMC_GAUSS_MV_H_

void metropolis_rw_n_mv(int N, int D, float* d_array_init, float sigma,
		float* d_array_out, float* h_args_p, int log, int nb, int nt);

void metropolis_rwpop_n_mv(int N, int D, float* d_array_init, float sigma,
		float* h_args_p, float* d_temps, float* d_array_out, int log, int nb,
		int nt);

void metropolis_rwpop_marginal_n_mv(int N, float* d_array_init, float sigma,
		float* h_args_p, float* d_temps, float* d_array_out, int log, int nb,
		int nt);

void metropolis_rw_ref_n_mv(int N, int D, float* init, float sigma,
		float* array_out, float* args_p, int log);

void metropolis_rwpop_marginal_ref_n_mv(int N, int D, float* array_init,
		float sigma, float* h_args_p, float* temps, float* array_out, int log,
		int numChains);


#endif /* MCMC_GAUSS_MV_H_ */
