/*
 * mcmc_mix_gauss.h
 *
 *  Created on: 04-Feb-2009
 *      Author: alee
 */

#ifndef MCMC_MIX_GAUSS_H_
#define MCMC_MIX_GAUSS_H_

void metropolis_rw_mn(int size, float* d_array_init, float sigma,
		float* d_array_out, float* h_args_p, int log, int nb,
		int nt);

//void metropolis_rw_steps_mn(int size, float* d_array_init, float sigma,
//		float* d_array_out, float* h_args_p, int nb,
//		int nt);

void metropolis_rwpop_mn(int size, float* d_array_init, float sigma,
		float* h_args_p, float* d_temps, float* d_array_out, int log, int nb, int nt);

void metropolis_rw_ref_mn(int size, float init, float sigma,
		float* d_array_out, float* args_p, int log);

void metropolis_rwpop_ref_mn(int size, float* init, float sigma, float* h_args_p,
		float* temps, float* h_array_out, int log, int nt);

#endif /* MCMC_MIX_GAUSS_H_ */
