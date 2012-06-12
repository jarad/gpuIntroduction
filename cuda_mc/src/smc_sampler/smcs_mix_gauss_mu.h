/*
 * smcs_mix_gauss.mu.h
 *
 *  Created on: 17-Mar-2009
 *      Author: Owner
 */

#ifndef SMCS_MIX_GAUSS_MU_H_
#define SMCS_MIX_GAUSS_MU_H_

void smcs_mgumu_mv(float* x_init, float* x, float* w, int N, int D, int T, int numSteps,
		float* cov_step, float* h_args_t1, float* h_args_t2, float* temps,
		float* ll, int nb, int nt);

void smcs_forget_mgumu_mv(float* x_init, float* x, float* w, int N, int D, int T, int numSteps,
		float* cov_step, float* h_args_t1, float* h_args_t2, float* temps,
		float* ll, int nb, int nt);

void smcs_ref_mgumu_mv(float* x_init, float* x, float* w, int N, int D, int T, int numSteps,
		float* cov_step, float* h_args_t1, float* h_args_t2, float* temps,
		float* ll);

#endif /* SMCS_MIX_GAUSS_MU_H_ */
