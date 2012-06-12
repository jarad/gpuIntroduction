/*
 * smcs_gauss_gauss.h
 *
 *  Created on: 18-Mar-2009
 *      Author: alee
 */

#ifndef SMCS_GAUSS_GAUSS_H_
#define SMCS_GAUSS_GAUSS_H_

void smcs_gg_mv(float* x_init, float* x, float* w, int N, int D, int T, int numSteps,
		float* cov_step, float* h_args_t1, float* h_args_t2, float* temps,
		float* ll, int nb, int nt);

void smcs_forget_gg_mv(float* x_init, float* x, float* w, int N, int D, int T, int numSteps,
		float* cov_step, float* h_args_t1, float* h_args_t2, float* temps,
		float* ll, int nb, int nt);

void smcs_ref_gg_mv(float* x_init, float* x, float* w, int N, int D, int T, int numSteps,
		float* cov_step, float* h_args_t1, float* h_args_t2, float* temps,
		float* ll);


#endif /* SMCS_GAUSS_GAUSS_H_ */
