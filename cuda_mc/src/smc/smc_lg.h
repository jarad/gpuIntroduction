/*
 * smc_lg.h
 *
 *  Created on: 10-Mar-2009
 *      Author: alee
 */

#ifndef SMC_LG_H_
#define SMC_LG_H_

void smc_lg(float* x_init, float* x, float* w, float* y, int N, int T, float* h_args_l,
        float scale_step, float sigma_step, float* ll, int nb, int nt);

void smc_forget_lg(float* x_init, float* w, float* x, float* y, int N, int T,
		float* h_args_l, float scale_step, float sigma_step, float* ll, int nb, int nt);

void smc_ref_lg(float* x_init, float* x, float* w, float* y, int N, int T, float* h_args_l,
        float scale_step, float sigma_step, float* ll);

void smc_ref_forget_lg(float* x_init, float* x, float* w, float* y, int N, int T, float* h_args_l,
        float scale_step, float sigma_step, float* ll);

#endif /* SMC_LG_H_ */
