/*
 * smc_usv.h
 *
 *  Created on: 09-Jul-2009
 *      Author: alee
 */

#ifndef SMC_USV_H_
#define SMC_USV_H_

void smc_usv(float* x_init, float* x, float* w, float* y, int N, int T, float* h_args_l,
        float scale_step, float sigma_step, float* ll, int nb, int nt);

void smc_forget_usv(float* x_init, float* w, float* x, float* y, int N, int T,
        float* h_args_l, float scale_step, float sigma_step, float* ll, int nb, int nt);

void smc_ref_usv(float* x_init, float* x, float* w, float* y, int N, int T, float* h_args_l,
        float scale_step, float sigma_step, float* ll);

void smc_ref_forget_usv(float* x_init, float* x, float* w, float* y, int N, int T, float* h_args_l,
        float scale_step, float sigma_step, float* ll);


#endif /* SMC_USV_H_ */
