/*
 * smc_fsv.h
 *
 *  Created on: 10-Mar-2009
 *      Author: alee
 */

#ifndef SMC_FSV_H_
#define SMC_FSV_H_

void smc_forget_fsv(float* x_init, float* x, float* w, float* y, int N, int Dx, int Dy,
		int T, float* h_args_l, float* scale_step, float* cov_step, float* ll,
		int nb, int nt);

void smc_fsv(float* x_init, float* x, float* w, float* y, int N, int Dx, int Dy, int T,
		float* h_args_l, float* scale_step, float* cov_step, float* ll, int nb,
		int nt);

//void smc_fsv(float* x_init, float* x, float* w, float* y, int N, int Dx, int Dy, int T,
//		float* h_args_l, float* scale_step, float* cov_step, float* ll,
//		int smooth, int nb, int nt);

void smc_ref_forget_fsv(float* x_init, float* w, float* x, float* y, int N, int Dx,
		int Dy, int T, float* h_args_l, float* scale_step, float* cov_step,
		float* ll);

void smc_ref_fsv(float* x_init, float* x, float* w, float* y, int N, int Dx, int Dy,
		int T, float* h_args_l, float* scale_step, float* cov_step, float* ll);

void smc_ref_fsv(double* x_init, double* x, double* w, double* y, int N, int Dx, int Dy, int T,
        double* h_args_l, double* scale_step, double* cov_step, double* ll);

void smc_ref_forget_fsv(double* x_init, double* x, double* w, double* y, int N, int Dx, int Dy, int T,
        double* h_args_l, double* scale_step, double* cov_step, double* ll);

#endif /* SMC_FSV_H_ */
