/*
 * smc_mvlg.h
 *
 *  Created on: 10-Mar-2009
 *      Author: alee
 */

#ifndef SMC_MVLG_H_
#define SMC_MVLG_H_

void smc_forget_mvlg(float* x_init, float* x, float* w, float* y, int N, int Dx, int Dy,
		int T, float* h_args_l, float* scale_step, float* cov_step, float* ll,
		int nb, int nt);

void smc_mvlg(float* x_init, float* x, float* w, float* y, int N, int Dx, int Dy, int T,
		float* h_args_l, float* scale_step, float* cov_step, float* ll, int nb,
		int nt);

void smc_ref_forget_mvlg(float* x_init, float* x, float* w, float* y, int N, int Dx, int Dy,
		int T, float* h_args_l, float* scale_step, float* cov_step, float* ll);

void smc_ref_mvlg(float* x_init, float* x, float* w, float* y, int N, int Dx, int Dy, int T,
		float* h_args_l, float* scale_step, float* cov_step, float* ll);

void smc_ref_mvlg(double* x_init, double* x, double* w, double* y, int N, int Dx, int Dy, int T,
        double* h_args_l, double* scale_step, double* cov_step, double* ll);

#endif /* SMC_MVLG_H_ */
