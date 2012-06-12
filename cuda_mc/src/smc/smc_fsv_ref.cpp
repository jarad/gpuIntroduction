/*
 * smc_fsv_ref.cpp
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#include "func.h"
#include "fsv.h"

#define TYPE fsv

#define LIKELIHOOD_H fsv_pdfh

#include "smc_ref_mv.cpp"

void smc_ref_forget_fsv(float* x_init, float* x, float* w, float* y, int N, int Dx, int Dy,
        int T, float* h_args_l, float* scale_step, float* cov_step, float* ll) {
    smc_ref_forget_fsv<float>(x_init, x, w, y, N, Dx, Dy, T, h_args_l, scale_step, cov_step, ll);
}

void smc_ref_forget_fsv(double* x_init, double* x, double* w, double* y, int N, int Dx, int Dy, int T,
        double* h_args_l, double* scale_step, double* cov_step, double* ll) {
    smc_ref_forget_fsv<double>(x_init, x, w, y, N, Dx, Dy, T, h_args_l, scale_step, cov_step, ll);
}

void smc_ref_fsv(float* x_init, float* x, float* w, float* y, int N, int Dx, int Dy,
        int T, float* h_args_l, float* scale_step, float* cov_step, float* ll) {
    smc_ref_fsv<float>(x_init, x, w, y, N, Dx, Dy, T, h_args_l, scale_step, cov_step, ll);
}

void smc_ref_fsv(double* x_init, double* x, double* w, double* y, int N, int Dx, int Dy, int T,
        double* h_args_l, double* scale_step, double* cov_step, double* ll) {
    smc_ref_fsv<double>(x_init, x, w, y, N, Dx, Dy, T, h_args_l, scale_step, cov_step, ll);
}
