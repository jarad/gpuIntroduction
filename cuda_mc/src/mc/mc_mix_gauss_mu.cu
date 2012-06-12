/*
 * mc_mix_gauss_mu.cu
 *
 *  Created on: 25-Mar-2009
 *      Author: alee
 */

#include "func.h"
#include "mix_gauss_uniform.ch"
#include "uniform.ch"

#define PROPOSAL uniform_pdf
#define TARGET mgu_mu_pdf
#define LOG_PROPOSAL log_uniform_pdf
#define LOG_TARGET log_mgu_mu_pdf
#define TYPE mgmu_mv
// max D is 10
#define NUM_AP 105 // need D*D + D + 1
#define NUM_AQ 2

#include "is_kernel_mv.cu"

void FUNC( is, TYPE)(int size, int D, float* d_array, float* d_warray,
        float* h_args_p, float* h_args_q, int log, int nb, int nt) {
    switch (D) {
    case 1:
        FUNC( is, TYPE) < 1 > (size, d_array, d_warray, h_args_p, h_args_q, log, nb, nt);
        break;
    case 2:
        FUNC( is, TYPE) < 2 > (size, d_array, d_warray, h_args_p, h_args_q, log, nb, nt);
        break;
    case 3:
        FUNC( is, TYPE) < 3 > (size, d_array, d_warray, h_args_p, h_args_q, log, nb, nt);
        break;
    case 4:
        FUNC( is, TYPE) < 4 > (size, d_array, d_warray, h_args_p, h_args_q, log, nb, nt);
        break;
    default:
        break;
    }
}
