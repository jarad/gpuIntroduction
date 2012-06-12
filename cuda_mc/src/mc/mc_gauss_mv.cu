/*
 * mc_gauss_mv.cu
 *
 *  Created on: 18-Feb-2009
 *      Author: alee
 */

#include "func.h"
#include "gauss.ch"

#define PROPOSAL gauss_pdf
#define TARGET gauss_pdf
#define LOG_PROPOSAL log_gauss_pdf
#define LOG_TARGET log_gauss_pdf
#define TYPE nn_mv
// max D is 10
#define NUM_AP 111 // need D*D + D + 1
#define NUM_AQ 111

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
