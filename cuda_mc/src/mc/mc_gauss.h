/*
 * mc_gauss.h
 *
 *  Created on: 02-Feb-2009
 *      Author: alee
 */

#ifndef MC_GAUSS_H_
#define MC_GAUSS_H_

void is_nn(int size, float* d_array, float* d_warray, float* h_args_p, float* h_args_q, int nb, int nt);

void is_ref_nn(int size, float* array, float* warray, float* h_args_p, float* h_args_q);

#endif /* MC_GAUSS_H_ */
