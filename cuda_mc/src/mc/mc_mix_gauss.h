/*
 * mc_mix_gauss.h
 *
 *  Created on: 04-Feb-2009
 *      Author: alee
 */

#ifndef MC_MIX_GAUSS_H_
#define MC_MIX_GAUSS_H_

void is_nmni(int size, float* d_array, float* d_warray, float* h_args_p, float* h_args_q, int nb, int nt);

void is_ref_nmni(int size, float* array, float* warray, float* args_p, float* args_q);

#endif /* MC_MIX_GAUSS_H_ */
