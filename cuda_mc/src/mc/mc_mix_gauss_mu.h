/*
 * mc_mix_gauss_mu.h
 *
 *  Created on: 25-Mar-2009
 *      Author: alee
 */

#ifndef MC_MIX_GAUSS_MU_H_
#define MC_MIX_GAUSS_MU_H_

void is_mgmu_mv(int size, int D, float* d_array, float* d_warray, float* h_args_p, float* h_args_q,
        int log, int nb, int nt);

void is_ref_mgmu_mv(int size, int D, float* d_array, float* d_warray, float* h_args_p, float* h_args_q, int log);

#endif /* MC_MIX_GAUSS_MU_H_ */
