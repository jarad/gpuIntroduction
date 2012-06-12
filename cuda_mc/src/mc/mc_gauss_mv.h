/*
 * mc_gauss_mv.h
 *
 *  Created on: 18-Feb-2009
 *      Author: alee
 */

#ifndef MC_GAUSS_MV_H_
#define MC_GAUSS_MV_H_

void is_nn_mv(int size, int D, float* d_array, float* d_warray, float* h_args_p, float* h_args_q,
        int log, int nb, int nt);

void is_ref_nn_mv(int size, int D, float* array, float* warray, float* args_p, float* args_q, int log);

#endif /* MC_GAUSS_MV_H_ */
