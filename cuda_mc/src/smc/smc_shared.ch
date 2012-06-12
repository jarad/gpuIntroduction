/*
 * smc_shared.ch
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#ifndef SMC_SHARED_CH_
#define SMC_SHARED_CH_

__global__ void resample_get_indices(float* cumw, int N, float* randu,
        int* indices, float sumw);

__global__ void resample(float* x, int N, int* indices, float* xt_copy);

__global__ void resample(float* x, int N, int D, int* indices,
        float* xt_copy);

__global__ void historify(float* x, int N, int D, int T, int* history, float* xcopy);

__global__ void historify(float* x, int N, int T, int* history, float* xcopy);

__global__ void history_identity(int* history, int N);

texture<float, 1, cudaReadModeElementType> tex_cw;

#endif /* SMC_SHARED_CH_ */
