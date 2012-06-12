/*
 * is_kernel_mv.cu
 *
 *  Created on: 18-Feb-2009
 *      Author: alee
 */

#include "matrix.ch"

__constant__ float args_p[NUM_AP];
__constant__ float args_q[NUM_AQ];

template <int D>
__global__ void FUNC(is_gpu, TYPE)(int size, float* d_array, float* d_warray, int log) {
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int tt = blockDim.x * gridDim.x;
	int i;
	float p, q;
	float* x;
	for (i = tid; i < size; i += tt) {
		x = d_vector_get(d_array, D, i);
		if (log) {
		    p = LOG_TARGET<D>(x, args_p);
		    q = LOG_PROPOSAL<D>(x, args_q);
		    d_warray[i] = p - q;
		} else {
		    p = TARGET<D>(x, args_p);
            q = PROPOSAL<D>(x, args_q);
            d_warray[i] = p / q;
		}
	}
}

template <int D>
void FUNC( is, TYPE)(
int size, float* d_array, float* d_warray, float* h_args_p, float* h_args_q, int log, int nb, int nt) {
	cudaMemcpyToSymbol(args_p, h_args_p, NUM_AP * sizeof(float));
	cudaMemcpyToSymbol(args_q, h_args_q, NUM_AQ * sizeof(float));
	FUNC(is_gpu, TYPE)<D><<<nb,nt>>>(size, d_array, d_warray, log);

}
