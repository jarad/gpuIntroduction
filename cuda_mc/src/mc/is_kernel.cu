/*
 * is_kernel.cu
 *
 *  Created on: 03-Feb-2009
 *      Author: alee
 */

__constant__ float args_p[NUM_AP];
__constant__ float args_q[NUM_AQ];

__global__ void FUNC( is, TYPE)(
int size, float* d_array, float* d_warray) {
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int tt = blockDim.x * gridDim.x;
	int i;
	float p, q, x;
	for (i = tid; i < size; i += tt) {
		x = d_array[i];
		p = TARGET(x, args_p);
		q = PROPOSAL(x, args_q);
		d_warray[i] = p / q;
	}
}

void FUNC( is, TYPE)(
int size, float* d_array, float* d_warray, float* h_args_p, float* h_args_q, int nb, int nt) {
	cudaMemcpyToSymbol(args_p, h_args_p, NUM_AP * sizeof(float));
	cudaMemcpyToSymbol(args_q, h_args_q, NUM_AQ * sizeof(float));
	FUNC(is, TYPE)<<<nb,nt>>>(size, d_array, d_warray);
}
