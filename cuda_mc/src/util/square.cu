#include "square.h"

__global__ void square_gpu(int size, float* d_data) {
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int tt = blockDim.x * gridDim.x;
	int i;
	for (i = 0; i * tt + tid < size; i++) {
		d_data[i * tt + tid] *= d_data[i * tt + tid];
	}
}

__global__ void square_gpu(int size, float* d_data, float* d_odata) {
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int tt = blockDim.x * gridDim.x;
	int i;
	for (i = 0; i * tt + tid < size; i++) {
		d_odata[i * tt + tid] = d_data[i * tt + tid] * d_data[i * tt + tid];
	}
}

void square(int size, float* d_data, int nb, int nt) {

	square_gpu<<< nb, nt>>>(size, d_data);

	cudaThreadSynchronize();
}

void square(int size, float* d_data, float* d_odata, int nb, int nt) {

	square_gpu<<< nb, nt>>>(size, d_data, d_odata);

	cudaThreadSynchronize();
}

//extern "C"
//void square(int size, float *d_data) {
//	square<float>(size, d_data);
//}
//
//extern "C"
//void square(int size, float *d_data, float* d_odata) {
//	square<float>(size, d_data, d_odata);
//}
