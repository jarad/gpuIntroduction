/*
 * rng_shared.cu
 *
 *  Created on: 23-Mar-2009
 *      Author: alee
 */

#include "func.h"

__device__ void BoxMuller(float& u1, float& u2) {
    float r = sqrtf(-2.0f * logf(u1));
    float phi = 2 * PI * u2;
    u1 = r * cosf(phi);
    u2 = r * sinf(phi);
}

__global__ void BoxMullerGPU(float *d_Random, int N) {
    const int tt = blockDim.x * gridDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < N; i += 2 * tt) {
        BoxMuller(d_Random[i], d_Random[i + tt]);
    }

}
