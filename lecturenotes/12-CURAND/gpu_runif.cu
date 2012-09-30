#include <curand_kernel.h>
#include "cutil_inline.h"

#define THREADS_PER_BLOCK 256

__global__ void setup_prng(unsigned long long seed, curandState *state)
{
    int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void runif_kernel(curandState *state, double ub, int ni, int nd, 
                             double *uniforms, int *counts)
{
    int i, a, count, id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    double b, u;

    // Copy state to local memory for efficiency */
    curandState localState = state[id];

    // Find random uniform below the upper bound 
    count  = -1;
    u = ub+1;
    while ( u>ub ) 
    {
        count++;
        u = curand_uniform_double(&localState);

        // Computational overhead
        a=0; for (i=0; i<ni; i++) a += 1; 
        b=1; for (i=0; i<nd; i++) b *= 1.00001;
    }

    // Copy state back to global memory */
    state[id] = localState ;

    // Store results */
    uniforms[id] = u;
    counts[id] = count;
}




//CURAND_RNG_PSEUDO_MTGP32

extern "C" {

void gpu_runif(int *n, double *ub, int *ni, int *nd, double *seed, double *u, int *c) 
{
    int nBlocks = *n/THREADS_PER_BLOCK, *d_c;
    size_t u_size = *n *sizeof(double), c_size = *n *sizeof(int);
    double *d_u;

    cutilSafeCall( cudaMalloc((void**)&d_u,  u_size) );
    cutilSafeCall( cudaMalloc((void**)&d_c,  c_size) );

    // Setup prng states
    curandState *d_states;
    cutilSafeCall( cudaMalloc((void**)&d_states, nBlocks*THREADS_PER_BLOCK*sizeof(curandState)) );
    setup_prng<<<nBlocks,THREADS_PER_BLOCK>>>(*seed, d_states);

    runif_kernel<<<nBlocks,THREADS_PER_BLOCK>>>(d_states, *ub, *ni, *nd, d_u, d_c);
 
    cutilSafeCall( cudaMemcpy(u,   d_u,  u_size, cudaMemcpyDeviceToHost) );
    cutilSafeCall( cudaMemcpy(c,   d_c,  c_size, cudaMemcpyDeviceToHost) );

    cutilSafeCall( cudaFree(d_u)      );
    cutilSafeCall( cudaFree(d_c)      );
    cutilSafeCall( cudaFree(d_states) );
}

} // end of extern "C"
