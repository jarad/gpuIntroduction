//Example 1. Application Using C and CUBLAS: 1−based indexing 
//−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−
#include <stdlib.h> 
#include <math.h> 
#include <cuda_runtime.h> 
#include "cublas_v2.h"
#define M 6

static __inline__ void modify (cublasHandle_t handle, float ∗m, int ldm, int n, int p,
  int q, float alpha, float beta){
  cublasSscal (handle, n-p+1, &alpha, &m[IDX2F(p,q,ldm)], ldm);
  cublasSscal (handle, ldm−p+1, &beta, &m[IDX2F(p,q,ldm)], 1); 
}

int main (void){
  if (!a) {
  }
    }

  cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a)); 
  if ( cudaStat != cudaSuccess ) {
  }
  }

  stat = cublasSetMatrix (M, N, sizeof(∗a), a, M, devPtrA, M);
    printf("data download failed");
    cudaFree(devPtrA);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  modify (handle, devPtrA, M, N, 2, 3, 16.0f, 12.0f);
  stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
  if( stat != CUBLAS_STATUS_SUCCESS ) {
    printf ("data upload failed");
    cudaFree ( devPtrA );
    cublasDestroy ( handle );
    return EXIT_FAILURE;
  }

  cudaFree ( devPtrA );
  cublasDestroy ( handle );

  for (j = 1; j <= N; j++) {
  }
}