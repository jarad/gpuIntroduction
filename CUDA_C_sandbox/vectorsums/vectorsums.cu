#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h> 

#define N 10

__global__ void add(int *a, int *b, int *c){
  int bid = blockIdx.x;
  if(bid < N)
    c[bid] = a[bid] + b[bid];
}

int main(void) {
  int i, a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;

  cudaMalloc((void**) &dev_a, N*sizeof(int));
  cudaMalloc((void**) &dev_b, N*sizeof(int));
  cudaMalloc((void**) &dev_c, N*sizeof(int));

  for(i=0; i<N; i++){
    a[i] = -i;
    b[i] = i*i;
  }

  cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

  add<<<N,1>>>(dev_a, dev_b, dev_c);

  cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

  printf("\na + b = c\n");
  for(i = 0; i<N; i++){
    printf("%5d + %5d = %5d\n", a[i], b[i], c[i]);
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}