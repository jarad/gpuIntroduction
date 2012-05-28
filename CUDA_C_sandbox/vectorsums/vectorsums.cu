#define N 1000000000
#include <stdio.h>
#include <stdlib.h>

__global__ void add(int *a, int *b, int *c){
  int tid = blockIdx.x;
  if(tid < N)
    c[tid] = a[tid] + b[tid];
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

  printf("Adding...");
  add<<<N,1>>>(dev_a, dev_b, dev_c);
  printf("Done.\n");

  printf("Clearing memory...");
  cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

  // printf("\ni =  : \t a[i] \t + \t b[i] \t = \t c[i] \n \n");
  // for(i = 0; i<N; i++){
  //   printf("i = %i: \t %d \t + \t %d \t = \t %d \n", i, a[i], b[i], c[i]);
  // }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  printf("Done.\n");

  return 0;
}