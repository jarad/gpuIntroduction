#include <stdio.h>
#include <stdlib.h>

__global__ void isExecuted(int *dev_a, int blockid, int threadid){
  
  if(blockIdx.x == blockid && threadIdx.x == threadid)
    *dev_a = 1;
  else
    *dev_a = 0;

}

int main(){

  // Declare variables and allocate memory on the GPU.
  int a[1], *dev_a;
  cudaMalloc((void**) &dev_a, sizeof(int));

  // Execute kernel and copy the result to CPU memory.
  isExecuted<<<100,100>>>(dev_a, 2, 4); // NOTE: INDEXING OF THREADS AND BLOCKS STARTS FROM 0.
  cudaMemcpy(a, dev_a, sizeof(int), cudaMemcpyDeviceToHost);

  // Print result and free dynamically allocated memory.
  printf("a[0] = %d\n", a[0]); // REMEMBER: INDEXING IN C STARTS FROM 0.
  cudaFree(dev_a);

}