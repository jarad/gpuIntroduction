#include <stdio.h>
#include <stdlib.h>

__global__ void colonel(int *dev_a){
  *dev_a = 1;
}

int main(){

  // Declare variables and allocate memory on the GPU.
  int a[1], *dev_a;
  cudaMalloc((void**) &dev_a, sizeof(int));

  // Execute kernel and copy the result to CPU memory.
  colonel<<<1,1>>>(dev_a);
  cudaMemcpy(a, dev_a, sizeof(int), cudaMemcpyDeviceToHost);

  // Print result and free dynamically allocated memory.
  printf("a[0] = %d\n", a[0]); // REMEMBER: INDEXING IN C STARTS FROM 0.
  cudaFree(dev_a);

}