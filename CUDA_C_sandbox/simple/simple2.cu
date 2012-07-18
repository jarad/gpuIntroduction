#include <stdio.h>
#include <stdlib.h>

__global__ void colonel(int *dev_a){
  *dev_a = *dev_a + 1;
}

int main(){

  // Declare variables and allocate memory on the GPU.
  int a[1], *dev_a;
  cudaMalloc((void**) &dev_a, sizeof(int));

  // Intitialize argument a, executed kernel, and store result back in a.
  a[0] = 1; // REMEMBER: INDEXING IN C STARTS FROM 0.
  cudaMemcpy(dev_a, a, sizeof(int), cudaMemcpyHostToDevice); 
  colonel<<<1,1>>>(dev_a);
  cudaMemcpy(a, dev_a, sizeof(int), cudaMemcpyDeviceToHost);

  // Print result and free dynamically allocated memory.
  printf("a[0] = %d\n", a[0]); // REMEMBER: INDEXING IN C STARTS FROM 0.
  cudaFree(dev_a);

}