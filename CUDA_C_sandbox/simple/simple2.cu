#include <stdio.h>
#include <stdlib.h>

__global__ void colonel(int *dev_b, int *dev_a){
  *dev_b = *dev_a + 1;
}

int main(){

  // Declare variables and allocate memory on the GPU.
  int a[1], b[1], *dev_a, *dev_b;
  cudaMalloc((void**) &dev_a, sizeof(int));
  cudaMalloc((void**) &dev_b, sizeof(int));

  // Intitialize argument a, executed kernel, and store result in b.
  a[0] = 1; // REMEMBER: INDEXING IN C STARTS FROM 0.
  cudaMemcpy(dev_a, a, sizeof(int), cudaMemcpyHostToDevice); // AFTER THIS LINE, *dev_a CANNOT BE CHANGED!
  colonel<<<1,1>>>(dev_b, dev_a);
  cudaMemcpy(b, dev_b, sizeof(int), cudaMemcpyDeviceToHost);

  // Print result and free dynamically allocated memory.
  printf("b[0] = %d\n", b[0]); // REMEMBER: INDEXING IN C STARTS FROM 0.
  cudaFree(dev_a);
  cudaFree(dev_b);

}