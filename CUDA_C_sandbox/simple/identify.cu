#include <stdio.h>
#include <stdlib.h>

__global__ void isExecuted(int *a_d, int blockid, int threadid){
  if(blockIdx.x == blockid && threadIdx.x == threadid)
    *a_d = 1;
}

int main(){

  int a = 0, *a_d;
  
  cudaMalloc((void**) &a_d, sizeof(int));
  cudaMemcpy(a_d, &a, sizeof(int), cudaMemcpyHostToDevice);

  isExecuted<<<100,100>>>(a_d, 2, 4); 
  
  cudaMemcpy(&a, a_d, sizeof(int), cudaMemcpyDeviceToHost);

  printf("a = %d\n", a);
  cudaFree(a_d);

}