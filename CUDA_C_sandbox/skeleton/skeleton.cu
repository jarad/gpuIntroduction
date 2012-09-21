#include <stdio.h> 
#include <stdlib.h> 
#include <cuda.h>
#include <cuda_runtime.h> 

int main (void){ 
  // Declare all variables.
  ...
  
  // Dynamically allocate host memory.
  ...
  
  // Dynamically allocate device memory.
  ...
  
  // Write to host memory.
  ...
  
  // Copy host memory to device memory.
  ...
  
  // Execute kernel on the device.
  ...
  
  // Write device memory back to host memory.
  ...
  
  // Free dynamically-allocated host memory
  ...

  // Free dynamically-allocated device memory    
  ...
}