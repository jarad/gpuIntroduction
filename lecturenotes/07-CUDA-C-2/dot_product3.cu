  // for reductions, threadsPerBlock must be a power of 2 // because of the following code
  int i = blockDim.x/2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();    i /= 2; 
  }
