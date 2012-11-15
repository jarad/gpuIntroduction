#include <thrust/partition.h> 

struct is_even{
  __host__ __device__ bool operator()(const int x){
    return (x % 2) == 0;
  }
};

int main(){
  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(A)/sizeof(int);
  thrust::partition(A, A + N,
                  is_even());
                     
  // A is now {2, 4, 6, 8, 10, 1, 3, 5, 7, 9}
  int i;
  for(i = 0; i < A.size(); ++i)
    std::cout << "A[" << i << "] = " << A[i] << std::endl;
  return 0;
}