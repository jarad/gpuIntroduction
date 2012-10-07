#include <thrust/scan.h>
#include <iostream>

int main(){
  int data[6] = {1, 0, 2, 2, 1, 3};
  thrust::inclusive_scan(data, data + 6, data); // in-place scan
  // data is now {1, 1, 3, 5, 6, 9}

 /* data[0] = data[0]
  * data[1] = data[0] + data[1]
  * data[2] = data[0] + data[1] + data[2]
  * ...
  * data[5] = data[0] + data[1] + ... + data[5]
  */ 
  
  int data2[6] = {1, 0, 2, 2, 1, 3};
  thrust::exclusive_scan(data2, data2 + 6, data2); // in-place scan

  // data2 is now {0, 1, 1, 3, 5, 6}
  
 /* data2[0] = 0
  * data2[1] = data2[0]
  * data2[2] = data2[0] + data2[1]
  * ...
  * data2[5] = data2[0] + data2[1] + ... + data[4]
  */ 
}