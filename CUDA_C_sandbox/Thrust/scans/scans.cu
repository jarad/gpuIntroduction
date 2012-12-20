#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <iostream>

int main(){
  thrust::device_vector<int> data(6, 0);
  data[0] = 1;
  data[1] = 0;
  data[2] = 2;
  data[3] = 2;
  data[4] = 1;
  data[5] = 3;

  thrust::inclusive_scan(data.begin(), data.end(), data.begin()); // in-place scan
  // data is now {1, 1, 3, 5, 6, 9}

 /* data[0] = data[0]
  * data[1] = data[0] + data[1]
  * data[2] = data[0] + data[1] + data[2]
  * ...
  * data[5] = data[0] + data[1] + ... + data[5]
  */ 

  data[0] = 1;
  data[1] = 0;
  data[2] = 2;
  data[3] = 2;
  data[4] = 1;
  data[5] = 3;
  thrust::exclusive_scan(data.begin(), data.end(), data.end()); // in-place scan

  // data is now {0, 1, 1, 3, 5, 6}
  
 /* data[0] = 0
  * data[1] = data[0]
  * data[2] = data[0] + data[1]
  * ...
  * data[5] = data[0] + data[1] + ... + data[4]
  */ 
}