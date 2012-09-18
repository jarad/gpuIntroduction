#include "/usr/local/cula/include/cula.h"
#include <stdlib.h>
#include <stdio.h>

int main(){

  culaStatus s;

  s = culaInitialize();

  if(s != culaNoError)
  {
    printf("%s\n", culaGetErrorString(s));
    /* ... Error Handling ... */
  }

  /* ... Your code ... */

  culaShutDown();

}