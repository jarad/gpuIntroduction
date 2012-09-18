#include <cula.h>

int main(){

  culaStatus s;

  s = culaInitialize();
  if(s != culaNoError)
  {
    printf("%s\n", culaGetErrorString(s));
  }

  /* ... Your code ... */

  culaShutdown();
}