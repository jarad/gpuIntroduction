#define N 1000000000
#include <stdio.h>
#include <stdlib.h>


void add(int *a, int *b, int *c){
  int i;
  for(i=0;i < N; i++)
    c[i] = a[i] + b[i];
}


int main(void) {
  int i, a[N], b[N], c[N];

  for(i=0; i<N; i++){
    a[i] = -i;
    b[i] = i*i;
  }

  printf("Adding...");
  add(a, b, c);
  //  printf("\ni =  : \t a[i] \t + \t b[i] \t = \t c[i] \n \n");
  //for(i = 0; i<N; i++){
  //  printf("i = %i: \t %d \t + \t %d \t = \t %d \n", i, a[i], b[i], c[i]);
  //}
  //printf("\n");
  printf("Done.\n");

  return 0;
}
