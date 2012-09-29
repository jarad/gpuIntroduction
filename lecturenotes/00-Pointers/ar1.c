#include <stdio.h>

void modify(int *a){
  int i;
  for(i = 0; i<6; ++i){
    a[i] = i;
  }
}

int main(){
  int i;
  int a[] = {1,23,17,4,-5,100};
  
  for(i = 0; i<6; ++i){
    printf("a[%d] = %d\n", i, a[i]);
  }
  printf("\nModifying...\n\n");
  
  modify(a);
  
  for(i = 0; i<6; ++i){
    printf("a[%d] = %d\n", i, a[i]);
  }
}
