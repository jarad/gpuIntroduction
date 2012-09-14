#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <cuda_runtime.h> 
#include <cublas_v2.h>

#define N 10 // number of rows (observations)
#define P 3 // number of columns (parameters)

#define RM2A(row, col, rowSize) (row * rowSize + col) // row major index -> linear array index
#define CM2A(col, row, colSize) (col * colSize + row) // column major index -> linear array index

float rfloat(){
  return (2.0 * (float) rand() / (float)RAND_MAX - 1.0) / 4.0;
}  

int main (void){

/* DECLARATIONS */

  cublasHandle_t handle;
  
  int n; // row index (observation index)
  int p; // column index (parameter index)
  
  float alpha = 1; // dummy scalar
  float beta = 1;  // dummy scalar
  
  float* X_h; // N by P design matrix: host 
  float* X_d; // N by P design matrix: device
  
  float* XtX_h; // X^T X, where x is the design matrix: host
  float* XtX_d; // X^T X, where x is the design matrix: device
  
  float* y_h; // N-element observation vector: host
  float* y_d; // N-element observation vector: device
  
  float* b_h; // P-element vector of parameter estimates: host
  float* b_d; // P-element vector of parameter estimates: device

/* MEMORY ALLOCATION */
  
  X_h =   (float *) malloc ( N * P * sizeof (float));  // Xh memory on host
  XtX_h = (float *) malloc ( P * P * sizeof (float)); // XtXh memory on host
  y_h =   (float *) malloc ( N     * sizeof (float));  // yh memory on host
  b_h =   (float *) malloc (     P * sizeof (float));  // bh memory on host
  

  cudaMalloc ((void**) &X_d,   N * P * sizeof(*X_h));   // Xd memory on device 
  cudaMalloc ((void**) &XtX_d, P * P * sizeof(*XtX_h)); // XtXd memory on device 
  cudaMalloc ((void**) &y_d,   N     * sizeof(*y_h));   // yd memory on device
  cudaMalloc ((void**) &b_d,       P * sizeof(*b_h));   // bd memory on device

/* CONSTRUCT INPUTS */

  cublasCreate(&handle); // create CUBLAS runtime context 

  for (p = 0; p <= P - 1; p++) { // fill host design matrix Xh
    for (n = 0; n <= N - 1; n++) { 
      X_h[CM2A(p,n,N)] = (float) pow(n + 1, p); 
    }
  }

 for (p = 0; p <= P - 1; p++) { // fill host matrix XtXh
    for (n = 0; n <= P - 1; n++) { 
      XtX_h[CM2A(p,n,N)] = (float) 0; 
    }
  }

  for(n = 0; n <= N - 1; n++){ // fill host observation vector yh
    y_h[n] = (float) pow(n - 5, 2) + 6.0 + rfloat();
  }

/* SEND INPUTS TO GPU */
  
  cublasStatus_t status;
  
  status = cublasSetMatrix (N, P, sizeof(float), X_h, N, X_d, N);

  if(status != CUBLAS_STATUS_SUCCESS){
    printf("\nnope\n");
  }

  status = cublasSetMatrix (P, P, sizeof(float), XtX_h, P, XtX_d, P);
  
  if(status != CUBLAS_STATUS_SUCCESS){
    printf("\nnope\n");
  }

  
  
  status = cublasSetVector (N, sizeof(float), y_h, 1, y_d, 1);
  
  if(status != CUBLAS_STATUS_SUCCESS){
    printf("\nnope\n");
  }

  
  
  
  

/* COMPUTE BETA = (X'X)^(-1) X' Y */

  cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, // X'X -> XtX_d
              P, N, &alpha, X_d, N, &beta, XtX_d, P);
              
  cublasStrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, // ((X'X)^(-1) X')' -> X_d
              CUBLAS_DIAG_NON_UNIT, N, P, &alpha, XtX_d, P, X_d, N);
 
 
 
  
/* COPY RESULTS BACK TO CPU */
  
  cublasGetMatrix(P, P, sizeof(float),
                  XtX_d, P, XtX_h, P);

  cublasGetMatrix(N, P, sizeof(float),
                  X_d, N, X_h, N);

/* PRINT RESULTS */

  printf("X^T X:\n");
  for (n = 0; n <= P - 1; n++) {
    for (p = 0; p <= P - 1; p++) {
      printf ("%7.3f", XtX_h[CM2A(p,n,P)]);
    }
    printf ( "\n" );
  }
  printf("\n");

  printf("\nDesign matrix:\n");
  for (n = 0; n <= N - 1; n++) {
    for (p = 0; p <= P - 1; p++) {
      printf ("%7.3f", X_h[CM2A(p,n,N)]);
    }
    printf ( "\n" );
  }

  printf("\nObservation Vector:\n");
  for(n = 0; n <= N - 1; n++){
    printf("%7.3f\n", y_h[n]);
  }
  printf("\n");

/*
  cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);

  //modify (handle, devPtrA, M, N, 2, 3, 16.0f, 12.0f);
  
  cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);

  cudaFree ( devPtrA );

*/
  
  cublasDestroy ( handle );
}