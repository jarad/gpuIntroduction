#ifndef __REDUCE_H__
#define __REDUCE_H__

//#define N_THREADS 128
//#define N_BLOCKS 32

void reduce(int size, float *d_idata, float& h_odata, int nb, int nt);

//void reduce2(int size, float *d_idata, float *h_odata, int nb, int nt);

//void reduce_ss(int size, float *d_idata, float *h_odata, int nb, int nt);

void reduce(int size, int D, float* d_idata, float* h_odata, int nb, int nt);

void reduce_ss(int size, float* d_idata, float& h_sum, int nb, int nt);

//void reduce(int N, int D, int skip, float* d_idata, float* h_odata, int nb, int nt);

#endif
