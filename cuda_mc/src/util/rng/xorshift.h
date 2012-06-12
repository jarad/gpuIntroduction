/*
 * xorshift.h
 *
 *  Created on: 14-Jan-2009
 *      Author: alee
 */

#ifndef XORSHIFT_H_
#define XORSHIFT_H_

void seedXS(int n_burn = 1073741824, int nb = 32, int nt = 128);
void seed_XS_REF(int N);
void killXS();

void populate_rand_XS(float* array, int N);
void populate_randn_XS(float* array, int N);

void populate_rand_XS_REF(float* array, int N);
void populate_randn_XS_REF(float* array, int N);
void populate_randIK_XS_REF(int* array, int N, int k);
void populate_rand_XS_REF_d(float* array, int N);
void populate_randn_XS_REF_d(float* array, int N);
void populate_randIK_XS_REF_d(int* array, int N, int k);

void populate_randIK_XS(int* h_array, int N, int k);
void populate_randIK_XS_d(int* d_array, int N, int k);

void populate_rand_XS_d(float* d_array, int N);
void populate_randn_XS_d(float* d_array, int N);

void populate_randUI_XS(unsigned int* array, int N);
void populate_randUI_XS_d(unsigned int* d_array, int N);
void populate_randUI_XS_REF(unsigned int* array, int N);
void populate_randUI_XS_REF_d(unsigned int* d_array, int N);

#endif /* XORSHIFT_H_ */
