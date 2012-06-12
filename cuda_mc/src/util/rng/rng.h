/*
 * rng.h
 *
 *  Created on: 20-Mar-2009
 *      Author: alee
 */

#ifndef RNG_H_
#define RNG_H_

void seed_rng(int n_burn = 32768, int nb = 32, int nt = 128);
void kill_rng();

void populate_rand(float* array, int N);
void populate_rand_d(float* d_array, int N);

void populate_randn(float* array, int N);
void populate_randn_d(float* d_array, int N);

// ints from 0 (inclusive) to k (exclusive)
void populate_randIK(int* h_array, int N, int k);
void populate_randIK_d(int* d_array, int N, int k);

void populate_randUI(unsigned int* array, int N);

void populate_randUI_d(unsigned int* d_array, int N);

void populate_rand(double* array, int N);
void populate_randn(double* array, int N);

#endif /* RNG_H_ */
