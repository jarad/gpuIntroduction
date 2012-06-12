/*
 * rng.cu
 *
 *  Created on: 20-Mar-2009
 *      Author: alee
 */

#include "xorshift.h"
#include "MRG.h"
#include <stdio.h>
#include "rng.h"
#include <stdlib.h>

#ifdef RNG_XORSHIFT

void seed_rng(int n_burn, int nb, int nt) {
	seedXS(n_burn, nb, nt);
}
void kill_rng() {
	killXS();
}

void populate_rand(float* array, int N) {
	populate_rand_XS(array, N);
}

void populate_rand_d(float* d_array, int N) {
	populate_rand_XS_d(d_array, N);
}

void populate_randn(float* array, int N) {
	populate_randn_XS(array, N);
}

void populate_randn_d(float* d_array, int N) {
	populate_randn_XS_d(d_array, N);
}

void populate_randIK(int* h_array, int N, int k) {
	populate_randIK_XS(h_array, N, k);
}

void populate_randIK_d(int* d_array, int N, int k) {
	populate_randIK_XS_d(d_array, N, k);
}

void populate_randUI(unsigned int* array, int N) {
    populate_randUI_XS(array, N);
}

void populate_randUI_d(unsigned int* d_array, int N) {
    populate_randUI_XS_d(d_array, N);
}

#endif

#ifdef RNG_XORSHIFT_REF

void seed_rng(int n_burn, int nb, int nt) {
    seed_XS_REF(n_burn);
}

void kill_rng() {

}

void populate_rand(float* array, int N) {
	populate_rand_XS_REF(array, N);
}

void populate_rand_d(float* d_array, int N) {
	populate_rand_XS_REF_d(d_array, N);
}

void populate_randn(float* array, int N) {
	populate_randn_XS_REF(array, N);
}

void populate_randn_d(float* d_array, int N) {
	populate_randn_XS_REF_d(d_array, N);
}

void populate_randIK(int* h_array, int N, int k) {
	populate_randIK_XS_REF(h_array, N, k);
}

void populate_randIK_d(int* d_array, int N, int k) {
	populate_randIK_XS_REF_d(d_array, N, k);
}

void populate_randUI(unsigned int* array, int N) {
    populate_randUI_XS_REF(array, N);
}

void populate_randUI_d(unsigned int* d_array, int N) {
    populate_randUI_XS_REF_d(d_array, N);
}

#endif

#ifdef RNG_MRG

void seed_rng(int n_burn, int nb, int nt) {
    unsigned long seeds[6] = { 12345, 12345, 12345, 12345, 12345, 12345 };
    seed_MRG(nb, nt, seeds);
//    seed_MRG(256, 64, seeds);
}
void kill_rng() {
    kill_MRG();
}

void populate_rand(float* array, int N) {
    populate_rand_MRG(array, N);
}

void populate_rand_d(float* d_array, int N) {
    populate_rand_MRG_d(d_array, N);
}

void populate_randn(float* array, int N) {
    populate_randn_MRG(array, N);
}

void populate_randn_d(float* d_array, int N) {
    populate_randn_MRG_d(d_array, N);
}

void populate_randIK(int* h_array, int N, int k) {
    populate_randIK_MRG(h_array, N, k);
}

void populate_randIK_d(int* d_array, int N, int k) {
    populate_randIK_MRG_d(d_array, N, k);
}

void populate_randUI(unsigned int* array, int N) {
    populate_randUI_MRG(array, N);
}

void populate_randUI_d(unsigned int* d_array, int N) {
    populate_randUI_MRG_d(d_array, N);
}

#endif


void populate_rand(double* array, int N) {
    float* temp = (float*) malloc(N * sizeof(float));
    populate_rand(temp, N);
    for (int i = 0; i < N; i++) {
        array[i] = (double) temp[i];
    }
    free(temp);
}

void populate_randn(double* array, int N) {
    float* temp = (float*) malloc(N * sizeof(float));
    populate_randn(temp, N);
    for (int i = 0; i < N; i++) {
        array[i] = (double) temp[i];
    }
    free(temp);
}
