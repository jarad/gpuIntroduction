/*
 * MRG.h
 *
 *  Created on: 24-Mar-2009
 *      Author: Owner
 */

#ifndef MRG_H_
#define MRG_H_

void seed_MRG(int nb, int nt, unsigned long* seeds);
void seed_MRG_ref(unsigned long* seeds);
void kill_MRG();

void populate_rand_MRG_REF(float* array, int N);
void populate_randn_MRG_REF(float* array, int N);
void populate_randUI_MRG_REF(unsigned int* array, int N);

void populate_rand_MRG(float* array, int N);
void populate_randn_MRG(float* array, int N);
void populate_randIK_MRG(int* h_array, int N, int k);
void populate_rand_MRG_d(float* d_array, int N);
void populate_randn_MRG_d(float* d_array, int N);
void populate_randIK_MRG_d(int* d_array, int N, int k);

void populate_randUI_MRG(unsigned int* array, int N);

void populate_randUI_MRG_d(unsigned int* d_array, int N);


#endif /* MRG_H_ */
