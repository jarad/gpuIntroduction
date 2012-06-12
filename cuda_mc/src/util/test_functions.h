/*
 * test_functions.h
 *
 *  Created on: 23-Jan-2009
 *      Author: alee
 */

#ifndef TEST_FUNCTIONS_H_
#define TEST_FUNCTIONS_H_

//#define PHI_IDENTITY 1
//#define PHI_SQUARED 2

void add(int size, float* d_array, float* d_output, float to_add, int nb, int nt);

void add(int size, float* d_array, float* d_output, float* d_to_add, int nb, int nt);

void multiply(int size, float* d_array, float* d_output, float to_multiply, int nb, int nt);

void multiply(int size, float* d_array, float* d_output, float* d_to_multiply, int nb, int nt);

void multiply(int size, int D, float* d_array, float* d_output, float* d_to_multiply, int nb, int nt);

void exp(int size, float* d_array, float* d_output, int nb, int nt);

void multiply_matrix(int size, int D, float* d_array, float* d_output,
		float* d_to_multiply, int nb, int nt);

void set(int size, float* d_array, float value, int nb, int nt);

#endif /* TEST_FUNCTIONS_H_ */
