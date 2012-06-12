/*
 * output.h
 *
 *  Created on: 27-Feb-2009
 *      Author: alee
 */

#ifndef OUTPUT_H_
#define OUTPUT_H_

void to_file(float* array, int N, char* filename);

void to_file(unsigned int* array, int N, char* filename);

//void to_file_T(float* array, int N);

void to_file(float* array, int N, int D, char* filename);

void to_file(double* array, int N, int D, char* filename);

void to_file(double* array, int N, char* filename);



#endif /* OUTPUT_H_ */
