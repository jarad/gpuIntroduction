/*
 * output.c
 *
 *  Created on: 27-Feb-2009
 *      Author: alee
 */

#include <stdio.h>
#include "output.h"

void to_file(float* array, int N, char* filename) {
    FILE* ofp;
    ofp = fopen(filename, "w");
    int i;
    for (i = 0; i < N; i++) {
        fprintf(ofp, "%f\n", array[i]);
    }
    fclose(ofp);
}

void to_file(unsigned int* array, int N, char* filename) {
    FILE* ofp;
    ofp = fopen(filename, "w");
    int i;
    for (i = 0; i < N; i+=10) {
        for (int j = 0; j < 10; j++) {
            fprintf(ofp, "%08x", array[i+j]);
        }
        fprintf(ofp, "\r\n");
    }
    fclose(ofp);
}

//void to_file(float* array, int N) {
//	FILE* ofp;
//	ofp = fopen("temp_out.txt", "w");
//	int i;
//	for (i = 0; i < N; i++) {
//		fprintf(ofp, "%f ", array[i]);
//	}
//	fclose(ofp);
//}


void to_file(float* array, int N, int D, char* filename) {
    FILE* ofp;
    ofp = fopen(filename, "w");
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < D; j++) {
            fprintf(ofp, "%f ", array[i * D + j]);
        }
        fprintf(ofp, "\n");
    }
    fclose(ofp);
}

void to_file(double* array, int N, char* filename) {
    FILE* ofp;
    ofp = fopen(filename, "w");
    int i;
    for (i = 0; i < N; i++) {
        fprintf(ofp, "%f\n", array[i]);
    }
    fclose(ofp);
}

void to_file(double* array, int N, int D, char* filename) {
    FILE* ofp;
    ofp = fopen(filename, "w");
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < D; j++) {
            fprintf(ofp, "%f ", array[i * D + j]);
        }
        fprintf(ofp, "\n");
    }
    fclose(ofp);
}
