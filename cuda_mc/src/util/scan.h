/*
 * scan.h
 *
 *  Created on: 02-Mar-2009
 *      Author: alee
 */

#ifndef SCAN_H_
#define SCAN_H_

void scan(int size, float *d_idata, float* d_odata, int nb, int nt);
void scan_log(int size, float *d_idata, float* d_odata, int nb, int nt);

void scan_init(int N);
void scan_destroy();

template <class T>
void scan_ref(int N, T *h_idata, T* h_odata) {
    double sum = 0;
    for (int i = 0; i < N; i++) {
        sum += h_idata[i];
        h_odata[i] = (T) sum;
    }
}

#endif /* SCAN_H_ */
