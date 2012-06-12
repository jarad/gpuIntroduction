/*
 * uniform.h
 *
 *  Created on: 16-Mar-2009
 *      Author: Owner
 */

#ifndef UNIFORM_H_
#define UNIFORM_H_

template <class T>
inline T log_uniform_pdfh(T* x, T* args, int D) {
    T a = args[0];
    T b = args[1];
    for (int i = 0; i < D; i++) {
        if (x[i] < a || x[i] > b) {
            return -FLT_MAX;
        }
    }
    return 0;
}

template <class T>
inline T uniform_pdfh(T* x, T* args, int D) {
    T a = args[0];
    T b = args[1];
    for (int i = 0; i < D; i++) {
        if (x[i] < a || x[i] > b) {
            return 0;
        }
    }
    return 1;
}

#endif /* UNIFORM_H_ */
