/*
 * usv.h
 *
 *  Created on: 09-Jul-2009
 *      Author: alee
 */

#ifndef USV_H_
#define USV_H_

#include "gauss.h"

template <class T>
void generate_data_usv(T* xs, T* ys, int total_time, T alpha, T sigma, T beta) {

    T* steps_x = (T*) malloc(total_time * sizeof(T));
    T* steps_y = (T*) malloc(total_time * sizeof(T));

    populate_randn(steps_x, total_time);
    populate_randn(steps_y, total_time);

    for (int i = 0; i < total_time; i++) {
        steps_x[i] *= sigma;
        steps_y[i] *= beta;
    }

    xs[0] = steps_x[0];
    ys[0] = ((T) exp(xs[0]/2.0)) * steps_y[0];

    for (int i = 1; i < total_time; i++) {
        xs[i] = alpha * xs[i-1] + steps_x[i];
        ys[i] = (float) (exp(xs[i]/2.0) * steps_y[i]);
    }

    free(steps_x);
    free(steps_y);
}

template <class T>
T usv_pdfh(T x, T y, T* args) {
    T beta = args[0];

    return gauss1_pdfh(y, (T) 0.0, (T) (beta * exp(x / 2)));
}

#endif /* USV_H_ */
