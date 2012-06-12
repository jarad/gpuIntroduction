/*
 * uniform.ch
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#ifndef UNIFORM_CH_
#define UNIFORM_CH_

#include <float.h>

template <int D>
__device__ float log_uniform_pdf(float* x, float* args) {
    float a = args[0];
    float b = args[1];
    for (int i = 0; i < D; i++) {
        if (x[i] < a || x[i] > b) {
            return -FLT_MAX;
        }
    }
    return 0;
}

template <int D>
__device__ float uniform_pdf(float* x, float* args) {
    float a = args[0];
    float b = args[1];
    for (int i = 0; i < D; i++) {
        if (x[i] < a || x[i] > b) {
            return 0;
        }
    }
    return 1;
}

#endif /* UNIFORM_CH_ */
