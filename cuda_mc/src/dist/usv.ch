/*
 * usv.ch
 *
 *  Created on: 09-Jul-2009
 *      Author: alee
 */

#include "gauss.ch"

__device__ float usv_pdf(float x, float y, float* args) {
    float beta = args[0];

    return gauss1_pdf(y, 0.0f, beta * exp(x / 2));
}
