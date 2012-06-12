/*
 * rng_shared_ref.cpp
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#include <math.h>
#include "func.h"
#include "rng_shared.h"

void BoxMuller_REF(float& u1, float& u2) {
    float r = sqrtf(-2.0f * logf(u1));
    float phi = 2 * PI * u2;
    u1 = r * cosf(phi);
    u2 = r * sinf(phi);
}
