/*
 * temper.inl
 *
 *  Created on: 11-Feb-2009
 *      Author: alee
 */

__device__ float temper(float d, float t) {
	return powf(d, t);
}
