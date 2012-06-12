/*
 * is_ref.cu
 *
 *  Created on: 22-Mar-2009
 *      Author: alee
 */

void FUNC( is_ref, TYPE)(
int size, float* array, float* warray, float* h_args_p, float* h_args_q) {

	int i;
	float p, q, x;
	for (i = 0; i < size; i++) {
		x = array[i];
		p = TARGET_H(x, h_args_p);
		q = PROPOSAL_H(x, h_args_q);
		warray[i] = p / q;
	}

}
