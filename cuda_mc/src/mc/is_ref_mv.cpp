/*
 * is_ref_mv.cu
 *
 *  Created on: 22-Mar-2009
 *      Author: alee
 */

void FUNC(is_ref, TYPE)(
int size, int D, float* array, float* warray, float* h_args_p, float* h_args_q, int log) {
    int i;
    float p, q;
    for (i = 0; i < size; i++) {
        float* x = array + i * D;
        if (log) {
            p = LOG_TARGET_H(x, h_args_p, D);
            q = LOG_PROPOSAL_H(x, h_args_q, D);
            warray[i] = p - q;
        } else {
            p = TARGET_H(x, h_args_p, D);
            q = PROPOSAL_H(x, h_args_q, D);
            warray[i] = p / q;
        }

    }
}
