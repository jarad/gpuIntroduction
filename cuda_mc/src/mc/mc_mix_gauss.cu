/*
 * mc_mix_gauss.cu
 *
 *  Created on: 02-Feb-2009
 *      Author: alee
 */

#include "mc_mix_gauss.h"

#include "func.h"
#include "mix_gauss.ch"

#define PROPOSAL gauss1_pdf
#define TARGET mix_gauss1_pdf
#define TYPE nmni
#define NUM_AP 101
#define NUM_AQ 3

#include "is_kernel.cu"
