/*
 * mcmc_mix_gauss.cu
 *
 *  Created on: 04-Feb-2009
 *      Author: alee
 */

#include "mcmc_mix_gauss.h"

#include "func.h"
#include "mix_gauss.ch"

#define TARGET mix_gauss1_pdf
#define LOG_TARGET log_mix_gauss1_pdf
#define TYPE mn
#define NUM_AP 101 // can do 25d

#include "mcmc_kernel.cu"
