/*
 * mcmc_gauss.cu
 *
 *  Created on: 04-Feb-2009
 *      Author: alee
 */

#include "mcmc_gauss.h"

#include "func.h"
#include "gauss.ch"

#define TARGET gauss1_pdf
#define LOG_TARGET log_gauss1_pdf
#define TYPE n
#define NUM_AP 3

#include "mcmc_kernel.cu"
