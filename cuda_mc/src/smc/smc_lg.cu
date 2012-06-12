/*
 * smc_lg.cu
 *
 *  Created on: 10-Mar-2009
 *      Author: alee
 */

#include "func.h"
#include "gauss.ch"

#define LIKELIHOOD gauss1_mean_pdf
#define TYPE lg

#define NUM_AL 100

#include "smc_kernel.cu"
