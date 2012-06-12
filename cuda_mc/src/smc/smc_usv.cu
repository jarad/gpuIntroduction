/*
 * smc_usv.cu
 *
 *  Created on: 09-Jul-2009
 *      Author: alee
 */

#include "func.h"
#include "usv.ch"

#define LIKELIHOOD usv_pdf
#define TYPE usv

#define NUM_AL 1

#include "smc_kernel.cu"
