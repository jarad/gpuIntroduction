/*
 * mcgauss.cu
 *
 *  Created on: 2-Feb-2009
 *      Author: Owner
 */

#include "mc_gauss.h"

#include "func.h"
#include "gauss.ch"

#define PROPOSAL gauss1_pdf
#define TARGET gauss1_pdf
#define TYPE nn
#define NUM_AP 3
#define NUM_AQ 3

#include "is_kernel.cu"


