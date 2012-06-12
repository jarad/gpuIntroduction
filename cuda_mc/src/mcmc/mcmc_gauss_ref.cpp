/*
 * mcmc_gauss_ref.cpp
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#include "func.h"
#include "gauss.h"

#define TYPE n

#define TARGET_H gauss1_pdfh
#define LOG_TARGET_H log_gauss1_pdfh

#include "mcmc_ref.cpp"
