/*
 * mcmc_gauss_mv_ref.cpp
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#include "func.h"
#include "gauss.h"

#define TYPE n_mv

#define TARGET_H gauss_pdfh
#define LOG_TARGET_H log_gauss_pdfh

#include "mcmc_ref_mv.cpp"
