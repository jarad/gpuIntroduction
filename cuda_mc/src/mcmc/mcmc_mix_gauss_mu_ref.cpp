/*
 * mcmc_mix_gauss_mu_ref.cpp
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#include "func.h"
#include "mix_gauss_uniform.h"

#define TYPE mgumu_mv

#define TARGET_H mgu_mu_pdfh
#define LOG_TARGET_H log_mgu_mu_pdfh

#include "mcmc_ref_mv.cpp"

