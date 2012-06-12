/*
 * mc_mix_gauss_mu.cpp
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#include "func.h"
#include "mix_gauss_uniform.h"
#include "uniform.h"

#define TYPE mgmu_mv

#define PROPOSAL_H uniform_pdfh
#define TARGET_H mgu_mu_pdfh
#define LOG_PROPOSAL_H log_uniform_pdfh
#define LOG_TARGET_H log_mgu_mu_pdfh

#include "is_ref_mv.cpp"
