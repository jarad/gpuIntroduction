/*
 * mcmc_mix_gauss_ref.cpp
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#include "func.h"
#include "mix_gauss.h"

#define TYPE mn

#define TARGET_H mix_gauss1_pdfh
#define LOG_TARGET_H log_mix_gauss1_pdfh

#include "mcmc_ref.cpp"
