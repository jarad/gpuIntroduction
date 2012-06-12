/*
 * smcs_mix_gauss_mu_ref.cpp
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#include "func.h"
#include "mix_gauss_uniform.h"
#include "uniform.h"

#define TYPE mgumu_mv

//#define TARGET1_H uniform_pdfh
#define LOG_TARGET1_H log_uniform_pdfh
//#define TARGET2_H mgu_mu_pdfh
#define LOG_TARGET2_H log_mgu_mu_pdfh

#include "smcs_ref_mv.cpp"
