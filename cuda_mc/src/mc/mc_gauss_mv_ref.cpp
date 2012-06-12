/*
 * mc_gauss_mv_ref.cpp
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#include "func.h"
#include "gauss.h"

#define TYPE nn_mv

#define PROPOSAL_H gauss_pdfh
#define TARGET_H gauss_pdfh
#define LOG_PROPOSAL_H log_gauss_pdfh
#define LOG_TARGET_H log_gauss_pdfh

#include "is_ref_mv.cpp"
