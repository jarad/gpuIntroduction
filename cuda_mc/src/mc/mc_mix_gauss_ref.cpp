/*
 * mc_mix_gauss_ref.c
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */

#include "func.h"
#include "mix_gauss.h"
#include "gauss.h"

#define TYPE nmni

#define PROPOSAL_H gauss1_pdfh
#define TARGET_H mix_gauss1_pdfh

#include "is_ref.cpp"
