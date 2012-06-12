/*
 * mc_gauss_ref.c
 *
 *  Created on: 08-Apr-2009
 *      Author: alee
 */


#include "func.h"
#include "gauss.h"

#define PROPOSAL_H gauss1_pdfh
#define TARGET_H gauss1_pdfh

#define TYPE nn
#include "is_ref.cpp"
