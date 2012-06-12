/*
 * kalman.h
 *
 *  Created on: 02-Mar-2009
 *      Author: alee
 */

#ifndef KALMAN_H_
#define KALMAN_H_

void kalman(float x_init, float* x, float* y, int T, float sigma_like,
		float sigma_step, float* ll);

//void kalman(float* x_init, float* x, float* y, int D, int T, float sigma_like,
//		float sigma_step, float* ll);
//
//void kalman(float* x_init, float* x, float* y, int D, int T, float* scale_step,
//		float* cov_step, float* scale_like, float* cov_like, float* ll);

void kalman(float* x_init, float* x, float* y, int Dx, int Dy, int T,
		float* scale_step, float* cov_step, float* scale_like, float* cov_like,
		float* ll);

void kalman(double* x_init, double* x, double* y, int Dx, int Dy, int T,
        double* scale_step, double* cov_step, double* scale_like, double* cov_like,
        double* ll);

#endif /* KALMAN_H_ */
