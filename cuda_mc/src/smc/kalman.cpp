/*
 * kalman.c
 *
 *  Created on: 02-Mar-2009
 *      Author: alee
 */

#include "kalman.h"
#include "gauss.h"
#include <stdio.h>
#include "matrix.h"

void kalman(float x_init, float* x, float* y, int T, float sigma_like,
		float sigma_step, float* ll) {
	float A, BB, C, DD;
	A = 1.0;
	BB = sigma_step * sigma_step;
	C = 1.0;
	DD = sigma_like * sigma_like;

	float mutt1;
	float sigmatt1;
	float S;
	float ytt1;

	float temp;

	float mutt = x_init;
	float sigmatt = 0;
	int i;

	double lld = 0;

	for (i = 0; i < T; i++) {
		mutt1 = A * mutt;
		sigmatt1 = A * sigmatt * A + BB;
		S = C * sigmatt1 * C + DD;
		ytt1 = C * mutt1;

		lld += log(gauss1_pdfh(y[i], ytt1, sqrtf(S)));

		temp = sigmatt1 * C / S;
		mutt = mutt1 + temp * (y[i] - ytt1);
		x[i] = mutt;
		sigmatt = sigmatt1 - (temp * C * sigmatt1);
	}

	*ll = (float) lld;

}

template <class T>
void kalman_impl(T* x_init, T* x, T* y, int Dx, int Dy, int total_time,
		T* scale_step, T* cov_step, T* scale_like, T* cov_like,
		T* ll) {

	T* A = scale_step;
	T* C = scale_like;
	T* BB = cov_step;
	T* DD = cov_like;

	T* mutt1 = (T*) malloc(Dx * sizeof(T));
	T* sigmatt1 = (T*) malloc(Dx * Dx * sizeof(T));
	T* S = (T*) malloc(Dy * Dy * sizeof(T));
	T* ytt1 = (T*) malloc(Dy * sizeof(T));

	T* temp_x = (T*) malloc(Dx * Dx * sizeof(T));
	T* temp_y = (T*) malloc(Dy * Dy * sizeof(T));

	T* mutt = (T*) malloc(Dx * sizeof(T));
	T* sigmatt = (T*) malloc(Dx * Dx * sizeof(T));

	T* Ct = (T*) malloc(Dx * Dy * sizeof(T));
	T* At = (T*) malloc(Dx * Dx * sizeof(T));
	matrix_transpose(A, At, Dx, Dx);
	matrix_transpose(C, Ct, Dy, Dx);

	T* Si = (T*) malloc(Dy * Dy * sizeof(T));

	T* stt1CtSi = (T*) malloc(Dx * Dy * sizeof(T));

	vector_set(mutt, x_init, Dx);
	matrix_zero(sigmatt, Dx, Dx);

	int i;

	double lld = 0;

	for (i = 0; i < total_time; i++) {

		matrix_times(A, mutt, mutt1, Dx, Dx, Dx, 1);
		//		mutt1 = A * mutt;

		matrix_times(A, sigmatt, temp_x, Dx, Dx, Dx, Dx);
		matrix_times(temp_x, At, sigmatt1, Dx, Dx, Dx, Dx);
		matrix_add(sigmatt1, BB, sigmatt1, Dx, Dx);
		//		sigmatt1 = A * sigmatt * A' + BB;

		matrix_times(C, sigmatt1, temp_y, Dy, Dx, Dx, Dx);
		matrix_times(temp_y, Ct, S, Dy, Dx, Dx, Dy);
		matrix_add(S, DD, S, Dy, Dy);
		//		S = C * sigmatt1 * C' + DD;

		matrix_times(C, mutt1, ytt1, Dy, Dx, Dx, 1);
		//		ytt1 = C * mutt1;

		lld += log(gauss_pdfh(vector_get(y, Dy, i), ytt1, S, Dy));

		matrix_inverse_pd(S, Si, Dy);

		matrix_times(sigmatt1, Ct, temp_y, Dx, Dx, Dx, Dy);
		matrix_times(temp_y, Si, stt1CtSi, Dx, Dy, Dy, Dy);

		matrix_minus(vector_get(y, Dy, i), ytt1, temp_y, Dy, 1);
		matrix_times(stt1CtSi, temp_y, mutt, Dx, Dy, Dy, 1);
		vector_add(mutt1, mutt, mutt, Dx);
		//		mutt = mutt1 + stt1CtSi * (y[i] - ytt1);

		vector_set(vector_get(x, Dx, i), mutt, Dx);
		//		x[i] = mutt;


		matrix_times(stt1CtSi, C, temp_y, Dx, Dy, Dy, Dx);
		matrix_times(temp_y, sigmatt1, sigmatt, Dx, Dx, Dx, Dx);
		matrix_minus(sigmatt1, sigmatt, sigmatt, Dx, Dx);
		//		sigmatt = sigmatt1 - (temp * C * sigmatt1);
	}

	*ll = (T) lld;

	free(mutt1);
	free(sigmatt1);
	free(S);
	free(ytt1);

	free(temp_x);
	free(temp_y);
	free(mutt);
	free(sigmatt);

	free(Ct);
	free(At);

	free(stt1CtSi);

}

void kalman(float* x_init, float* x, float* y, int Dx, int Dy, int T,
        float* scale_step, float* cov_step, float* scale_like, float* cov_like,
        float* ll) {
    kalman_impl(x_init, x, y, Dx, Dy, T, scale_step, cov_step, scale_like, cov_like, ll);
}

void kalman(double* x_init, double* x, double* y, int Dx, int Dy, int T,
        double* scale_step, double* cov_step, double* scale_like, double* cov_like,
        double* ll) {
    kalman_impl(x_init, x, y, Dx, Dy, T, scale_step, cov_step, scale_like, cov_like, ll);
}

//void kalman(double* x_init, double* x, T* y, int Dx, int Dy, int T,
//        double* scale_step, double* cov_step, float* scale_like, float* cov_like,
//        double* ll) {
//
//	double* A = (double*) malloc(Dx * Dx * sizeof(double));
//	double* C = (double*) malloc(Dy * Dx * sizeof(double));
//	double* BB = (double*) malloc(Dx * Dx * sizeof(double));
//	double* DD = (double*) malloc(Dy * Dy * sizeof(double));
//
//	int i;
//
//	double* x_init_double = (double*) malloc(Dx * sizeof(double));
//	for (i = 0; i < Dx; i++) {
//		x_init_double[i] = x_init[i];
//	}
//
//	for (i = 0; i < Dx * Dx; i++) {
//		A[i] = scale_step[i];
//		BB[i] = cov_step[i];
//	}
//	for (i = 0; i < Dy * Dx; i++) {
//		C[i] = scale_like[i];
//	}
//	for (i = 0; i < Dy * Dy; i++) {
//		DD[i] = cov_like[i];
//	}
//
//	double* mutt1 = (double*) malloc(Dx * sizeof(double));
//	double* sigmatt1 = (double*) malloc(Dx * Dx * sizeof(double));
//	double* S = (double*) malloc(Dy * Dy * sizeof(double));
//	double* ytt1 = (double*) malloc(Dy * sizeof(double));
//
//	double* temp_x = (double*) malloc(Dx * Dx * sizeof(double));
//	double* temp_y = (double*) malloc(Dy * Dy * sizeof(double));
//
//	double* mutt = (double*) malloc(Dx * sizeof(double));
//	double* sigmatt = (double*) malloc(Dx * Dx * sizeof(double));
//
//	double* Ct = (double*) malloc(Dx * Dy * sizeof(double));
//	double* At = (double*) malloc(Dx * Dx * sizeof(double));
//	matrix_transpose(A, At, Dx, Dx);
//	matrix_transpose(C, Ct, Dy, Dx);
//
//	double* Si = (double*) malloc(Dy * Dy * sizeof(double));
//
//	double* stt1CtSi = (double*) malloc(Dx * Dy * sizeof(double));
//
//	vector_set(mutt, x_init_double, Dx);
//	matrix_zero(sigmatt, Dx, Dx);
//
//	int j;
//
//	double lld = 0;
//
//	float* ytt1f = (float*) malloc(Dy * sizeof(float));
//	float* muttf = (float*) malloc(Dx * sizeof(float));
//	float* Sf = (float*) malloc(Dy * Dy * sizeof(float));
//	double* ytd = (double*) malloc(Dy * sizeof(double));
//
//	for (i = 0; i < T; i++) {
//
//		matrix_times(A, mutt, mutt1, Dx, Dx, Dx, 1);
//		//		mutt1 = A * mutt;
//
//		matrix_times(A, sigmatt, temp_x, Dx, Dx, Dx, Dx);
//		matrix_times(temp_x, At, sigmatt1, Dx, Dx, Dx, Dx);
//		matrix_add(sigmatt1, BB, sigmatt1, Dx, Dx);
//		//		sigmatt1 = A * sigmatt * A' + BB;
//
//		matrix_times(C, sigmatt1, temp_y, Dy, Dx, Dx, Dx);
//		matrix_times(temp_y, Ct, S, Dy, Dx, Dx, Dy);
//		matrix_add(S, DD, S, Dy, Dy);
//		//		S = C * sigmatt1 * C' + DD;
//
//		matrix_times(C, mutt1, ytt1, Dy, Dx, Dx, 1);
//		//		ytt1 = C * mutt1;
//
//		for (j = 0; j < Dy; j++) {
//			ytt1f[j] = (float) ytt1[j];
//		}
//		for (j = 0; j < Dy * Dy; j++) {
//			Sf[j] = (float) S[j];
//		}
//
//		lld += log(gauss_pdfh(vector_get(y, Dy, i), ytt1f, Sf, Dy));
//
//		matrix_inverse_pd(S, Si, Dy);
//
//		matrix_times(sigmatt1, Ct, temp_y, Dx, Dx, Dx, Dy);
//		matrix_times(temp_y, Si, stt1CtSi, Dx, Dy, Dy, Dy);
//
//		for (j = 0; j < Dy; j++) {
//			ytd[j] = vector_get(y, Dy, i)[j];
//		}
//
//		matrix_minus(ytd, ytt1, temp_y, Dy, 1);
//		matrix_times(stt1CtSi, temp_y, mutt, Dx, Dy, Dy, 1);
//		vector_add(mutt1, mutt, mutt, Dx);
//		//		mutt = mutt1 + stt1CtSi * (y[i] - ytt1);
//
//		for (j = 0; j < Dx; j++) {
//			muttf[j] = (float) mutt[j];
//		}
//		vector_set(vector_get(x, Dx, i), muttf, Dx);
//		//		x[i] = mutt;
//
//
//		matrix_times(stt1CtSi, C, temp_y, Dx, Dy, Dy, Dx);
//		matrix_times(temp_y, sigmatt1, sigmatt, Dx, Dx, Dx, Dx);
//		matrix_minus(sigmatt1, sigmatt, sigmatt, Dx, Dx);
//		//		sigmatt = sigmatt1 - (temp * C * sigmatt1);
//	}
//
//	*ll = (float) lld;
//
//	free(mutt1);
//	free(sigmatt1);
//	free(S);
//	free(ytt1);
//
//	free(temp_x);
//	free(temp_y);
//	free(mutt);
//	free(sigmatt);
//
//	free(Ct);
//	free(At);
//
//	free(stt1CtSi);
//
//}
